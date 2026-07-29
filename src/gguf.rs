//! GGUF container parsing (spec versions 2 and 3).
//!
//! Layout: magic, version, tensor count, metadata count, then the metadata
//! key/value table, then the tensor info table, then padding to
//! `general.alignment`, then the tensor data blob. Tensor offsets are relative
//! to the start of that blob, not the file.
//!
//! Everything here treats the file as untrusted: a truncated download and a
//! hostile file look identical, so every read is bounds-checked and no
//! allocation is sized from a number the file supplied.

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::io::{Error, ErrorKind, Result};
use std::path::Path;

use crate::map::Mmap;

const MAGIC: [u8; 4] = *b"GGUF";
const DEFAULT_ALIGNMENT: u64 = 32;
const MAX_STRING: u64 = 1 << 26;
const MAX_DIMS: usize = 4;

fn bad(msg: impl Into<String>) -> Error {
    Error::new(ErrorKind::InvalidData, msg.into())
}

// ---------------------------------------------------------------- ggml types

/// Tensor element encoding. `block_size` elements share one quantization
/// block of `type_size` bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[allow(non_camel_case_types)]
pub enum GgmlType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2_K,
    Q3_K,
    Q4_K,
    Q5_K,
    Q6_K,
    Q8_K,
    I8,
    I16,
    I32,
    I64,
    F64,
    BF16,
    /// A type we can name but not size — the IQ family, mostly. Inspecting a
    /// model that uses one works; reading its bytes does not.
    Other(u32),
}

impl GgmlType {
    pub fn from_id(id: u32) -> Self {
        match id {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2_K,
            11 => Self::Q3_K,
            12 => Self::Q4_K,
            13 => Self::Q5_K,
            14 => Self::Q6_K,
            15 => Self::Q8_K,
            24 => Self::I8,
            25 => Self::I16,
            26 => Self::I32,
            27 => Self::I64,
            28 => Self::F64,
            30 => Self::BF16,
            other => Self::Other(other),
        }
    }

    /// `(block_size, type_size)`. `None` for types we cannot size.
    pub fn layout(self) -> Option<(u64, u64)> {
        Some(match self {
            Self::F32 | Self::I32 => (1, 4),
            Self::F16 | Self::BF16 | Self::I16 => (1, 2),
            Self::I8 => (1, 1),
            Self::F64 | Self::I64 => (1, 8),
            Self::Q4_0 => (32, 18),
            Self::Q4_1 => (32, 20),
            Self::Q5_0 => (32, 22),
            Self::Q5_1 => (32, 24),
            Self::Q8_0 => (32, 34),
            Self::Q8_1 => (32, 36),
            Self::Q2_K => (256, 84),
            Self::Q3_K => (256, 110),
            Self::Q4_K => (256, 144),
            Self::Q5_K => (256, 176),
            Self::Q6_K => (256, 210),
            Self::Q8_K => (256, 292),
            Self::Other(_) => return None,
        })
    }

    pub fn name(self) -> String {
        match self {
            Self::Other(id) => format!("type{id}"),
            other => format!("{other:?}"),
        }
    }
}

// -------------------------------------------------------------------- values

/// A metadata value. GGUF's 13 types, with arrays holding a homogeneous list.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(Vec<Value>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl Value {
    /// Any integer type widened. `None` for non-integers.
    pub fn as_u64(&self) -> Option<u64> {
        Some(match self {
            Self::U8(v) => *v as u64,
            Self::U16(v) => *v as u64,
            Self::U32(v) => *v as u64,
            Self::U64(v) => *v,
            Self::I8(v) if *v >= 0 => *v as u64,
            Self::I16(v) if *v >= 0 => *v as u64,
            Self::I32(v) if *v >= 0 => *v as u64,
            Self::I64(v) if *v >= 0 => *v as u64,
            Self::Bool(v) => *v as u64,
            _ => return None,
        })
    }

    pub fn as_f32(&self) -> Option<f32> {
        Some(match self {
            Self::F32(v) => *v,
            Self::F64(v) => *v as f32,
            Self::I8(v) => *v as f32,
            Self::I16(v) => *v as f32,
            Self::I32(v) => *v as f32,
            Self::I64(v) => *v as f32,
            other => other.as_u64()? as f32,
        })
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Self::Array(v) => Some(v),
            _ => None,
        }
    }

    /// Borrowed view of a string array, e.g. `tokenizer.ggml.tokens`.
    pub fn as_strings(&self) -> Option<Vec<&str>> {
        self.as_array()?.iter().map(Value::as_str).collect()
    }

    pub fn as_f32s(&self) -> Option<Vec<f32>> {
        self.as_array()?.iter().map(Value::as_f32).collect()
    }

    /// The GGUF wire tag for this value. Inverse of the match in
    /// `Cursor::value`, and what a writer needs.
    pub fn type_id(&self) -> u32 {
        match self {
            Self::U8(_) => 0,
            Self::I8(_) => 1,
            Self::U16(_) => 2,
            Self::I16(_) => 3,
            Self::U32(_) => 4,
            Self::I32(_) => 5,
            Self::F32(_) => 6,
            Self::Bool(_) => 7,
            Self::String(_) => 8,
            Self::Array(_) => 9,
            Self::U64(_) => 10,
            Self::I64(_) => 11,
            Self::F64(_) => 12,
        }
    }
}

impl fmt::Display for Value {
    /// Long arrays are summarized — a vocab is 128k entries and nobody wants
    /// it printed.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::U8(v) => write!(f, "{v}"),
            Self::I8(v) => write!(f, "{v}"),
            Self::U16(v) => write!(f, "{v}"),
            Self::I16(v) => write!(f, "{v}"),
            Self::U32(v) => write!(f, "{v}"),
            Self::I32(v) => write!(f, "{v}"),
            Self::U64(v) => write!(f, "{v}"),
            Self::I64(v) => write!(f, "{v}"),
            Self::F32(v) => write!(f, "{v}"),
            Self::F64(v) => write!(f, "{v}"),
            Self::Bool(v) => write!(f, "{v}"),
            Self::String(s) if s.len() > 120 => write!(f, "{:.117}...", s),
            Self::String(s) => write!(f, "{s}"),
            Self::Array(items) => {
                let shown = items.len().min(6);
                write!(f, "[")?;
                for (i, item) in items[..shown].iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{item}")?;
                }
                if items.len() > shown {
                    write!(f, ", ... {} total", items.len())?;
                }
                write!(f, "]")
            }
        }
    }
}

// -------------------------------------------------------------------- cursor

struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

macro_rules! read_prim {
    ($name:ident, $ty:ty) => {
        fn $name(&mut self) -> Result<$ty> {
            const N: usize = std::mem::size_of::<$ty>();
            let bytes: [u8; N] = self.take(N)?.try_into().unwrap();
            Ok(<$ty>::from_le_bytes(bytes))
        }
    };
}

impl<'a> Cursor<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or_else(|| bad("offset overflow"))?;
        let out = self
            .buf
            .get(self.pos..end)
            .ok_or_else(|| bad("unexpected end of file"))?;
        self.pos = end;
        Ok(out)
    }

    read_prim!(u8_, u8);
    read_prim!(i8_, i8);
    read_prim!(u16_, u16);
    read_prim!(i16_, i16);
    read_prim!(u32_, u32);
    read_prim!(i32_, i32);
    read_prim!(u64_, u64);
    read_prim!(i64_, i64);
    read_prim!(f32_, f32);
    read_prim!(f64_, f64);

    fn string(&mut self) -> Result<String> {
        let n = self.u64_()?;
        if n > MAX_STRING {
            return Err(bad(format!("implausible string length {n}")));
        }
        let bytes = self.take(n as usize)?;
        String::from_utf8(bytes.to_vec()).map_err(|_| bad("metadata string is not utf-8"))
    }

    fn value(&mut self, ty: u32, depth: u32) -> Result<Value> {
        Ok(match ty {
            0 => Value::U8(self.u8_()?),
            1 => Value::I8(self.i8_()?),
            2 => Value::U16(self.u16_()?),
            3 => Value::I16(self.i16_()?),
            4 => Value::U32(self.u32_()?),
            5 => Value::I32(self.i32_()?),
            6 => Value::F32(self.f32_()?),
            7 => Value::Bool(self.u8_()? != 0),
            8 => Value::String(self.string()?),
            9 => {
                if depth > 0 {
                    return Err(bad("nested arrays are not valid GGUF"));
                }
                let inner = self.u32_()?;
                let count = self.u64_()?;
                // Deliberately no `with_capacity`: `count` is attacker-controlled
                // and `take` is what bounds each element.
                let mut items = Vec::new();
                for _ in 0..count {
                    items.push(self.value(inner, depth + 1)?);
                }
                Value::Array(items)
            }
            10 => Value::U64(self.u64_()?),
            11 => Value::I64(self.i64_()?),
            12 => Value::F64(self.f64_()?),
            other => return Err(bad(format!("unknown metadata value type {other}"))),
        })
    }
}

// ------------------------------------------------------------------- tensors

#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub name: String,
    /// GGUF stores dimensions fastest-varying first.
    pub dims: Vec<u64>,
    pub ty: GgmlType,
    /// Byte offset from the start of the tensor data blob.
    pub offset: u64,
    /// `None` when the type has no known layout.
    pub byte_len: Option<u64>,
}

impl TensorInfo {
    pub fn elements(&self) -> u64 {
        self.dims.iter().product()
    }

    pub fn shape(&self) -> String {
        let dims: Vec<String> = self.dims.iter().map(|d| d.to_string()).collect();
        dims.join("x")
    }
}

// ---------------------------------------------------------------------- file

pub struct Gguf {
    map: Mmap,
    pub version: u32,
    pub alignment: u64,
    pub meta: BTreeMap<String, Value>,
    pub tensors: Vec<TensorInfo>,
    /// Byte offset of the tensor data blob within the file.
    pub data_offset: u64,
    index: HashMap<String, usize>,
}

impl fmt::Debug for Gguf {
    /// Deliberately not the full metadata map — a vocab array would bury the
    /// three fields anyone actually wants in a panic message.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Gguf")
            .field("version", &self.version)
            .field("arch", &self.arch())
            .field("tensors", &self.tensors.len())
            .field("bytes", &self.map.len())
            .finish()
    }
}

impl Gguf {
    pub fn open(path: &Path) -> Result<Self> {
        let map = Mmap::open(path)?;
        let file_len = map.len() as u64;
        let mut cur = Cursor::new(map.as_slice());

        if cur.take(4)? != MAGIC {
            return Err(bad("not a GGUF file (bad magic)"));
        }
        let version = cur.u32_()?;
        if !(2..=3).contains(&version) {
            return Err(bad(format!(
                "GGUF version {version} unsupported (need 2 or 3)"
            )));
        }
        let tensor_count = cur.u64_()?;
        let meta_count = cur.u64_()?;

        let mut meta = BTreeMap::new();
        for _ in 0..meta_count {
            let key = cur.string()?;
            let ty = cur.u32_()?;
            let value = cur.value(ty, 0)?;
            meta.insert(key, value);
        }

        let alignment = match meta.get("general.alignment") {
            Some(v) => v
                .as_u64()
                .ok_or_else(|| bad("general.alignment not an int"))?,
            None => DEFAULT_ALIGNMENT,
        };
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(bad(format!("alignment {alignment} is not a power of two")));
        }

        let mut tensors = Vec::new();
        let mut index = HashMap::new();
        for _ in 0..tensor_count {
            let name = cur.string()?;
            let n_dims = cur.u32_()? as usize;
            if n_dims == 0 || n_dims > MAX_DIMS {
                return Err(bad(format!("tensor {name} has {n_dims} dimensions")));
            }
            let mut dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                dims.push(cur.u64_()?);
            }
            let ty = GgmlType::from_id(cur.u32_()?);
            let offset = cur.u64_()?;

            let elements: u64 = dims
                .iter()
                .try_fold(1u64, |acc, d| acc.checked_mul(*d))
                .ok_or_else(|| bad(format!("tensor {name} element count overflows")))?;

            let byte_len = match ty.layout() {
                Some((block, size)) => {
                    if elements % block != 0 {
                        return Err(bad(format!(
                            "tensor {name}: {elements} elements is not a multiple of {} block {block}",
                            ty.name()
                        )));
                    }
                    Some(
                        (elements / block)
                            .checked_mul(size)
                            .ok_or_else(|| bad(format!("tensor {name} byte length overflows")))?,
                    )
                }
                None => None,
            };

            if index.insert(name.clone(), tensors.len()).is_some() {
                return Err(bad(format!("duplicate tensor name {name}")));
            }
            tensors.push(TensorInfo {
                name,
                dims,
                ty,
                offset,
                byte_len,
            });
        }

        let data_offset =
            align_up(cur.pos as u64, alignment).ok_or_else(|| bad("data offset overflows"))?;
        if data_offset > file_len {
            return Err(bad("file ends before the tensor data begins"));
        }

        // Every sized tensor must fit inside the file. Catches truncated
        // downloads before a kernel reads past the mapping.
        for t in &tensors {
            let Some(len) = t.byte_len else { continue };
            let end = data_offset
                .checked_add(t.offset)
                .and_then(|s| s.checked_add(len))
                .ok_or_else(|| bad(format!("tensor {} extent overflows", t.name)))?;
            if end > file_len {
                return Err(bad(format!(
                    "tensor {} runs {} bytes past end of file (truncated download?)",
                    t.name,
                    end - file_len
                )));
            }
        }

        Ok(Self {
            map,
            version,
            alignment,
            meta,
            tensors,
            data_offset,
            index,
        })
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        self.meta.get(key)
    }

    pub fn meta_str(&self, key: &str) -> Option<&str> {
        self.get(key)?.as_str()
    }

    pub fn meta_u64(&self, key: &str) -> Option<u64> {
        self.get(key)?.as_u64()
    }

    pub fn meta_f32(&self, key: &str) -> Option<f32> {
        self.get(key)?.as_f32()
    }

    /// `llama`, `qwen2`, `gemma`, ... Drives which weight names to expect.
    pub fn arch(&self) -> Option<&str> {
        self.meta_str("general.architecture")
    }

    /// Architecture-scoped key: `arch_key("block_count")` reads
    /// `llama.block_count` for a llama model.
    pub fn arch_u64(&self, suffix: &str) -> Option<u64> {
        let arch = self.arch()?;
        self.meta_u64(&format!("{arch}.{suffix}"))
    }

    pub fn arch_f32(&self, suffix: &str) -> Option<f32> {
        let arch = self.arch()?;
        self.meta_f32(&format!("{arch}.{suffix}"))
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        Some(&self.tensors[*self.index.get(name)?])
    }

    /// Raw bytes of a tensor, straight out of the mapping — no copy.
    pub fn tensor_bytes(&self, t: &TensorInfo) -> Result<&[u8]> {
        let len = t.byte_len.ok_or_else(|| {
            bad(format!(
                "tensor {} has unsupported type {}",
                t.name,
                t.ty.name()
            ))
        })?;
        let start = (self.data_offset + t.offset) as usize;
        let end = start + len as usize;
        self.map
            .as_slice()
            .get(start..end)
            .ok_or_else(|| bad(format!("tensor {} out of bounds", t.name)))
    }

    pub fn file_len(&self) -> u64 {
        self.map.len() as u64
    }

    /// Bytes per quantization type, largest first. This is what tells you at a
    /// glance whether a "Q4_K_M" file is really mostly Q4_K.
    pub fn quant_histogram(&self) -> Vec<(GgmlType, u64, u64)> {
        let mut by_type: HashMap<GgmlType, (u64, u64)> = HashMap::new();
        for t in &self.tensors {
            let entry = by_type.entry(t.ty).or_insert((0, 0));
            entry.0 += 1;
            entry.1 += t.byte_len.unwrap_or(0);
        }
        let mut out: Vec<(GgmlType, u64, u64)> = by_type
            .into_iter()
            .map(|(ty, (count, bytes))| (ty, count, bytes))
            .collect();
        out.sort_by_key(|(_, _, bytes)| std::cmp::Reverse(*bytes));
        out
    }
}

pub(crate) fn align_up(value: u64, alignment: u64) -> Option<u64> {
    let rem = value % alignment;
    if rem == 0 {
        Some(value)
    } else {
        value.checked_add(alignment - rem)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alignment_rounds_up() {
        assert_eq!(align_up(0, 32), Some(0));
        assert_eq!(align_up(1, 32), Some(32));
        assert_eq!(align_up(32, 32), Some(32));
        assert_eq!(align_up(33, 32), Some(64));
        assert_eq!(align_up(u64::MAX, 32), None);
    }

    #[test]
    fn q4k_block_layout() {
        // 256 elements per 144-byte super-block: the ratio that makes a 1B
        // model ~700 MB instead of 4 GB.
        assert_eq!(GgmlType::Q4_K.layout(), Some((256, 144)));
        assert_eq!(GgmlType::from_id(12), GgmlType::Q4_K);
        assert_eq!(GgmlType::from_id(999), GgmlType::Other(999));
        assert_eq!(GgmlType::Other(999).layout(), None);
    }

    #[test]
    fn value_widening() {
        assert_eq!(Value::U32(7).as_u64(), Some(7));
        assert_eq!(Value::I32(-1).as_u64(), None);
        assert_eq!(Value::String("x".into()).as_u64(), None);
    }
}

//! Minimal GGUF writer, for fixtures.
//!
//! A parser you can only test against an 800 MB download is a parser nobody
//! tests. This writes small valid files instead, so the round trip runs in CI
//! with no network.

use std::fs;
use std::io::Result;
use std::path::Path;

use crate::gguf::{align_up, GgmlType, Value};

const ALIGNMENT: u64 = 32;

#[derive(Default)]
pub struct Builder {
    meta: Vec<(String, Value)>,
    tensors: Vec<(String, Vec<u64>, GgmlType, Vec<u8>)>,
}

impl Builder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn meta(mut self, key: &str, value: Value) -> Self {
        self.meta.push((key.to_string(), value));
        self
    }

    /// Panics if `data` does not match the tensor's block layout — this is a
    /// test helper, and a fixture that lies is worse than one that crashes.
    pub fn tensor(mut self, name: &str, dims: &[u64], ty: GgmlType, data: Vec<u8>) -> Self {
        let elements: u64 = dims.iter().product();
        let (block, size) = ty.layout().expect("tensor type has no known layout");
        assert_eq!(
            elements % block,
            0,
            "{name}: elements must fill whole blocks"
        );
        assert_eq!(
            data.len() as u64,
            elements / block * size,
            "{name}: data length does not match dims and type"
        );
        self.tensors
            .push((name.to_string(), dims.to_vec(), ty, data));
        self
    }

    pub fn build(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"GGUF");
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&(self.tensors.len() as u64).to_le_bytes());
        out.extend_from_slice(&(self.meta.len() as u64).to_le_bytes());

        for (key, value) in &self.meta {
            write_string(&mut out, key);
            out.extend_from_slice(&value.type_id().to_le_bytes());
            write_value(&mut out, value);
        }

        // Data offsets are relative to the blob, and each tensor starts aligned.
        let mut offset = 0u64;
        for (name, dims, ty, data) in &self.tensors {
            write_string(&mut out, name);
            out.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for d in dims {
                out.extend_from_slice(&d.to_le_bytes());
            }
            out.extend_from_slice(&ty_id(*ty).to_le_bytes());
            out.extend_from_slice(&offset.to_le_bytes());
            offset = align_up(offset + data.len() as u64, ALIGNMENT).expect("offset overflow");
        }

        pad_to(&mut out, ALIGNMENT);
        let data_start = out.len() as u64;
        for (_, _, _, data) in &self.tensors {
            out.extend_from_slice(data);
            let written = out.len() as u64 - data_start;
            let target = align_up(written, ALIGNMENT).expect("padding overflow");
            out.resize((data_start + target) as usize, 0);
        }
        out
    }

    pub fn write(&self, path: &Path) -> Result<()> {
        fs::write(path, self.build())
    }
}

fn write_string(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

fn write_value(out: &mut Vec<u8>, value: &Value) {
    match value {
        Value::U8(v) => out.push(*v),
        Value::I8(v) => out.extend_from_slice(&v.to_le_bytes()),
        Value::U16(v) => out.extend_from_slice(&v.to_le_bytes()),
        Value::I16(v) => out.extend_from_slice(&v.to_le_bytes()),
        Value::U32(v) => out.extend_from_slice(&v.to_le_bytes()),
        Value::I32(v) => out.extend_from_slice(&v.to_le_bytes()),
        Value::F32(v) => out.extend_from_slice(&v.to_le_bytes()),
        Value::U64(v) => out.extend_from_slice(&v.to_le_bytes()),
        Value::I64(v) => out.extend_from_slice(&v.to_le_bytes()),
        Value::F64(v) => out.extend_from_slice(&v.to_le_bytes()),
        Value::Bool(v) => out.push(*v as u8),
        Value::String(s) => write_string(out, s),
        Value::Array(items) => {
            let inner = items.first().map(Value::type_id).unwrap_or(4);
            out.extend_from_slice(&inner.to_le_bytes());
            out.extend_from_slice(&(items.len() as u64).to_le_bytes());
            for item in items {
                write_value(out, item);
            }
        }
    }
}

fn pad_to(out: &mut Vec<u8>, alignment: u64) {
    let target = align_up(out.len() as u64, alignment).expect("padding overflow");
    out.resize(target as usize, 0);
}

fn ty_id(ty: GgmlType) -> u32 {
    match ty {
        GgmlType::F32 => 0,
        GgmlType::F16 => 1,
        GgmlType::Q4_0 => 2,
        GgmlType::Q4_1 => 3,
        GgmlType::Q5_0 => 6,
        GgmlType::Q5_1 => 7,
        GgmlType::Q8_0 => 8,
        GgmlType::Q8_1 => 9,
        GgmlType::Q2_K => 10,
        GgmlType::Q3_K => 11,
        GgmlType::Q4_K => 12,
        GgmlType::Q5_K => 13,
        GgmlType::Q6_K => 14,
        GgmlType::Q8_K => 15,
        GgmlType::I8 => 24,
        GgmlType::I16 => 25,
        GgmlType::I32 => 26,
        GgmlType::I64 => 27,
        GgmlType::F64 => 28,
        GgmlType::BF16 => 30,
        GgmlType::Other(id) => id,
    }
}

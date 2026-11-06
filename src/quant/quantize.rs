//! GGUF weight quantization and rewriting.

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

use crate::gguf::{align_up, GgmlType, Gguf, Value};

/// Quantize supported floating-point tensors and copy other tensor encodings
/// without decoding them. Tensor order, dimensions, and metadata are retained.
pub fn quantize_file(src: &Path, dst: &Path, target: GgmlType) -> io::Result<()> {
    if !matches!(target, GgmlType::Q4_0 | GgmlType::Q8_0 | GgmlType::Q4_K) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "target must be Q4_0, Q8_0, or Q4_K",
        ));
    }
    let model = Gguf::open(src)?;
    let mut output = BufWriter::new(File::create(dst)?);
    let mut converted = Vec::with_capacity(model.tensors.len());
    for tensor in &model.tensors {
        let new_ty = if matches!(tensor.ty, GgmlType::F32 | GgmlType::F16) {
            target
        } else {
            tensor.ty
        };
        let bytes = if new_ty == tensor.ty {
            model.tensor_bytes(tensor)?.to_vec()
        } else {
            let raw = model.tensor_bytes(tensor)?;
            let values = crate::quant::dequantize_to_vec(
                tensor.ty,
                raw,
                usize::try_from(tensor.elements()).map_err(|_| invalid("tensor too large"))?,
            )?;
            encode(&values, target)?
        };
        converted.push((new_ty, bytes));
    }

    let mut header = Vec::new();
    header.extend_from_slice(b"GGUF");
    header.extend_from_slice(&model.version.to_le_bytes());
    header.extend_from_slice(&(model.tensors.len() as u64).to_le_bytes());
    header.extend_from_slice(&(model.meta.len() as u64).to_le_bytes());
    for (key, value) in &model.meta {
        write_string(&mut header, key)?;
        header.extend_from_slice(&value.type_id().to_le_bytes());
        write_value(&mut header, value)?;
    }
    let mut offsets = Vec::with_capacity(model.tensors.len());
    let mut offset = 0u64;
    for (tensor, (ty, bytes)) in model.tensors.iter().zip(&converted) {
        offsets.push(offset);
        write_string(&mut header, &tensor.name)?;
        header.extend_from_slice(&(tensor.dims.len() as u32).to_le_bytes());
        for dim in &tensor.dims {
            header.extend_from_slice(&dim.to_le_bytes());
        }
        header.extend_from_slice(&type_id(*ty).to_le_bytes());
        header.extend_from_slice(&offset.to_le_bytes());
        offset = align_up(
            offset
                .checked_add(bytes.len() as u64)
                .ok_or_else(|| invalid("output size overflow"))?,
            model.alignment,
        )
        .ok_or_else(|| invalid("output offset overflow"))?;
    }
    pad_vec(&mut header, model.alignment)?;
    output.write_all(&header)?;
    for (i, (_, bytes)) in converted.into_iter().enumerate() {
        output.write_all(&bytes)?;
        let next = align_up(
            offsets[i]
                .checked_add(bytes.len() as u64)
                .ok_or_else(|| invalid("output size overflow"))?,
            model.alignment,
        )
        .ok_or_else(|| invalid("output offset overflow"))?;
        let zeros = [0u8; 4096];
        let mut remaining = next - offsets[i] - bytes.len() as u64;
        while remaining > 0 {
            let n = remaining.min(zeros.len() as u64) as usize;
            output.write_all(&zeros[..n])?;
            remaining -= n as u64;
        }
    }
    output.flush()
}

fn encode(values: &[f32], target: GgmlType) -> io::Result<Vec<u8>> {
    let (block, _) = target.layout().expect("supported output format");
    let block = block as usize;
    if values.len() % block != 0 {
        return Err(invalid("tensor dimensions do not fill quantization blocks"));
    }
    let mut out = Vec::with_capacity(values.len() / block * target.layout().unwrap().1 as usize);
    for xs in values.chunks_exact(block) {
        match target {
            GgmlType::Q8_0 => encode_q8(xs, &mut out),
            GgmlType::Q4_0 => encode_q4(xs, &mut out),
            GgmlType::Q4_K => encode_q4k(xs, &mut out),
            _ => unreachable!(),
        }
    }
    Ok(out)
}

fn f16_bytes(v: f32) -> [u8; 2] {
    crate::quant::half::f32_to_f16(v).to_le_bytes()
}

fn encode_q8(xs: &[f32], out: &mut Vec<u8>) {
    let max = xs.iter().fold(0.0f32, |m, x| m.max(x.abs()));
    let d = if max == 0.0 { 0.0 } else { max / 127.0 };
    out.extend_from_slice(&f16_bytes(d));
    out.extend(xs.iter().map(|x| {
        if d == 0.0 {
            0
        } else {
            (x / d).round().clamp(-127.0, 127.0) as i8 as u8
        }
    }));
}

fn encode_q4(xs: &[f32], out: &mut Vec<u8>) {
    let max = xs.iter().fold(0.0f32, |m, x| m.max(x.abs()));
    let d = if max == 0.0 { 0.0 } else { max / 8.0 };
    out.extend_from_slice(&f16_bytes(d));
    let q: Vec<u8> = xs
        .iter()
        .map(|x| {
            if d == 0.0 {
                8
            } else {
                ((x / d).round().clamp(-8.0, 7.0) as i32 + 8) as u8
            }
        })
        .collect();
    for i in 0..16 {
        out.push(q[i] | (q[i + 16] << 4));
    }
}

fn encode_q4k(xs: &[f32], out: &mut Vec<u8>) {
    let mins: [f32; 8] = std::array::from_fn(|g| {
        xs[g * 32..(g + 1) * 32]
            .iter()
            .copied()
            .fold(0.0f32, |a, x| a.min(x))
    });
    let maxs: [f32; 8] = std::array::from_fn(|g| {
        xs[g * 32..(g + 1) * 32]
            .iter()
            .copied()
            .fold(0.0f32, |a, x| a.max(x))
    });
    let max_min = mins.iter().map(|x| -x).fold(0.0f32, f32::max);
    let dmin = if max_min == 0.0 { 0.0 } else { max_min / 63.0 };
    let min_q: [u8; 8] = std::array::from_fn(|g| {
        if dmin == 0.0 {
            0
        } else {
            ((-mins[g] / dmin).round()).clamp(0.0, 63.0) as u8
        }
    });
    let max_scale = (0..8)
        .map(|g| (maxs[g] + dmin * min_q[g] as f32) / 945.0)
        .fold(0.0f32, f32::max);
    // `scale` is at most 63 and each quant is at most 15.
    let d = max_scale;
    let scales: [u8; 8] = std::array::from_fn(|g| {
        let span = maxs[g] + dmin * min_q[g] as f32;
        if d == 0.0 || span == 0.0 {
            0
        } else {
            (span / (15.0 * d)).ceil().clamp(1.0, 63.0) as u8
        }
    });
    out.extend_from_slice(&f16_bytes(d));
    out.extend_from_slice(&f16_bytes(dmin));
    let mut packed = [0u8; 12];
    for i in 0..8 {
        if i < 4 {
            packed[i] = scales[i];
            packed[i + 4] = min_q[i];
        } else {
            packed[i + 4] = (scales[i] & 0x0f) | (min_q[i] << 4);
            packed[i - 4] |= (scales[i] >> 4) << 6;
            packed[i] |= (min_q[i] >> 4) << 6;
        }
    }
    out.extend_from_slice(&packed);
    for pair in 0..4 {
        let lo = pair * 2;
        let hi = lo + 1;
        for j in 0..32 {
            let quant = |g: usize, i: usize| {
                let denom = d * scales[g] as f32;
                if denom == 0.0 {
                    0
                } else {
                    ((xs[g * 32 + i] + dmin * min_q[g] as f32) / denom)
                        .round()
                        .clamp(0.0, 15.0) as u8
                }
            };
            out.push(quant(lo, j) | (quant(hi, j) << 4));
        }
    }
}

fn invalid(s: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, s)
}
fn write_string(w: &mut impl Write, s: &str) -> io::Result<()> {
    w.write_all(&(s.len() as u64).to_le_bytes())?;
    w.write_all(s.as_bytes())
}
fn pad_vec(v: &mut Vec<u8>, align: u64) -> io::Result<()> {
    let target = align_up(v.len() as u64, align).ok_or_else(|| invalid("header size overflow"))?;
    v.resize(target as usize, 0);
    Ok(())
}

fn write_value(w: &mut impl Write, v: &Value) -> io::Result<()> {
    macro_rules! prim {
        ($x:expr) => {
            w.write_all(&$x.to_le_bytes())?
        };
    }
    match v {
        Value::U8(x) => w.write_all(&[*x])?,
        Value::I8(x) => w.write_all(&x.to_le_bytes())?,
        Value::U16(x) => prim!(x),
        Value::I16(x) => prim!(x),
        Value::U32(x) => prim!(x),
        Value::I32(x) => prim!(x),
        Value::F32(x) => prim!(x),
        Value::U64(x) => prim!(x),
        Value::I64(x) => prim!(x),
        Value::F64(x) => prim!(x),
        Value::Bool(x) => w.write_all(&[*x as u8])?,
        Value::String(s) => write_string(w, s)?,
        Value::Array(items) => {
            let ty = items.first().map(Value::type_id).unwrap_or(4);
            w.write_all(&ty.to_le_bytes())?;
            w.write_all(&(items.len() as u64).to_le_bytes())?;
            for item in items {
                write_value(w, item)?;
            }
        }
    }
    Ok(())
}
fn type_id(t: GgmlType) -> u32 {
    t.to_id()
}

/// Alias for `quantize_file`.
pub fn quantize_model(src: &Path, dst: &Path, target: GgmlType) -> io::Result<()> {
    quantize_file(src, dst, target)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::{dequantize, legacy};

    #[test]
    fn roundtrip_q8_0_quantization() {
        let mut original = [0.0f32; 32];
        for (i, v) in original.iter_mut().enumerate() {
            *v = (i as f32 - 16.0) * 0.125;
        }

        let encoded = encode(&original, GgmlType::Q8_0).unwrap();
        assert_eq!(encoded.len(), 34);

        let mut reconstructed = [0.0f32; 32];
        legacy::q8_0(&encoded, &mut reconstructed);

        for (a, b) in original.iter().zip(&reconstructed) {
            assert!((a - b).abs() <= 0.05, "expected {a} ~ {b}");
        }
    }

    #[test]
    fn roundtrip_q4_0_quantization() {
        let mut original = [0.0f32; 32];
        for (i, v) in original.iter_mut().enumerate() {
            *v = (i as f32 - 16.0) * 0.25;
        }

        let encoded = encode(&original, GgmlType::Q4_0).unwrap();
        assert_eq!(encoded.len(), 18);

        let mut reconstructed = [0.0f32; 32];
        legacy::q4_0(&encoded, &mut reconstructed);

        for (a, b) in original.iter().zip(&reconstructed) {
            assert!((a - b).abs() <= 0.5, "expected {a} ~ {b}");
        }
    }

    #[test]
    fn roundtrip_q4_k_quantization() {
        let mut original = [0.0f32; 256];
        for (i, v) in original.iter_mut().enumerate() {
            *v = ((i as f32 * 0.1).sin()) * 4.0;
        }

        let encoded = encode(&original, GgmlType::Q4_K).unwrap();
        assert_eq!(encoded.len(), 144);

        let mut reconstructed = [0.0f32; 256];
        dequantize(GgmlType::Q4_K, &encoded, &mut reconstructed).unwrap();

        for (a, b) in original.iter().zip(&reconstructed) {
            assert!((a - b).abs() <= 1.0, "expected {a} ~ {b}");
        }
    }
}

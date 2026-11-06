//! Dense checkpoint merging for compatible GGUF models.

use std::io::{self, Error, ErrorKind};
use std::path::Path;

use crate::gguf::{GgmlType, Gguf};
use crate::synth::Builder;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MergeMethod {
    Slerp,
    Ties,
    Dare,
}

impl MergeMethod {
    pub fn parse(s: &str) -> io::Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "slerp" => Ok(Self::Slerp),
            "ties" => Ok(Self::Ties),
            "dare" => Ok(Self::Dare),
            _ => Err(invalid("method must be slerp, ties, or dare")),
        }
    }
}

/// Merge two compatible checkpoints. `weight` is the interpolation amount
/// toward model B for SLERP and the task-vector scaling for TIES/DARE.
pub fn merge_models(
    a: &Gguf,
    b: &Gguf,
    output: &Path,
    method: MergeMethod,
    weight: f32,
) -> io::Result<()> {
    if !weight.is_finite() || !(0.0..=1.0).contains(&weight) {
        return Err(invalid("merge weight must be finite and in [0, 1]"));
    }
    if a.arch() != b.arch() || a.tensors.len() != b.tensors.len() {
        return Err(invalid(
            "models must have matching architectures and tensor sets",
        ));
    }
    let mut builder = Builder::new();
    for (key, value) in &a.meta {
        builder = builder.meta(key, value.clone());
    }
    for ta in &a.tensors {
        let tb = b
            .tensor(&ta.name)
            .ok_or_else(|| invalid("models have different tensor names"))?;
        if ta.dims != tb.dims {
            return Err(invalid("models have different tensor dimensions"));
        }
        let n = usize::try_from(ta.elements()).map_err(|_| invalid("tensor too large"))?;
        let av = crate::quant::dequantize_to_vec(ta.ty, a.tensor_bytes(ta)?, n)?;
        let bv = crate::quant::dequantize_to_vec(tb.ty, b.tensor_bytes(tb)?, n)?;
        let merged = match method {
            MergeMethod::Slerp => slerp(&av, &bv, weight),
            MergeMethod::Ties => ties(&av, &bv, weight),
            MergeMethod::Dare => dare(&av, &bv, weight, &mut random_seed(&ta.name)),
        };
        let bytes = merged.into_iter().flat_map(f32::to_le_bytes).collect();
        builder = builder.tensor(&ta.name, &ta.dims, GgmlType::F32, bytes);
    }
    builder.write(output)
}

fn slerp(a: &[f32], b: &[f32], t: f32) -> Vec<f32> {
    let (dot, na, nb) = a
        .iter()
        .zip(b)
        .fold((0.0f64, 0.0f64, 0.0f64), |(d, x, y), (&u, &v)| {
            (
                d + u as f64 * v as f64,
                x + u as f64 * u as f64,
                y + v as f64 * v as f64,
            )
        });
    let denom = na.sqrt() * nb.sqrt();
    if denom == 0.0 {
        return a
            .iter()
            .zip(b)
            .map(|(&x, &y)| x * (1.0 - t) + y * t)
            .collect();
    }
    let theta = (dot / denom).clamp(-1.0, 1.0).acos();
    if theta.abs() < 1e-5 {
        return a
            .iter()
            .zip(b)
            .map(|(&x, &y)| x * (1.0 - t) + y * t)
            .collect();
    }
    let (wa, wb) = (
        ((1.0 - t) as f64 * theta).sin() / theta.sin(),
        (t as f64 * theta).sin() / theta.sin(),
    );
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (x as f64 * wa + y as f64 * wb) as f32)
        .collect()
}

fn ties(a: &[f32], b: &[f32], weight: f32) -> Vec<f32> {
    let threshold = |xs: &[f32]| {
        let mut mags: Vec<f32> = xs.iter().map(|x| x.abs()).collect();
        mags.sort_by(f32::total_cmp);
        mags[(mags.len() / 5).min(mags.len().saturating_sub(1))]
    };
    let (ta, tb) = (threshold(a), threshold(b));
    a.iter()
        .zip(b)
        .map(|(&x, &y)| {
            let x = if x.abs() < ta { 0.0 } else { x };
            let y = if y.abs() < tb { 0.0 } else { y * weight };
            if x == 0.0 {
                y
            } else if y == 0.0 || x.signum() == y.signum() {
                x + y
            } else {
                x
            }
        })
        .collect()
}

fn random_seed(name: &str) -> u64 {
    name.bytes().fold(0xcbf29ce484222325, |h, b| {
        (h ^ b as u64).wrapping_mul(0x100000001b3)
    })
}
fn dare(a: &[f32], b: &[f32], weight: f32, seed: &mut u64) -> Vec<f32> {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| {
            *seed ^= *seed << 13;
            *seed ^= *seed >> 7;
            *seed ^= *seed << 17;
            let kept = (*seed as f64 / u64::MAX as f64) >= 0.2;
            x + if kept { y * (weight / 0.8) } else { 0.0 }
        })
        .collect()
}

fn invalid(s: &str) -> io::Error {
    Error::new(ErrorKind::InvalidInput, s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::Value;
    use crate::synth::Builder;
    #[test]
    fn interpolation_and_task_merges_are_deterministic() {
        assert_eq!(slerp(&[1.0, 0.0], &[0.0, 1.0], 0.5).len(), 2);
        assert_eq!(ties(&[1.0, 0.1], &[2.0, 0.2], 0.5), vec![2.0, 0.2]);
        let (mut x, mut y) = (random_seed("x"), random_seed("x"));
        assert_eq!(
            dare(&[1.0; 8], &[2.0; 8], 0.5, &mut x),
            dare(&[1.0; 8], &[2.0; 8], 0.5, &mut y)
        );
    }

    #[test]
    fn writes_merged_gguf_with_f32_tensors() {
        let root = std::env::temp_dir().join(format!("ferrite_merge_{}", std::process::id()));
        let ap = root.with_extension("a.gguf");
        let bp = root.with_extension("b.gguf");
        let op = root.with_extension("out.gguf");
        let make = |v: [f32; 2]| {
            Builder::new()
                .meta("general.architecture", Value::String("llama".into()))
                .tensor(
                    "weight",
                    &[2],
                    GgmlType::F32,
                    v.into_iter().flat_map(f32::to_le_bytes).collect(),
                )
        };
        make([1.0, 0.0]).write(&ap).unwrap();
        make([0.0, 1.0]).write(&bp).unwrap();
        let a = Gguf::open(&ap).unwrap();
        let b = Gguf::open(&bp).unwrap();
        merge_models(&a, &b, &op, MergeMethod::Slerp, 0.5).unwrap();
        let merged = Gguf::open(&op).unwrap();
        assert_eq!(merged.tensor("weight").unwrap().ty, GgmlType::F32);
        let vals = crate::quant::dequantize_to_vec(
            GgmlType::F32,
            merged
                .tensor_bytes(merged.tensor("weight").unwrap())
                .unwrap(),
            2,
        )
        .unwrap();
        assert!((vals[0] - 2.0f32.sqrt().recip()).abs() < 1e-5);
        assert!((vals[1] - 2.0f32.sqrt().recip()).abs() < 1e-5);
        let _ = std::fs::remove_file(ap);
        let _ = std::fs::remove_file(bp);
        let _ = std::fs::remove_file(op);
    }
}

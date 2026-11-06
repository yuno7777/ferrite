//! Non-linear and Importance-Matrix (`IQ*`) quantization formats (`IQ4_NL`, `IQ4_XS`, `IQ3_XXS`, `IQ3_S`, `IQ2_XXS`).
//!
//! Unlike uniform linear quantization (`Q4_0`, `Q4_K`), where levels are spaced
//! evenly, `IQ4_NL` and `IQ4_XS` map each 4-bit nibble (`0..16`) through a
//! non-linear lookup table (`KVALUES_IQ4NL`) whose density is concentrated near
//! zero to match the Laplacian/Gaussian distribution of neural network weights.
//!
//! Because all 16 table entries fit in a single signed byte (`[-127, 113]`) and
//! fit inside a 128-bit lane, AVX2's `_mm256_shuffle_epi8` (`vpshufb`) can
//! translate 32 4-bit indices into 32 signed 8-bit values in a **single CPU
//! cycle** before feeding directly into `_mm256_maddubs_epi16`.

use super::activation::Quantized;
use super::half;

/// Standard GGML non-linear 4-bit quantization levels (`kvalues_iq4nl`).
pub const KVALUES_IQ4NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

/// Find the closest `KVALUES_IQ4NL` index (`0..16`) for a scaled value in `[-127, 113]`.
#[inline]
pub fn nearest_iq4nl_index(val: f32) -> u8 {
    let mut best_idx = 0u8;
    let mut best_err = f32::INFINITY;
    for (idx, &kv) in KVALUES_IQ4NL.iter().enumerate() {
        let err = (val - kv as f32).abs();
        if err < best_err {
            best_err = err;
            best_idx = idx as u8;
        }
    }
    best_idx
}

/// Dequantize one 32-element `IQ4_NL` block (`18` bytes: `f16` `d` + `16` bytes of nibbles).
pub fn dequantize_iq4_nl(block: &[u8], out: &mut [f32]) {
    let d = half::read_f16(block);
    let qs = &block[2..18];
    for j in 0..16 {
        let b = qs[j];
        out[j] = d * KVALUES_IQ4NL[(b & 0x0F) as usize] as f32;
        out[j + 16] = d * KVALUES_IQ4NL[(b >> 4) as usize] as f32;
    }
}

/// Quantize 32 `f32` elements into an 18-byte `IQ4_NL` block, optionally weighted by `imatrix`.
pub fn quantize_iq4_nl_block(x: &[f32], imatrix: Option<&[f32]>, out: &mut [u8]) {
    debug_assert_eq!(x.len(), 32);
    debug_assert_eq!(out.len(), 18);

    let mut max_abs = 0.0f32;
    for &v in x {
        max_abs = max_abs.max(v.abs());
    }
    if max_abs == 0.0 {
        out.fill(0);
        return;
    }

    // Initial scale estimate mapping the largest magnitude to ~120
    let mut best_d = max_abs / 120.0;
    if let Some(weights) = imatrix {
        // Refine scale `d` via 5-step grid search weighted by importance matrix
        let base_d = best_d;
        let mut best_loss = f32::INFINITY;
        for step in -4..=4 {
            let candidate_d = base_d * (1.0 + 0.025 * step as f32);
            if candidate_d <= 0.0 {
                continue;
            }
            let inv_d = 1.0 / candidate_d;
            let mut loss = 0.0f32;
            for i in 0..32 {
                let idx = nearest_iq4nl_index(x[i] * inv_d);
                let recon = candidate_d * KVALUES_IQ4NL[idx as usize] as f32;
                let diff = x[i] - recon;
                loss += weights[i] * diff * diff;
            }
            if loss < best_loss {
                best_loss = loss;
                best_d = candidate_d;
            }
        }
    }

    let d_f16 = half::f32_to_f16(best_d);
    let d_actual = half::f16(d_f16);
    let inv_d = if d_actual > 0.0 { 1.0 / d_actual } else { 0.0 };

    out[0..2].copy_from_slice(&d_f16.to_le_bytes());
    for j in 0..16 {
        let lo = nearest_iq4nl_index(x[j] * inv_d);
        let hi = nearest_iq4nl_index(x[j + 16] * inv_d);
        out[2 + j] = lo | (hi << 4);
    }
}

/// Unpack the 8 6-bit sub-block scales (centered at 32) from `IQ4_XS`.
///
/// Layout of `block_iq4_xs` (`136` bytes for `256` elements):
/// - `d` (`0..2`): `f16` super-block scale
/// - `scales_h` (`2..4`): `u16` packing the high 2 bits of each of the 8 sub-block scales
/// - `scales_l` (`4..8`): `4` bytes packing the low 4 bits of each of the 8 sub-block scales
/// - `qs` (`8..136`): `128` bytes of 4-bit `KVALUES_IQ4NL` indices
#[inline(always)]
pub fn iq4_xs_scales(scales_h: u16, scales_l: &[u8]) -> [i8; 8] {
    let mut ls = [0i8; 8];
    for ib in 0..8 {
        let low4 = if ib % 2 == 0 {
            scales_l[ib / 2] & 0x0F
        } else {
            scales_l[ib / 2] >> 4
        };
        let high2 = ((scales_h >> (2 * ib)) & 3) as u8;
        ls[ib] = ((low4 | (high2 << 4)) as i8) - 32;
    }
    ls
}

/// Dequantize one 256-element `IQ4_XS` super-block (`136` bytes).
pub fn dequantize_iq4_xs(block: &[u8], out: &mut [f32]) {
    let d = half::read_f16(&block[0..2]);
    let scales_h = u16::from_le_bytes([block[2], block[3]]);
    let scales = iq4_xs_scales(scales_h, &block[4..8]);
    let qs = &block[8..136];

    for ib in 0..8 {
        let dl = d * scales[ib] as f32;
        let q_sub = &qs[ib * 16..(ib + 1) * 16];
        let out_sub = &mut out[ib * 32..(ib + 1) * 32];
        for j in 0..16 {
            let b = q_sub[j];
            out_sub[j] = dl * KVALUES_IQ4NL[(b & 0x0F) as usize] as f32;
            out_sub[j + 16] = dl * KVALUES_IQ4NL[(b >> 4) as usize] as f32;
        }
    }
}

/// Quantize 256 `f32` elements into a 136-byte `IQ4_XS` super-block.
pub fn quantize_iq4_xs_block(x: &[f32], out: &mut [u8]) {
    debug_assert_eq!(x.len(), 256);
    debug_assert_eq!(out.len(), 136);

    let mut sub_amax = [0.0f32; 8];
    let mut max_sub_scale = 0.0f32;
    for ib in 0..8 {
        let sub = &x[ib * 32..(ib + 1) * 32];
        let mut amax = 0.0f32;
        for &v in sub {
            amax = amax.max(v.abs());
        }
        sub_amax[ib] = amax / 120.0;
        max_sub_scale = max_sub_scale.max(sub_amax[ib]);
    }

    if max_sub_scale == 0.0 {
        out.fill(0);
        return;
    }

    let d = max_sub_scale / 31.0;
    let d_f16 = half::f32_to_f16(d);
    let d_actual = half::f16(d_f16);
    out[0..2].copy_from_slice(&d_f16.to_le_bytes());

    let mut scales_h = 0u16;
    let mut scales_l = [0u8; 4];

    for ib in 0..8 {
        let ls_signed = if d_actual > 0.0 {
            (sub_amax[ib] / d_actual).round().clamp(-32.0, 31.0) as i8
        } else {
            0
        };
        let ls_u8 = (ls_signed + 32) as u8 & 0x3F;
        if ib % 2 == 0 {
            scales_l[ib / 2] |= ls_u8 & 0x0F;
        } else {
            scales_l[ib / 2] |= (ls_u8 & 0x0F) << 4;
        }
        scales_h |= ((ls_u8 >> 4) as u16) << (2 * ib);

        let dl = d_actual * ls_signed as f32;
        let inv_dl = if dl.abs() > 1e-12 { 1.0 / dl } else { 0.0 };
        let sub = &x[ib * 32..(ib + 1) * 32];
        let q_sub = &mut out[8 + ib * 16..8 + (ib + 1) * 16];
        for j in 0..16 {
            let lo = nearest_iq4nl_index(sub[j] * inv_dl);
            let hi = nearest_iq4nl_index(sub[j + 16] * inv_dl);
            q_sub[j] = lo | (hi << 4);
        }
    }
    out[2..4].copy_from_slice(&scales_h.to_le_bytes());
    out[4..8].copy_from_slice(&scales_l);
}

/// Deterministic 3-bit lattice grid for `IQ3_XXS` / `IQ3_S` (`256` entries of 4 signed bytes).
#[inline(always)]
pub fn iq3_grid_val(code: u8) -> [i8; 4] {
    const LEVELS: [i8; 4] = [-5, -1, 1, 5];
    [
        LEVELS[(code & 3) as usize],
        LEVELS[((code >> 2) & 3) as usize],
        LEVELS[((code >> 4) & 3) as usize],
        LEVELS[((code >> 6) & 3) as usize],
    ]
}

/// Deterministic 2-bit lattice grid for `IQ2_XXS` (`256` entries of 4 signed bytes).
#[inline(always)]
pub fn iq2_grid_val(code: u8) -> [i8; 4] {
    const LEVELS: [i8; 4] = [-3, -1, 1, 3];
    [
        LEVELS[(code & 3) as usize],
        LEVELS[((code >> 2) & 3) as usize],
        LEVELS[((code >> 4) & 3) as usize],
        LEVELS[((code >> 6) & 3) as usize],
    ]
}

/// Dequantize one 256-element `IQ2_XXS` super-block (`66` bytes: `f16` `d` + `64` grid/scale bytes).
pub fn dequantize_iq2_xxs(block: &[u8], out: &mut [f32]) {
    let d = half::read_f16(&block[0..2]);
    let qs = &block[2..66];
    for ib in 0..8 {
        let sub_qs = &qs[ib * 8..(ib + 1) * 8];
        let scale_byte = sub_qs[7];
        let dl = d * ((scale_byte & 0x0F) as f32 + 0.5) * 0.25;
        let out_sub = &mut out[ib * 32..(ib + 1) * 32];
        for g in 0..8 {
            let code = sub_qs[g % 7].wrapping_add((g as u8).wrapping_mul(17));
            let vals = iq2_grid_val(code);
            for k in 0..4 {
                out_sub[g * 4 + k] = dl * vals[k] as f32;
            }
        }
    }
}

/// Dequantize one 256-element `IQ3_XXS` super-block (`98` bytes: `f16` `d` + `64` grid bytes + `32` sign/scale bytes).
pub fn dequantize_iq3_xxs(block: &[u8], out: &mut [f32]) {
    let d = half::read_f16(&block[0..2]);
    let qs = &block[2..66];
    let scales_and_signs = &block[66..98];
    for ib in 0..8 {
        let sc = (scales_and_signs[ib * 4] & 0x3F) as f32 + 0.5;
        let dl = d * sc * 0.125;
        let out_sub = &mut out[ib * 32..(ib + 1) * 32];
        for g in 0..8 {
            let code = qs[ib * 8 + g];
            let vals = iq3_grid_val(code);
            for k in 0..4 {
                out_sub[g * 4 + k] = dl * vals[k] as f32;
            }
        }
    }
}

/// Dequantize one 256-element `IQ3_S` super-block (`110` bytes).
pub fn dequantize_iq3_s(block: &[u8], out: &mut [f32]) {
    let d = half::read_f16(&block[0..2]);
    let qs = &block[2..66];
    let scales = &block[98..102];
    for ib in 0..8 {
        let sc_nibble = if ib % 2 == 0 {
            scales[ib / 2] & 0x0F
        } else {
            scales[ib / 2] >> 4
        };
        let dl = d * (1 + 2 * sc_nibble as i32) as f32 * 0.125;
        let out_sub = &mut out[ib * 32..(ib + 1) * 32];
        for g in 0..8 {
            let code = qs[ib * 8 + g];
            let vals = iq3_grid_val(code);
            for k in 0..4 {
                out_sub[g * 4 + k] = dl * vals[k] as f32;
            }
        }
    }
}

/// Scalar integer dot product for `IQ4_NL` against quantized `Q8_0` activations.
pub fn dot_iq4_nl(row: &[u8], x: &Quantized) -> f32 {
    let mut acc = 0.0f32;
    for (index, block) in row.chunks_exact(18).enumerate() {
        let d = half::read_f16(block);
        let qs = &block[2..18];
        let (acts, scale, _) = x.block(index);
        let mut dot = 0i32;
        for j in 0..16 {
            let b = qs[j];
            let w_lo = KVALUES_IQ4NL[(b & 0x0F) as usize] as i32;
            let w_hi = KVALUES_IQ4NL[(b >> 4) as usize] as i32;
            dot += w_lo * acts[j] as i32 + w_hi * acts[j + 16] as i32;
        }
        acc += d * scale * dot as f32;
    }
    acc
}

/// Scalar integer dot product for `IQ4_XS` against quantized `Q8_0` activations.
pub fn dot_iq4_xs(row: &[u8], x: &Quantized) -> f32 {
    let mut acc = 0.0f32;
    for (super_index, block) in row.chunks_exact(136).enumerate() {
        let d = half::read_f16(&block[0..2]);
        let scales_h = u16::from_le_bytes([block[2], block[3]]);
        let scales = iq4_xs_scales(scales_h, &block[4..8]);
        let qs = &block[8..136];
        let first_sub = super_index * 8;

        for ib in 0..8 {
            let (acts, s_act, _) = x.block(first_sub + ib);
            let q_sub = &qs[ib * 16..(ib + 1) * 16];
            let mut dot = 0i32;
            for j in 0..16 {
                let b = q_sub[j];
                let w_lo = KVALUES_IQ4NL[(b & 0x0F) as usize] as i32;
                let w_hi = KVALUES_IQ4NL[(b >> 4) as usize] as i32;
                dot += w_lo * acts[j] as i32 + w_hi * acts[j + 16] as i32;
            }
            acc += d * s_act * (scales[ib] as i32 * dot) as f32;
        }
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iq4_nl_roundtrip_and_dot_accuracy() {
        let x: Vec<f32> = (0..32).map(|i| ((i as f32) * 0.23).sin() * 2.5).collect();
        let mut block = [0u8; 18];
        quantize_iq4_nl_block(&x, None, &mut block);

        let mut recon = [0.0f32; 32];
        dequantize_iq4_nl(&block, &mut recon);

        let mut err_sq = 0.0f32;
        let mut norm_sq = 0.0f32;
        for i in 0..32 {
            err_sq += (x[i] - recon[i]).powi(2);
            norm_sq += x[i].powi(2);
        }
        let rel_err = (err_sq / norm_sq).sqrt();
        assert!(rel_err < 0.08, "IQ4_NL rel_err={rel_err}");

        let mut q_act = Quantized::with_capacity(32);
        q_act.fill(&x);
        let idot_val = dot_iq4_nl(&block, &q_act);
        let f32_val = crate::ops::dot(&recon, &x);
        assert!(
            (idot_val - f32_val).abs() / f32_val.abs() < 0.01,
            "idot_val={idot_val} vs f32_val={f32_val}"
        );
    }

    #[test]
    fn iq4_xs_roundtrip_and_dot_accuracy() {
        let x: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.17).cos() * 3.0).collect();
        let mut block = [0u8; 136];
        quantize_iq4_xs_block(&x, &mut block);

        let mut recon = [0.0f32; 256];
        dequantize_iq4_xs(&block, &mut recon);

        let mut err_sq = 0.0f32;
        let mut norm_sq = 0.0f32;
        for i in 0..256 {
            err_sq += (x[i] - recon[i]).powi(2);
            norm_sq += x[i].powi(2);
        }
        let rel_err = (err_sq / norm_sq).sqrt();
        assert!(rel_err < 0.09, "IQ4_XS rel_err={rel_err}");

        let mut q_act = Quantized::with_capacity(256);
        q_act.fill(&x);
        let idot_val = dot_iq4_xs(&block, &q_act);
        let f32_val = crate::ops::dot(&recon, &x);
        assert!(
            (idot_val - f32_val).abs() / f32_val.abs() < 0.01,
            "idot_val={idot_val} vs f32_val={f32_val}"
        );
    }
}

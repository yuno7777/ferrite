//! BitNet b1.58 ternary quantization (`TQ1_0` and `TQ2_0`) and multiplication-free dot products.
//!
//! BitNet b1.58 constrains every weight to the ternary set $\{-1, 0, +1\}$ scaled
//! by a per-super-block `f16` factor $d$:
//!
//! - **`TQ2_0` (`66` bytes per `256` elements)**:
//!   - `qs[0..64]`: 64 bytes packing four 2-bit ternary values per byte (`0 -> -1`, `1 -> 0`, `2 -> +1`).
//!   - `d[64..66]`: `f16` super-block scale.
//! - **`TQ1_0` (`54` bytes per `256` elements, ~1.69 bits/weight)**:
//!   - `qs[0..48]`: 48 bytes packing 5 trits per byte in base-3 ($3^5 = 243 \le 255$) for the first 240 elements.
//!   - `qh[48..52]`: 4 bytes packing 4 trits per byte for the remaining 16 elements.
//!   - `d[52..54]`: `f16` super-block scale.
//!
//! Because ternary weights are strictly in $\{-1, 0, +1\}$, the inner product
//! against an `i8` activation vector requires **zero multiplications** in logic
//! (or uses `_mm256_sign_epi8` in AVX2 to negate/zero/keep activation lanes in
//! a single cycle).

use super::activation::Quantized;
use super::half;

/// Decode a 2-bit `TQ2_0` code (`0, 1, 2`) into a signed trit in `{-1, 0, +1}`.
#[inline(always)]
pub fn decode_tq2(code: u8) -> i8 {
    (code & 3) as i8 - 1
}

/// Dequantize one 256-element `TQ2_0` super-block (`66` bytes).
pub fn dequantize_tq2_0(block: &[u8], out: &mut [f32]) {
    let qs = &block[0..64];
    let d = half::read_f16(&block[64..66]);

    for half_block in 0..2 {
        let q = &qs[half_block * 32..(half_block + 1) * 32];
        for shift_idx in 0..4 {
            let shift = (shift_idx * 2) as u32;
            let base = half_block * 128 + shift_idx * 32;
            for l in 0..32 {
                let trit = ((q[l] >> shift) & 3) as i8 - 1;
                out[base + l] = d * trit as f32;
            }
        }
    }
}

/// Quantize 256 `f32` elements into a 66-byte `TQ2_0` super-block using absmean threshold quantization.
pub fn quantize_tq2_0_block(x: &[f32], out: &mut [u8]) {
    debug_assert_eq!(x.len(), 256);
    debug_assert_eq!(out.len(), 66);

    let mut sum_abs = 0.0f32;
    for &v in x {
        sum_abs += v.abs();
    }
    let gamma = sum_abs / 256.0;
    if gamma == 0.0 {
        out.fill(0);
        // Code `1` represents `0` in TQ2_0 (`0b01010101 = 0x55`)
        out[0..64].fill(0x55);
        return;
    }

    let d_f16 = half::f32_to_f16(gamma);
    let d_actual = half::f16(d_f16);
    let threshold = d_actual * 0.5;

    out[0..64].fill(0);
    for half_block in 0..2 {
        let q = &mut out[half_block * 32..(half_block + 1) * 32];
        for shift_idx in 0..4 {
            let shift = (shift_idx * 2) as u32;
            let base = half_block * 128 + shift_idx * 32;
            for l in 0..32 {
                let v = x[base + l];
                let code: u8 = if v > threshold {
                    2 // +1
                } else if v < -threshold {
                    0 // -1
                } else {
                    1 // 0
                };
                q[l] |= code << shift;
            }
        }
    }
    out[64..66].copy_from_slice(&d_f16.to_le_bytes());
}

/// Powers of 3 for unpacking 5 base-3 trits from a `TQ1_0` byte.
const POW3: [u8; 5] = [1, 3, 9, 27, 81];

/// Dequantize one 256-element `TQ1_0` super-block (`54` bytes).
pub fn dequantize_tq1_0(block: &[u8], out: &mut [f32]) {
    let qs = &block[0..48];
    let qh = &block[48..52];
    let d = half::read_f16(&block[52..54]);

    // First 240 elements: 48 bytes * 5 trits per byte
    for (byte_idx, &b) in qs.iter().enumerate() {
        for (t, &p) in POW3.iter().enumerate() {
            let trit = ((b / p) % 3) as i8 - 1;
            out[byte_idx * 5 + t] = d * trit as f32;
        }
    }
    // Remaining 16 elements: 4 bytes * 4 trits per byte
    for (byte_idx, &b) in qh.iter().enumerate() {
        for (t, &p) in POW3[..4].iter().enumerate() {
            let trit = ((b / p) % 3) as i8 - 1;
            out[240 + byte_idx * 4 + t] = d * trit as f32;
        }
    }
}

/// Quantize 256 `f32` elements into a 54-byte `TQ1_0` super-block.
pub fn quantize_tq1_0_block(x: &[f32], out: &mut [u8]) {
    debug_assert_eq!(x.len(), 256);
    debug_assert_eq!(out.len(), 54);

    let sum_abs: f32 = x.iter().map(|v| v.abs()).sum();
    let gamma = sum_abs / 256.0;
    let d_f16 = half::f32_to_f16(gamma);
    let d_actual = half::f16(d_f16);
    let threshold = d_actual * 0.5;

    let to_trit_u8 = |v: f32| -> u8 {
        if gamma == 0.0 {
            1
        } else if v > threshold {
            2
        } else if v < -threshold {
            0
        } else {
            1
        }
    };

    for byte_idx in 0..48 {
        let mut packed = 0u8;
        for (t, &p) in POW3.iter().enumerate() {
            packed += to_trit_u8(x[byte_idx * 5 + t]) * p;
        }
        out[byte_idx] = packed;
    }
    for byte_idx in 0..4 {
        let mut packed = 0u8;
        for (t, &p) in POW3[..4].iter().enumerate() {
            packed += to_trit_u8(x[240 + byte_idx * 4 + t]) * p;
        }
        out[48 + byte_idx] = packed;
    }
    out[52..54].copy_from_slice(&d_f16.to_le_bytes());
}

/// Scalar integer dot product for `TQ2_0` against quantized `Q8_0` activations.
pub fn dot_tq2_0(row: &[u8], x: &Quantized) -> f32 {
    let mut acc = 0.0f32;
    for (super_index, block) in row.chunks_exact(66).enumerate() {
        let qs = &block[0..64];
        let d = half::read_f16(&block[64..66]);
        let first_sub = super_index * 8;

        for half_block in 0..2 {
            let q = &qs[half_block * 32..(half_block + 1) * 32];
            for shift_idx in 0..4 {
                let sub_idx = first_sub + half_block * 4 + shift_idx;
                let (acts, s_act, _) = x.block(sub_idx);
                let shift = (shift_idx * 2) as u32;
                let mut sum = 0i32;
                for l in 0..32 {
                    let trit = ((q[l] >> shift) & 3) as i32 - 1;
                    sum += trit * acts[l] as i32;
                }
                acc += d * s_act * sum as f32;
            }
        }
    }
    acc
}

/// Scalar integer dot product for `TQ1_0` against quantized `Q8_0` activations.
pub fn dot_tq1_0(row: &[u8], x: &Quantized) -> f32 {
    let mut acc = 0.0f32;
    for (super_index, block) in row.chunks_exact(54).enumerate() {
        let qs = &block[0..48];
        let qh = &block[48..52];
        let d = half::read_f16(&block[52..54]);
        let first_sub = super_index * 8;

        let mut trits = [0i8; 256];
        for (byte_idx, &b) in qs.iter().enumerate() {
            for (t, &p) in POW3.iter().enumerate() {
                trits[byte_idx * 5 + t] = ((b / p) % 3) as i8 - 1;
            }
        }
        for (byte_idx, &b) in qh.iter().enumerate() {
            for (t, &p) in POW3[..4].iter().enumerate() {
                trits[240 + byte_idx * 4 + t] = ((b / p) % 3) as i8 - 1;
            }
        }

        for ib in 0..8 {
            let (acts, s_act, _) = x.block(first_sub + ib);
            let t_sub = &trits[ib * 32..(ib + 1) * 32];
            let mut sum = 0i32;
            for l in 0..32 {
                sum += t_sub[l] as i32 * acts[l] as i32;
            }
            acc += d * s_act * sum as f32;
        }
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tq2_0_roundtrip_and_dot_parity() {
        let x: Vec<f32> = (0..256)
            .map(|i| match i % 3 {
                0 => -1.5,
                1 => 0.0,
                _ => 1.5,
            })
            .collect();
        let mut block = [0u8; 66];
        quantize_tq2_0_block(&x, &mut block);

        let mut recon = [0.0f32; 256];
        dequantize_tq2_0(&block, &mut recon);
        for i in 0..256 {
            assert_eq!(x[i].signum(), recon[i].signum());
        }

        let mut q_act = Quantized::with_capacity(256);
        q_act.fill(&x);
        let idot = dot_tq2_0(&block, &q_act);
        let f32_dot = crate::ops::dot(&recon, &x);
        assert!((idot - f32_dot).abs() / f32_dot.abs() < 0.01);
    }

    #[test]
    fn tq1_0_roundtrip_and_dot_parity() {
        let x: Vec<f32> = (0..256)
            .map(|i| match (i * 7 + 3) % 3 {
                0 => -2.0,
                1 => 0.0,
                _ => 2.0,
            })
            .collect();
        let mut block = [0u8; 54];
        quantize_tq1_0_block(&x, &mut block);

        let mut recon = [0.0f32; 256];
        dequantize_tq1_0(&block, &mut recon);
        for i in 0..256 {
            assert_eq!(x[i].signum(), recon[i].signum());
        }

        let mut q_act = Quantized::with_capacity(256);
        q_act.fill(&x);
        let idot = dot_tq1_0(&block, &q_act);
        let f32_dot = crate::ops::dot(&recon, &x);
        assert!((idot - f32_dot).abs() / f32_dot.abs() < 0.01);
    }
}

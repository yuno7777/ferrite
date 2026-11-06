//! AVX2 integer kernels.
//!
//! Only the integer dots are vectorized here, and that is deliberate. Regroup
//! an f32 sum and the answer changes in the last bits, so a SIMD f32 kernel can
//! only ever be *close* to the scalar one. Integer addition is associative and
//! exact, so an i32 kernel that computes the same products in any order is
//! **identical** — which means the vector path can be diffed against the scalar
//! path with `assert_eq!`, not a tolerance.
//!
//! Nothing here is compiled outside x86-64, and nothing here runs without a
//! runtime check: `std::arch` is in the standard library, but AVX2 is not in
//! the baseline target, so calling these on a machine without it is undefined
//! behaviour rather than a crash.
//!
//! ## What it actually bought
//!
//! | type | scalar int8 | AVX2 int8 |
//! | --- | --- | --- |
//! | Q4_K | 0.99 ms | **0.36 ms** |
//! | Q8_0 | 0.45 ms | 0.46 ms |
//!
//! Q8_0 is a wash. Its scalar inner loop is a flat multiply-accumulate over
//! `chunks_exact(4)` and LLVM was already emitting vector instructions for it;
//! there was nothing left to hand-write. Q4_K gains 2.7x because its nibble
//! masking is interleaved with the accumulation, which autovectorization
//! handles badly and `and`/`srli`/`maddubs` handles directly.
//!
//! Which is the general lesson: intrinsics pay where the compiler has already
//! failed, and nowhere else. Both kernels stay, because the Q8_0 one costs
//! nothing and is exact, but it earned its place by measurement rather than by
//! being SIMD.

use std::arch::x86_64::*;
use std::sync::OnceLock;

use super::activation::Quantized;
use super::half;
use super::k;

/// Whether this CPU has AVX2. Detected once, then free.
pub fn available() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| is_x86_feature_detected!("avx2"))
}

/// Whether this CPU has AVX2 + FMA + F16C for the 4-row register-tiled kernels.
pub fn available_fma_f16c() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("fma")
            && is_x86_feature_detected!("f16c")
    })
}

/// Horizontal sum of eight i32 lanes.
///
/// Exact regardless of order, which is the whole reason this file only does
/// integers.
#[target_feature(enable = "avx2")]
unsafe fn hsum(value: __m256i) -> i32 {
    let halves = _mm_add_epi32(
        _mm256_castsi256_si128(value),
        _mm256_extracti128_si256(value, 1),
    );
    let pairs = _mm_add_epi32(halves, _mm_shuffle_epi32(halves, 0b01_00_11_10));
    let single = _mm_add_epi32(pairs, _mm_shuffle_epi32(pairs, 0b00_01_00_01));
    _mm_cvtsi128_si32(single)
}

/// `Σ w[i]·a[i]` over 32 signed bytes each.
///
/// `maddubs` needs its first operand unsigned, so the sign is moved onto the
/// activation with `sign_epi8` and the weight is made absolute — the classic
/// trick, and the products stay inside i16: 127·127·2 = 32258.
#[target_feature(enable = "avx2")]
unsafe fn dot32(weights: *const i8, acts: *const i8) -> i32 {
    let w = _mm256_loadu_si256(weights as *const __m256i);
    let a = _mm256_loadu_si256(acts as *const __m256i);
    let magnitude = _mm256_sign_epi8(w, w);
    let signed = _mm256_sign_epi8(a, w);
    let pairs = _mm256_maddubs_epi16(magnitude, signed);
    hsum(_mm256_madd_epi16(pairs, _mm256_set1_epi16(1)))
}

/// `Σ n[i]·a[i]` where `n` are 32 unsigned nibbles already isolated in a
/// register. Products reach 15·127·2 = 3810, comfortably inside i16.
#[target_feature(enable = "avx2")]
unsafe fn dot_nibbles(nibbles: __m256i, acts: *const i8) -> i32 {
    let a = _mm256_loadu_si256(acts as *const __m256i);
    let pairs = _mm256_maddubs_epi16(nibbles, a);
    hsum(_mm256_madd_epi16(pairs, _mm256_set1_epi16(1)))
}

/// # Safety
/// Caller must have checked [`available`].
#[target_feature(enable = "avx2")]
pub unsafe fn q8_0(row: &[u8], x: &Quantized) -> f32 {
    let mut acc = 0.0f32;
    for (index, block) in row.chunks_exact(34).enumerate() {
        let d = half::read_f16(block);
        let (acts, scale, _) = x.block(index);
        let sum = dot32(block[2..].as_ptr() as *const i8, acts.as_ptr());
        acc += d * scale * sum as f32;
    }
    acc
}

/// # Safety
/// Caller must have checked [`available`].
///
/// Only the integer dot is vectorized; the per-sub-block combination stays in
/// scalar f32 and in the same order as the scalar kernel, so the two agree
/// exactly rather than approximately.
#[target_feature(enable = "avx2")]
pub unsafe fn q4_k(row: &[u8], x: &Quantized) -> f32 {
    let low_mask = _mm256_set1_epi8(0x0F);
    let mut acc = 0.0f32;

    for (super_index, block) in row.chunks_exact(144).enumerate() {
        let d = half::read_f16(block);
        let dmin = half::read_f16(&block[2..]);
        let scales = &block[4..16];
        let quants = &block[16..144];
        let first_sub = super_index * 8;

        for group in 0..4 {
            let bytes = _mm256_loadu_si256(quants[group * 32..].as_ptr() as *const __m256i);
            let low = _mm256_and_si256(bytes, low_mask);
            let high = _mm256_and_si256(_mm256_srli_epi16(bytes, 4), low_mask);

            let (scale_low, min_low) = k::scale_min(group * 2, scales);
            let (scale_high, min_high) = k::scale_min(group * 2 + 1, scales);
            let (acts_low, s_low, sum_low) = x.block(first_sub + group * 2);
            let (acts_high, s_high, sum_high) = x.block(first_sub + group * 2 + 1);

            let dot_low = dot_nibbles(low, acts_low.as_ptr());
            let dot_high = dot_nibbles(high, acts_high.as_ptr());

            acc += s_low
                * ((d * scale_low as f32) * dot_low as f32
                    - (dmin * min_low as f32) * sum_low as f32);
            acc += s_high
                * ((d * scale_high as f32) * dot_high as f32
                    - (dmin * min_high as f32) * sum_high as f32);
        }
    }
    acc
}

/// # Safety
/// Caller must have checked [`available`].
#[target_feature(enable = "avx2")]
pub unsafe fn q4_0(row: &[u8], x: &Quantized) -> f32 {
    let low_mask = _mm_set1_epi8(0x0F);
    let ones = _mm256_set1_epi16(1);
    let mut acc = 0.0f32;

    for (index, block) in row.chunks_exact(18).enumerate() {
        let d = half::read_f16(block);
        let quants = _mm_loadu_si128(block[2..18].as_ptr() as *const __m128i);
        let low = _mm_and_si128(quants, low_mask);
        let high = _mm_and_si128(_mm_srli_epi16(quants, 4), low_mask);
        let nibbles = _mm256_set_m128i(high, low);

        let (acts, scale, sum) = x.block(index);
        let a = _mm256_loadu_si256(acts.as_ptr() as *const __m256i);
        let pairs = _mm256_maddubs_epi16(nibbles, a);
        let dot = hsum(_mm256_madd_epi16(pairs, ones));

        acc += d * scale * (dot - 8 * sum) as f32;
    }
    acc
}

/// # Safety
/// Caller must have checked [`available`].
#[target_feature(enable = "avx2")]
pub unsafe fn q5_k(row: &[u8], x: &Quantized) -> f32 {
    let low_mask = _mm256_set1_epi8(0x0F);
    let bit16 = _mm256_set1_epi8(16);
    let zeros = _mm256_setzero_si256();
    let mut acc = 0.0f32;

    for (super_index, block) in row.chunks_exact(176).enumerate() {
        let d = half::read_f16(block);
        let dmin = half::read_f16(&block[2..]);
        let scales = &block[4..16];
        let qh = _mm256_loadu_si256(block[16..48].as_ptr() as *const __m256i);
        let qs = &block[48..176];
        let first_sub = super_index * 8;

        for group in 0..4 {
            let bytes = _mm256_loadu_si256(qs[group * 32..].as_ptr() as *const __m256i);
            let low4 = _mm256_and_si256(bytes, low_mask);
            let high4 = _mm256_and_si256(_mm256_srli_epi16(bytes, 4), low_mask);

            let mask_low = _mm256_set1_epi8(1i8 << (group * 2));
            let mask_high = _mm256_set1_epi8(2i8 << (group * 2));

            let has_low = _mm256_cmpeq_epi8(_mm256_and_si256(qh, mask_low), zeros);
            let fifth_low = _mm256_andnot_si256(has_low, bit16);
            let has_high = _mm256_cmpeq_epi8(_mm256_and_si256(qh, mask_high), zeros);
            let fifth_high = _mm256_andnot_si256(has_high, bit16);

            let low = _mm256_or_si256(low4, fifth_low);
            let high = _mm256_or_si256(high4, fifth_high);

            let (scale_low, min_low) = k::scale_min(group * 2, scales);
            let (scale_high, min_high) = k::scale_min(group * 2 + 1, scales);
            let (acts_low, s_low, sum_low) = x.block(first_sub + group * 2);
            let (acts_high, s_high, sum_high) = x.block(first_sub + group * 2 + 1);

            let dot_low = dot_nibbles(low, acts_low.as_ptr());
            let dot_high = dot_nibbles(high, acts_high.as_ptr());

            acc += s_low
                * ((d * scale_low as f32) * dot_low as f32
                    - (dmin * min_low as f32) * sum_low as f32);
            acc += s_high
                * ((d * scale_high as f32) * dot_high as f32
                    - (dmin * min_high as f32) * sum_high as f32);
        }
    }
    acc
}

/// Horizontal sum of four i32 lanes in a 128-bit register.
#[target_feature(enable = "avx2")]
unsafe fn hsum128(v: __m128i) -> i32 {
    let pairs = _mm_add_epi32(v, _mm_shuffle_epi32(v, 0b01_00_11_10));
    let single = _mm_add_epi32(pairs, _mm_shuffle_epi32(pairs, 0b00_01_00_01));
    _mm_cvtsi128_si32(single)
}

/// # Safety
/// Caller must have checked [`available`].
#[target_feature(enable = "avx2")]
pub unsafe fn q6_k(row: &[u8], x: &Quantized) -> f32 {
    let low_mask = _mm256_set1_epi8(0x0F);
    let high_mask = _mm256_set1_epi8(0x30);
    let offset32 = _mm256_set1_epi8(32);
    let ones128 = _mm_set1_epi16(1);
    let mut acc = 0.0f32;

    for (super_index, block) in row.chunks_exact(210).enumerate() {
        let ql = &block[0..128];
        let qh = &block[128..192];
        let scales = &block[192..208];
        let d = half::read_f16(&block[208..]);
        let first_sub = super_index * 8;

        for half_block in 0..2 {
            let ql_half = &ql[half_block * 64..(half_block + 1) * 64];
            let qh_vec = _mm256_loadu_si256(qh[half_block * 32..].as_ptr() as *const __m256i);
            let ql_0 = _mm256_loadu_si256(ql_half[0..32].as_ptr() as *const __m256i);
            let ql_1 = _mm256_loadu_si256(ql_half[32..64].as_ptr() as *const __m256i);
            let sc = &scales[half_block * 8..(half_block + 1) * 8];

            let q0 = _mm256_sub_epi8(
                _mm256_or_si256(
                    _mm256_and_si256(ql_0, low_mask),
                    _mm256_and_si256(_mm256_slli_epi16(qh_vec, 4), high_mask),
                ),
                offset32,
            );
            let q1 = _mm256_sub_epi8(
                _mm256_or_si256(
                    _mm256_and_si256(ql_1, low_mask),
                    _mm256_and_si256(_mm256_slli_epi16(qh_vec, 2), high_mask),
                ),
                offset32,
            );
            let q2 = _mm256_sub_epi8(
                _mm256_or_si256(
                    _mm256_and_si256(_mm256_srli_epi16(ql_0, 4), low_mask),
                    _mm256_and_si256(qh_vec, high_mask),
                ),
                offset32,
            );
            let q3 = _mm256_sub_epi8(
                _mm256_or_si256(
                    _mm256_and_si256(_mm256_srli_epi16(ql_1, 4), low_mask),
                    _mm256_and_si256(_mm256_srli_epi16(qh_vec, 2), high_mask),
                ),
                offset32,
            );

            let quads = [q0, q1, q2, q3];
            for (quad_idx, &w) in quads.iter().enumerate() {
                let sub_idx = first_sub + half_block * 4 + quad_idx;
                let (acts, s_act, _) = x.block(sub_idx);
                let a = _mm256_loadu_si256(acts.as_ptr() as *const __m256i);
                let mag = _mm256_sign_epi8(w, w);
                let signed = _mm256_sign_epi8(a, w);
                let pairs = _mm256_maddubs_epi16(mag, signed);

                let lo16 = _mm256_castsi256_si128(pairs);
                let hi16 = _mm256_extracti128_si256(pairs, 1);
                let dot_lo = hsum128(_mm_madd_epi16(lo16, ones128));
                let dot_hi = hsum128(_mm_madd_epi16(hi16, ones128));

                let sc_lo = sc[quad_idx * 2] as i8 as i32;
                let sc_hi = sc[quad_idx * 2 + 1] as i8 as i32;
                let weighted = sc_lo * dot_lo + sc_hi * dot_hi;
                acc += d * s_act * weighted as f32;
            }
        }
    }
    acc
}

/// Horizontal sum of 8 f32 lanes.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum_ps(v: __m256) -> f32 {
    let hi = _mm256_extractf128_ps(v, 1);
    let sum128 = _mm_add_ps(_mm256_castps256_ps128(v), hi);
    let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    let sum32 = _mm_add_ss(sum64, _mm_movehdup_ps(sum64));
    _mm_cvtss_f32(sum32)
}

/// Hardware F16C conversion of a 2-byte little-endian IEEE-754 half-precision float.
#[target_feature(enable = "avx2,f16c")]
#[inline]
unsafe fn f16c_read(ptr: *const u8) -> f32 {
    let bits = u16::from_le_bytes([*ptr, *ptr.add(1)]);
    let vec = _mm_cvtsi32_si128(bits as i32);
    _mm_cvtss_f32(_mm_cvtph_ps(vec))
}

/// Unpack all 8 sub-block 6-bit scales and 6-bit minimums from a 12-byte K-quant scale table.
#[inline(always)]
unsafe fn unpack_k_scales(scales: *const u8) -> ([i32; 8], [i32; 8]) {
    let s0 = *scales;
    let s1 = *scales.add(1);
    let s2 = *scales.add(2);
    let s3 = *scales.add(3);
    let s4 = *scales.add(4);
    let s5 = *scales.add(5);
    let s6 = *scales.add(6);
    let s7 = *scales.add(7);
    let s8 = *scales.add(8);
    let s9 = *scales.add(9);
    let s10 = *scales.add(10);
    let s11 = *scales.add(11);

    let sc = [
        (s0 & 63) as i32,
        (s1 & 63) as i32,
        (s2 & 63) as i32,
        (s3 & 63) as i32,
        ((s8 & 0x0F) | ((s0 >> 6) << 4)) as i32,
        ((s9 & 0x0F) | ((s1 >> 6) << 4)) as i32,
        ((s10 & 0x0F) | ((s2 >> 6) << 4)) as i32,
        ((s11 & 0x0F) | ((s3 >> 6) << 4)) as i32,
    ];
    let m = [
        (s4 & 63) as i32,
        (s5 & 63) as i32,
        (s6 & 63) as i32,
        (s7 & 63) as i32,
        ((s8 >> 4) | ((s4 >> 6) << 4)) as i32,
        ((s9 >> 4) | ((s5 >> 6) << 4)) as i32,
        ((s10 >> 4) | ((s6 >> 6) << 4)) as i32,
        ((s11 >> 4) | ((s7 >> 6) << 4)) as i32,
    ];
    (sc, m)
}

/// 4-row register-tiled `Q8_0` dot product using AVX2 + FMA + F16C.
///
/// Accumulates in four 8-lane `__m256` registers across all blocks and performs
/// only a single horizontal reduction at the very end of the row.
///
/// # Safety
/// Caller must have checked [`available`].
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn q8_0_4rows(rows: [&[u8]; 4], x: &Quantized) -> [f32; 4] {
    let ones = _mm256_set1_epi16(1);
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();

    let n_blocks = rows[0].len() / 34;
    let r0 = rows[0].as_ptr();
    let r1 = rows[1].as_ptr();
    let r2 = rows[2].as_ptr();
    let r3 = rows[3].as_ptr();
    let q_ptr = x.quants.as_ptr();
    let s_ptr = x.scales.as_ptr();

    for index in 0..n_blocks {
        let offset = index * 34;
        let b0 = r0.add(offset);
        let b1 = r1.add(offset);
        let b2 = r2.add(offset);
        let b3 = r3.add(offset);

        let scale = *s_ptr.add(index);
        let d0 = _mm256_set1_ps(f16c_read(b0) * scale);
        let d1 = _mm256_set1_ps(f16c_read(b1) * scale);
        let d2 = _mm256_set1_ps(f16c_read(b2) * scale);
        let d3 = _mm256_set1_ps(f16c_read(b3) * scale);

        let a = _mm256_loadu_si256(q_ptr.add(index * 32) as *const __m256i);

        let w0 = _mm256_loadu_si256(b0.add(2) as *const __m256i);
        let w1 = _mm256_loadu_si256(b1.add(2) as *const __m256i);
        let w2 = _mm256_loadu_si256(b2.add(2) as *const __m256i);
        let w3 = _mm256_loadu_si256(b3.add(2) as *const __m256i);

        let p0 = _mm256_madd_epi16(
            _mm256_maddubs_epi16(_mm256_sign_epi8(w0, w0), _mm256_sign_epi8(a, w0)),
            ones,
        );
        let p1 = _mm256_madd_epi16(
            _mm256_maddubs_epi16(_mm256_sign_epi8(w1, w1), _mm256_sign_epi8(a, w1)),
            ones,
        );
        let p2 = _mm256_madd_epi16(
            _mm256_maddubs_epi16(_mm256_sign_epi8(w2, w2), _mm256_sign_epi8(a, w2)),
            ones,
        );
        let p3 = _mm256_madd_epi16(
            _mm256_maddubs_epi16(_mm256_sign_epi8(w3, w3), _mm256_sign_epi8(a, w3)),
            ones,
        );

        acc0 = _mm256_fmadd_ps(d0, _mm256_cvtepi32_ps(p0), acc0);
        acc1 = _mm256_fmadd_ps(d1, _mm256_cvtepi32_ps(p1), acc1);
        acc2 = _mm256_fmadd_ps(d2, _mm256_cvtepi32_ps(p2), acc2);
        acc3 = _mm256_fmadd_ps(d3, _mm256_cvtepi32_ps(p3), acc3);
    }
    [hsum_ps(acc0), hsum_ps(acc1), hsum_ps(acc2), hsum_ps(acc3)]
}

/// 4-row register-tiled `Q4_0` dot product using AVX2 + FMA + F16C.
///
/// Subtracts the center offset `8` directly in `epi8` (`_mm256_sub_epi8`) and
/// accumulates in `__m256` vector registers across the entire row.
///
/// # Safety
/// Caller must have checked [`available`].
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn q4_0_4rows(rows: [&[u8]; 4], x: &Quantized) -> [f32; 4] {
    let low_mask = _mm_set1_epi8(0x0F);
    let off8 = _mm256_set1_epi8(8);
    let ones = _mm256_set1_epi16(1);
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();

    let n_blocks = rows[0].len() / 18;
    let r0 = rows[0].as_ptr();
    let r1 = rows[1].as_ptr();
    let r2 = rows[2].as_ptr();
    let r3 = rows[3].as_ptr();
    let q_ptr = x.quants.as_ptr();
    let s_ptr = x.scales.as_ptr();

    for index in 0..n_blocks {
        let offset = index * 18;
        let b0 = r0.add(offset);
        let b1 = r1.add(offset);
        let b2 = r2.add(offset);
        let b3 = r3.add(offset);

        let scale = *s_ptr.add(index);
        let d0 = _mm256_set1_ps(f16c_read(b0) * scale);
        let d1 = _mm256_set1_ps(f16c_read(b1) * scale);
        let d2 = _mm256_set1_ps(f16c_read(b2) * scale);
        let d3 = _mm256_set1_ps(f16c_read(b3) * scale);

        let a = _mm256_loadu_si256(q_ptr.add(index * 32) as *const __m256i);

        let dot_block = |ptr: *const u8| -> __m256i {
            let raw = _mm_loadu_si128(ptr.add(2) as *const __m128i);
            let lo = _mm_and_si128(raw, low_mask);
            let hi = _mm_and_si128(_mm_srli_epi16(raw, 4), low_mask);
            let w = _mm256_sub_epi8(_mm256_set_m128i(hi, lo), off8);
            let pairs = _mm256_maddubs_epi16(_mm256_sign_epi8(w, w), _mm256_sign_epi8(a, w));
            _mm256_madd_epi16(pairs, ones)
        };

        acc0 = _mm256_fmadd_ps(d0, _mm256_cvtepi32_ps(dot_block(b0)), acc0);
        acc1 = _mm256_fmadd_ps(d1, _mm256_cvtepi32_ps(dot_block(b1)), acc1);
        acc2 = _mm256_fmadd_ps(d2, _mm256_cvtepi32_ps(dot_block(b2)), acc2);
        acc3 = _mm256_fmadd_ps(d3, _mm256_cvtepi32_ps(dot_block(b3)), acc3);
    }
    [hsum_ps(acc0), hsum_ps(acc1), hsum_ps(acc2), hsum_ps(acc3)]
}

/// 4-row register-tiled `Q4_K` dot product using AVX2 + FMA + F16C.
///
/// Accumulates all 8 sub-blocks per super-block in `__m256` vector registers
/// without any inner-loop `hsum` calls.
///
/// # Safety
/// Caller must have checked [`available`].
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn q4_k_4rows(rows: [&[u8]; 4], x: &Quantized) -> [f32; 4] {
    let low_mask = _mm256_set1_epi8(0x0F);
    let ones = _mm256_set1_epi16(1);
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();
    let mut min_acc0 = 0.0f32;
    let mut min_acc1 = 0.0f32;
    let mut min_acc2 = 0.0f32;
    let mut min_acc3 = 0.0f32;

    let n_super = rows[0].len() / 144;
    let r0 = rows[0].as_ptr();
    let r1 = rows[1].as_ptr();
    let r2 = rows[2].as_ptr();
    let r3 = rows[3].as_ptr();
    let q_ptr = x.quants.as_ptr();
    let s_ptr = x.scales.as_ptr();
    let sum_ptr = x.sums.as_ptr();

    for super_index in 0..n_super {
        let off = super_index * 144;
        let b0 = r0.add(off);
        let b1 = r1.add(off);
        let b2 = r2.add(off);
        let b3 = r3.add(off);

        let d0 = f16c_read(b0);
        let dmin0 = f16c_read(b0.add(2));
        let d1 = f16c_read(b1);
        let dmin1 = f16c_read(b1.add(2));
        let d2 = f16c_read(b2);
        let dmin2 = f16c_read(b2.add(2));
        let d3 = f16c_read(b3);
        let dmin3 = f16c_read(b3.add(2));

        let (sc0, m0) = unpack_k_scales(b0.add(4));
        let (sc1, m1) = unpack_k_scales(b1.add(4));
        let (sc2, m2) = unpack_k_scales(b2.add(4));
        let (sc3, m3) = unpack_k_scales(b3.add(4));

        let first_sub = super_index * 8;

        for group in 0..4 {
            let sub_lo = first_sub + group * 2;
            let sub_hi = sub_lo + 1;
            let s_low = *s_ptr.add(sub_lo);
            let s_high = *s_ptr.add(sub_hi);
            let sum_low_f = (*sum_ptr.add(sub_lo) as f32) * s_low;
            let sum_high_f = (*sum_ptr.add(sub_hi) as f32) * s_high;

            let a_low = _mm256_loadu_si256(q_ptr.add(sub_lo * 32) as *const __m256i);
            let a_high = _mm256_loadu_si256(q_ptr.add(sub_hi * 32) as *const __m256i);

            let q_off = 16 + group * 32;

            macro_rules! step_row {
                ($b:ident, $d:ident, $dmin:ident, $sc:ident, $m:ident, $acc:ident, $min_acc:ident) => {{
                    let bytes = _mm256_loadu_si256($b.add(q_off) as *const __m256i);
                    let low = _mm256_and_si256(bytes, low_mask);
                    let high = _mm256_and_si256(_mm256_srli_epi16(bytes, 4), low_mask);
                    let p_lo = _mm256_madd_epi16(_mm256_maddubs_epi16(low, a_low), ones);
                    let p_hi = _mm256_madd_epi16(_mm256_maddubs_epi16(high, a_high), ones);
                    let w_lo = _mm256_set1_ps($d * s_low * ($sc[group * 2] as f32));
                    let w_hi = _mm256_set1_ps($d * s_high * ($sc[group * 2 + 1] as f32));
                    $acc = _mm256_fmadd_ps(w_lo, _mm256_cvtepi32_ps(p_lo), $acc);
                    $acc = _mm256_fmadd_ps(w_hi, _mm256_cvtepi32_ps(p_hi), $acc);
                    $min_acc += $dmin
                        * (($m[group * 2] as f32) * sum_low_f
                            + ($m[group * 2 + 1] as f32) * sum_high_f);
                }};
            }

            step_row!(b0, d0, dmin0, sc0, m0, acc0, min_acc0);
            step_row!(b1, d1, dmin1, sc1, m1, acc1, min_acc1);
            step_row!(b2, d2, dmin2, sc2, m2, acc2, min_acc2);
            step_row!(b3, d3, dmin3, sc3, m3, acc3, min_acc3);
        }
    }
    [
        hsum_ps(acc0) - min_acc0,
        hsum_ps(acc1) - min_acc1,
        hsum_ps(acc2) - min_acc2,
        hsum_ps(acc3) - min_acc3,
    ]
}

/// 4-row register-tiled `Q5_K` dot product using AVX2 + FMA + F16C.
///
/// # Safety
/// Caller must have checked [`available`].
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn q5_k_4rows(rows: [&[u8]; 4], x: &Quantized) -> [f32; 4] {
    let low_mask = _mm256_set1_epi8(0x0F);
    let bit16 = _mm256_set1_epi8(16);
    let zeros = _mm256_setzero_si256();
    let ones = _mm256_set1_epi16(1);
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();
    let mut min_acc0 = 0.0f32;
    let mut min_acc1 = 0.0f32;
    let mut min_acc2 = 0.0f32;
    let mut min_acc3 = 0.0f32;

    let n_super = rows[0].len() / 176;
    let r0 = rows[0].as_ptr();
    let r1 = rows[1].as_ptr();
    let r2 = rows[2].as_ptr();
    let r3 = rows[3].as_ptr();
    let q_ptr = x.quants.as_ptr();
    let s_ptr = x.scales.as_ptr();
    let sum_ptr = x.sums.as_ptr();

    for super_index in 0..n_super {
        let off = super_index * 176;
        let b0 = r0.add(off);
        let b1 = r1.add(off);
        let b2 = r2.add(off);
        let b3 = r3.add(off);

        let d0 = f16c_read(b0);
        let dmin0 = f16c_read(b0.add(2));
        let d1 = f16c_read(b1);
        let dmin1 = f16c_read(b1.add(2));
        let d2 = f16c_read(b2);
        let dmin2 = f16c_read(b2.add(2));
        let d3 = f16c_read(b3);
        let dmin3 = f16c_read(b3.add(2));

        let (sc0, m0) = unpack_k_scales(b0.add(4));
        let (sc1, m1) = unpack_k_scales(b1.add(4));
        let (sc2, m2) = unpack_k_scales(b2.add(4));
        let (sc3, m3) = unpack_k_scales(b3.add(4));

        let qh0 = _mm256_loadu_si256(b0.add(16) as *const __m256i);
        let qh1 = _mm256_loadu_si256(b1.add(16) as *const __m256i);
        let qh2 = _mm256_loadu_si256(b2.add(16) as *const __m256i);
        let qh3 = _mm256_loadu_si256(b3.add(16) as *const __m256i);

        let first_sub = super_index * 8;

        for group in 0..4 {
            let sub_lo = first_sub + group * 2;
            let sub_hi = sub_lo + 1;
            let s_low = *s_ptr.add(sub_lo);
            let s_high = *s_ptr.add(sub_hi);
            let sum_low_f = (*sum_ptr.add(sub_lo) as f32) * s_low;
            let sum_high_f = (*sum_ptr.add(sub_hi) as f32) * s_high;

            let a_low = _mm256_loadu_si256(q_ptr.add(sub_lo * 32) as *const __m256i);
            let a_high = _mm256_loadu_si256(q_ptr.add(sub_hi * 32) as *const __m256i);

            let mask_low = _mm256_set1_epi8(1i8 << (group * 2));
            let mask_high = _mm256_set1_epi8(2i8 << (group * 2));
            let q_off = 48 + group * 32;

            macro_rules! step_q5_row {
                ($b:ident, $qh:ident, $d:ident, $dmin:ident, $sc:ident, $m:ident, $acc:ident, $min_acc:ident) => {{
                    let bytes = _mm256_loadu_si256($b.add(q_off) as *const __m256i);
                    let low4 = _mm256_and_si256(bytes, low_mask);
                    let high4 = _mm256_and_si256(_mm256_srli_epi16(bytes, 4), low_mask);

                    let has_low = _mm256_cmpeq_epi8(_mm256_and_si256($qh, mask_low), zeros);
                    let fifth_low = _mm256_andnot_si256(has_low, bit16);
                    let has_high = _mm256_cmpeq_epi8(_mm256_and_si256($qh, mask_high), zeros);
                    let fifth_high = _mm256_andnot_si256(has_high, bit16);

                    let low = _mm256_or_si256(low4, fifth_low);
                    let high = _mm256_or_si256(high4, fifth_high);

                    let p_lo = _mm256_madd_epi16(_mm256_maddubs_epi16(low, a_low), ones);
                    let p_hi = _mm256_madd_epi16(_mm256_maddubs_epi16(high, a_high), ones);
                    let w_lo = _mm256_set1_ps($d * s_low * ($sc[group * 2] as f32));
                    let w_hi = _mm256_set1_ps($d * s_high * ($sc[group * 2 + 1] as f32));
                    $acc = _mm256_fmadd_ps(w_lo, _mm256_cvtepi32_ps(p_lo), $acc);
                    $acc = _mm256_fmadd_ps(w_hi, _mm256_cvtepi32_ps(p_hi), $acc);
                    $min_acc += $dmin
                        * (($m[group * 2] as f32) * sum_low_f
                            + ($m[group * 2 + 1] as f32) * sum_high_f);
                }};
            }

            step_q5_row!(b0, qh0, d0, dmin0, sc0, m0, acc0, min_acc0);
            step_q5_row!(b1, qh1, d1, dmin1, sc1, m1, acc1, min_acc1);
            step_q5_row!(b2, qh2, d2, dmin2, sc2, m2, acc2, min_acc2);
            step_q5_row!(b3, qh3, d3, dmin3, sc3, m3, acc3, min_acc3);
        }
    }
    [
        hsum_ps(acc0) - min_acc0,
        hsum_ps(acc1) - min_acc1,
        hsum_ps(acc2) - min_acc2,
        hsum_ps(acc3) - min_acc3,
    ]
}

/// 4-row register-tiled `Q6_K` dot product using AVX2 + FMA + F16C.
///
/// Multiplies each 16-element half-lane by its signed 8-bit scale directly in
/// 16-bit SIMD (`_mm256_madd_epi16(pairs, scale_16_vec)`) before accumulating
/// into `__m256` FMA registers, eliminating all inner-loop horizontal reductions.
///
/// # Safety
/// Caller must have checked [`available`].
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn q6_k_4rows(rows: [&[u8]; 4], x: &Quantized) -> [f32; 4] {
    let low_mask = _mm256_set1_epi8(0x0F);
    let high_mask = _mm256_set1_epi8(0x30);
    let offset32 = _mm256_set1_epi8(32);
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();

    let n_super = rows[0].len() / 210;
    let r0 = rows[0].as_ptr();
    let r1 = rows[1].as_ptr();
    let r2 = rows[2].as_ptr();
    let r3 = rows[3].as_ptr();
    let q_ptr = x.quants.as_ptr();
    let s_ptr = x.scales.as_ptr();

    for super_index in 0..n_super {
        let off = super_index * 210;
        let b0 = r0.add(off);
        let b1 = r1.add(off);
        let b2 = r2.add(off);
        let b3 = r3.add(off);

        let d0 = f16c_read(b0.add(208));
        let d1 = f16c_read(b1.add(208));
        let d2 = f16c_read(b2.add(208));
        let d3 = f16c_read(b3.add(208));
        let first_sub = super_index * 8;

        for half_block in 0..2 {
            let sub_base = first_sub + half_block * 4;
            let s0 = *s_ptr.add(sub_base);
            let s1 = *s_ptr.add(sub_base + 1);
            let s2 = *s_ptr.add(sub_base + 2);
            let s3 = *s_ptr.add(sub_base + 3);
            let s_vec = [s0, s1, s2, s3];

            let a0 = _mm256_loadu_si256(q_ptr.add(sub_base * 32) as *const __m256i);
            let a1 = _mm256_loadu_si256(q_ptr.add((sub_base + 1) * 32) as *const __m256i);
            let a2 = _mm256_loadu_si256(q_ptr.add((sub_base + 2) * 32) as *const __m256i);
            let a3 = _mm256_loadu_si256(q_ptr.add((sub_base + 3) * 32) as *const __m256i);
            let acts_vec = [a0, a1, a2, a3];

            let ql_off = half_block * 64;
            let qh_off = 128 + half_block * 32;
            let sc_off = 192 + half_block * 8;

            macro_rules! step_q6_half {
                ($b:ident, $d:ident, $acc:ident) => {{
                    let qh_vec = _mm256_loadu_si256($b.add(qh_off) as *const __m256i);
                    let ql_0 = _mm256_loadu_si256($b.add(ql_off) as *const __m256i);
                    let ql_1 = _mm256_loadu_si256($b.add(ql_off + 32) as *const __m256i);
                    let sc = $b.add(sc_off) as *const i8;

                    let q0 = _mm256_sub_epi8(
                        _mm256_or_si256(
                            _mm256_and_si256(ql_0, low_mask),
                            _mm256_and_si256(_mm256_slli_epi16(qh_vec, 4), high_mask),
                        ),
                        offset32,
                    );
                    let q1 = _mm256_sub_epi8(
                        _mm256_or_si256(
                            _mm256_and_si256(ql_1, low_mask),
                            _mm256_and_si256(_mm256_slli_epi16(qh_vec, 2), high_mask),
                        ),
                        offset32,
                    );
                    let q2 = _mm256_sub_epi8(
                        _mm256_or_si256(
                            _mm256_and_si256(_mm256_srli_epi16(ql_0, 4), low_mask),
                            _mm256_and_si256(qh_vec, high_mask),
                        ),
                        offset32,
                    );
                    let q3 = _mm256_sub_epi8(
                        _mm256_or_si256(
                            _mm256_and_si256(_mm256_srli_epi16(ql_1, 4), low_mask),
                            _mm256_and_si256(_mm256_srli_epi16(qh_vec, 2), high_mask),
                        ),
                        offset32,
                    );

                    let quads = [q0, q1, q2, q3];
                    for quad_idx in 0..4 {
                        let w = quads[quad_idx];
                        let a = acts_vec[quad_idx];
                        let mag = _mm256_sign_epi8(w, w);
                        let signed = _mm256_sign_epi8(a, w);
                        let pairs = _mm256_maddubs_epi16(mag, signed);

                        let sc_lo = *sc.add(quad_idx * 2) as i16;
                        let sc_hi = *sc.add(quad_idx * 2 + 1) as i16;
                        let sc_vec = _mm256_set_m128i(_mm_set1_epi16(sc_hi), _mm_set1_epi16(sc_lo));
                        let scaled_i32 = _mm256_madd_epi16(pairs, sc_vec);
                        let scale_f = _mm256_set1_ps($d * s_vec[quad_idx]);
                        $acc = _mm256_fmadd_ps(scale_f, _mm256_cvtepi32_ps(scaled_i32), $acc);
                    }
                }};
            }

            step_q6_half!(b0, d0, acc0);
            step_q6_half!(b1, d1, acc1);
            step_q6_half!(b2, d2, acc2);
            step_q6_half!(b3, d3, acc3);
        }
    }
    [hsum_ps(acc0), hsum_ps(acc1), hsum_ps(acc2), hsum_ps(acc3)]
}

/// # Safety
/// Caller must have checked [`available`].
#[target_feature(enable = "avx2")]
pub unsafe fn q4_1(row: &[u8], x: &Quantized) -> f32 {
    let low_mask = _mm_set1_epi8(0x0F);
    let ones = _mm256_set1_epi16(1);
    let mut acc = 0.0f32;

    for (index, block) in row.chunks_exact(20).enumerate() {
        let d = half::read_f16(block);
        let min = half::read_f16(&block[2..4]);
        let quants = _mm_loadu_si128(block[4..20].as_ptr() as *const __m128i);
        let low = _mm_and_si128(quants, low_mask);
        let high = _mm_and_si128(_mm_srli_epi16(quants, 4), low_mask);
        let nibbles = _mm256_set_m128i(high, low);

        let (acts, scale, sum) = x.block(index);
        let a = _mm256_loadu_si256(acts.as_ptr() as *const __m256i);
        let pairs = _mm256_maddubs_epi16(nibbles, a);
        let dot = hsum(_mm256_madd_epi16(pairs, ones));

        acc += scale * (d * dot as f32 + min * sum as f32);
    }
    acc
}

/// Unpack a 32-bit `qh` mask into 32 bytes where each byte is `0x10` if the
/// corresponding bit in `qh` is set, or `0x00` otherwise.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn unpack_qh_u32(qh: u32) -> __m256i {
    let b0 = _mm256_set1_epi8((qh & 0xFF) as i8);
    let b1 = _mm256_set1_epi8(((qh >> 8) & 0xFF) as i8);
    let b2 = _mm256_set1_epi8(((qh >> 16) & 0xFF) as i8);
    let b3 = _mm256_set1_epi8(((qh >> 24) & 0xFF) as i8);

    // Combine into a 256-bit vector where bytes 0..8 have b0, 8..16 have b1,
    // 16..24 have b2, and 24..32 have b3:
    let lo128 = _mm_unpacklo_epi64(_mm256_castsi256_si128(b0), _mm256_castsi256_si128(b1));
    let hi128 = _mm_unpacklo_epi64(_mm256_castsi256_si128(b2), _mm256_castsi256_si128(b3));
    let bytes = _mm256_set_m128i(hi128, lo128);

    let bit_mask = _mm256_setr_epi8(
        1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64, -128,
        1, 2, 4, 8, 16, 32, 64, -128,
    );
    let is_zero = _mm256_cmpeq_epi8(_mm256_and_si256(bytes, bit_mask), _mm256_setzero_si256());
    _mm256_andnot_si256(is_zero, _mm256_set1_epi8(0x10))
}

/// # Safety
/// Caller must have checked [`available`].
#[target_feature(enable = "avx2")]
pub unsafe fn q5_0(row: &[u8], x: &Quantized) -> f32 {
    let low_mask = _mm_set1_epi8(0x0F);
    let ones = _mm256_set1_epi16(1);
    let mut acc = 0.0f32;

    for (index, block) in row.chunks_exact(22).enumerate() {
        let d = half::read_f16(block);
        let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
        let quants = _mm_loadu_si128(block[6..22].as_ptr() as *const __m128i);
        let low = _mm_and_si128(quants, low_mask);
        let high = _mm_and_si128(_mm_srli_epi16(quants, 4), low_mask);
        let nibbles = _mm256_or_si256(_mm256_set_m128i(high, low), unpack_qh_u32(qh));

        let (acts, scale, sum) = x.block(index);
        let a = _mm256_loadu_si256(acts.as_ptr() as *const __m256i);
        let pairs = _mm256_maddubs_epi16(nibbles, a);
        let dot = hsum(_mm256_madd_epi16(pairs, ones));

        acc += d * scale * (dot - 16 * sum) as f32;
    }
    acc
}

/// # Safety
/// Caller must have checked [`available`].
#[target_feature(enable = "avx2")]
pub unsafe fn q5_1(row: &[u8], x: &Quantized) -> f32 {
    let low_mask = _mm_set1_epi8(0x0F);
    let ones = _mm256_set1_epi16(1);
    let mut acc = 0.0f32;

    for (index, block) in row.chunks_exact(24).enumerate() {
        let d = half::read_f16(block);
        let min = half::read_f16(&block[2..4]);
        let qh = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
        let quants = _mm_loadu_si128(block[8..24].as_ptr() as *const __m128i);
        let low = _mm_and_si128(quants, low_mask);
        let high = _mm_and_si128(_mm_srli_epi16(quants, 4), low_mask);
        let nibbles = _mm256_or_si256(_mm256_set_m128i(high, low), unpack_qh_u32(qh));

        let (acts, scale, sum) = x.block(index);
        let a = _mm256_loadu_si256(acts.as_ptr() as *const __m256i);
        let pairs = _mm256_maddubs_epi16(nibbles, a);
        let dot = hsum(_mm256_madd_epi16(pairs, ones));

        acc += scale * (d * dot as f32 + min * sum as f32);
    }
    acc
}

/// # Safety
/// Caller must have checked [`available`].
#[target_feature(enable = "avx2")]
pub unsafe fn q2_k(row: &[u8], x: &Quantized) -> f32 {
    let mask2 = _mm256_set1_epi8(0x03);
    let ones128 = _mm_set1_epi16(1);
    let ones_u8 = _mm256_set1_epi8(1);
    let mut acc = 0.0f32;

    for (super_index, block) in row.chunks_exact(84).enumerate() {
        let scales = &block[0..16];
        let qs = &block[16..80];
        let d = half::read_f16(&block[80..82]);
        let dmin = half::read_f16(&block[82..84]);
        let first_sub = super_index * 8;

        for half_block in 0..2 {
            let q_vec = _mm256_loadu_si256(qs[half_block * 32..].as_ptr() as *const __m256i);
            for shift_idx in 0..4 {
                let sub_idx = first_sub + half_block * 4 + shift_idx;
                let (acts, s_act, _) = x.block(sub_idx);
                let a = _mm256_loadu_si256(acts.as_ptr() as *const __m256i);

                let shifted = match shift_idx {
                    0 => q_vec,
                    1 => _mm256_srli_epi16(q_vec, 2),
                    2 => _mm256_srli_epi16(q_vec, 4),
                    _ => _mm256_srli_epi16(q_vec, 6),
                };
                let w = _mm256_and_si256(shifted, mask2);
                let pairs = _mm256_maddubs_epi16(w, a);
                let dot_lo = hsum128(_mm_madd_epi16(_mm256_castsi256_si128(pairs), ones128));
                let dot_hi = hsum128(_mm_madd_epi16(_mm256_extracti128_si256(pairs, 1), ones128));

                let sum_pairs = _mm256_maddubs_epi16(ones_u8, a);
                let sum_lo = hsum128(_mm_madd_epi16(_mm256_castsi256_si128(sum_pairs), ones128));
                let sum_hi = hsum128(_mm_madd_epi16(
                    _mm256_extracti128_si256(sum_pairs, 1),
                    ones128,
                ));

                let sc_lo = scales[half_block * 8 + shift_idx * 2];
                let sc_hi = scales[half_block * 8 + shift_idx * 2 + 1];
                let weighted_q = (sc_lo & 0x0F) as i32 * dot_lo + (sc_hi & 0x0F) as i32 * dot_hi;
                let weighted_m = (sc_lo >> 4) as i32 * sum_lo + (sc_hi >> 4) as i32 * sum_hi;
                acc += s_act * (d * weighted_q as f32 - dmin * weighted_m as f32);
            }
        }
    }
    acc
}

/// # Safety
/// Caller must have checked [`available`].
#[target_feature(enable = "avx2")]
pub unsafe fn q3_k(row: &[u8], x: &Quantized) -> f32 {
    let mask2 = _mm256_set1_epi8(0x03);
    let bit4 = _mm256_set1_epi8(4);
    let zeros = _mm256_setzero_si256();
    let ones128 = _mm_set1_epi16(1);
    let mut acc = 0.0f32;

    for (super_index, block) in row.chunks_exact(110).enumerate() {
        let hmask = _mm256_loadu_si256(block[0..32].as_ptr() as *const __m256i);
        let qs = &block[32..96];
        let scales = k::q3_k_scales(&block[96..108]);
        let d = half::read_f16(&block[108..110]);
        let first_sub = super_index * 8;

        for half_block in 0..2 {
            let q_vec = _mm256_loadu_si256(qs[half_block * 32..].as_ptr() as *const __m256i);
            for shift_idx in 0..4 {
                let sub_idx = first_sub + half_block * 4 + shift_idx;
                let (acts, s_act, _) = x.block(sub_idx);
                let a = _mm256_loadu_si256(acts.as_ptr() as *const __m256i);

                let shifted = match shift_idx {
                    0 => q_vec,
                    1 => _mm256_srli_epi16(q_vec, 2),
                    2 => _mm256_srli_epi16(q_vec, 4),
                    _ => _mm256_srli_epi16(q_vec, 6),
                };
                let low2 = _mm256_and_si256(shifted, mask2);
                let m_bit = _mm256_set1_epi8(1i8 << (half_block * 4 + shift_idx));
                let is_zero = _mm256_cmpeq_epi8(_mm256_and_si256(hmask, m_bit), zeros);
                let sub_four = _mm256_and_si256(is_zero, bit4);
                let w = _mm256_sub_epi8(low2, sub_four);

                let mag = _mm256_sign_epi8(w, w);
                let signed = _mm256_sign_epi8(a, w);
                let pairs = _mm256_maddubs_epi16(mag, signed);

                let dot_lo = hsum128(_mm_madd_epi16(_mm256_castsi256_si128(pairs), ones128));
                let dot_hi = hsum128(_mm_madd_epi16(_mm256_extracti128_si256(pairs, 1), ones128));

                let sc_lo = scales[half_block * 8 + shift_idx * 2] as i32;
                let sc_hi = scales[half_block * 8 + shift_idx * 2 + 1] as i32;
                let weighted = sc_lo * dot_lo + sc_hi * dot_hi;
                acc += d * s_act * weighted as f32;
            }
        }
    }
    acc
}

/// AVX2 non-linear lookup table (`KVALUES_IQ4NL` duplicated across both 128-bit lanes).
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn iq4nl_lut() -> __m256i {
    _mm256_setr_epi8(
        -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113, -127, -104, -83,
        -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
    )
}

/// # Safety
/// Caller must have checked [`available`].
#[target_feature(enable = "avx2")]
pub unsafe fn iq4_nl(row: &[u8], x: &Quantized) -> f32 {
    let lut = iq4nl_lut();
    let low_mask = _mm_set1_epi8(0x0F);
    let ones = _mm256_set1_epi16(1);
    let mut acc = 0.0f32;

    for (index, block) in row.chunks_exact(18).enumerate() {
        let d = half::read_f16(block);
        let raw = _mm_loadu_si128(block[2..18].as_ptr() as *const __m128i);
        let lo = _mm_and_si128(raw, low_mask);
        let hi = _mm_and_si128(_mm_srli_epi16(raw, 4), low_mask);
        let indices = _mm256_set_m128i(hi, lo);
        let w = _mm256_shuffle_epi8(lut, indices);

        let (acts, scale, _) = x.block(index);
        let a = _mm256_loadu_si256(acts.as_ptr() as *const __m256i);
        let mag = _mm256_sign_epi8(w, w);
        let signed = _mm256_sign_epi8(a, w);
        let pairs = _mm256_maddubs_epi16(mag, signed);
        let dot = hsum(_mm256_madd_epi16(pairs, ones));
        acc += d * scale * dot as f32;
    }
    acc
}

/// # Safety
/// Caller must have checked [`available`].
#[target_feature(enable = "avx2")]
pub unsafe fn iq4_xs(row: &[u8], x: &Quantized) -> f32 {
    let lut = iq4nl_lut();
    let low_mask = _mm_set1_epi8(0x0F);
    let ones = _mm256_set1_epi16(1);
    let mut acc = 0.0f32;

    for (super_index, block) in row.chunks_exact(136).enumerate() {
        let d = half::read_f16(&block[0..2]);
        let scales_h = u16::from_le_bytes([block[2], block[3]]);
        let scales = super::iq::iq4_xs_scales(scales_h, &block[4..8]);
        let qs = &block[8..136];
        let first_sub = super_index * 8;

        for ib in 0..8 {
            let (acts, s_act, _) = x.block(first_sub + ib);
            let raw = _mm_loadu_si128(qs[ib * 16..].as_ptr() as *const __m128i);
            let lo = _mm_and_si128(raw, low_mask);
            let hi = _mm_and_si128(_mm_srli_epi16(raw, 4), low_mask);
            let indices = _mm256_set_m128i(hi, lo);
            let w = _mm256_shuffle_epi8(lut, indices);

            let a = _mm256_loadu_si256(acts.as_ptr() as *const __m256i);
            let mag = _mm256_sign_epi8(w, w);
            let signed = _mm256_sign_epi8(a, w);
            let pairs = _mm256_maddubs_epi16(mag, signed);
            let dot = hsum(_mm256_madd_epi16(pairs, ones));
            acc += d * s_act * (scales[ib] as i32 * dot) as f32;
        }
    }
    acc
}

/// # Safety
/// Caller must have checked [`available`].
#[target_feature(enable = "avx2")]
pub unsafe fn tq2_0(row: &[u8], x: &Quantized) -> f32 {
    let mask2 = _mm256_set1_epi8(0x03);
    let ones_i8 = _mm256_set1_epi8(1);
    let ones16 = _mm256_set1_epi16(1);
    let mut acc = 0.0f32;

    for (super_index, block) in row.chunks_exact(66).enumerate() {
        let qs = &block[0..64];
        let d = half::read_f16(&block[64..66]);
        let first_sub = super_index * 8;

        for half_block in 0..2 {
            let q_vec = _mm256_loadu_si256(qs[half_block * 32..].as_ptr() as *const __m256i);
            for shift_idx in 0..4 {
                let sub_idx = first_sub + half_block * 4 + shift_idx;
                let (acts, s_act, _) = x.block(sub_idx);
                let a = _mm256_loadu_si256(acts.as_ptr() as *const __m256i);

                let shifted = match shift_idx {
                    0 => q_vec,
                    1 => _mm256_srli_epi16(q_vec, 2),
                    2 => _mm256_srli_epi16(q_vec, 4),
                    _ => _mm256_srli_epi16(q_vec, 6),
                };
                // trit in {-1, 0, +1}:
                let trit = _mm256_sub_epi8(_mm256_and_si256(shifted, mask2), ones_i8);
                // Multiplication-free ternary sign application:
                let signed_a = _mm256_sign_epi8(a, trit);
                let pairs = _mm256_maddubs_epi16(ones_i8, signed_a);
                let sum = hsum(_mm256_madd_epi16(pairs, ones16));
                acc += d * s_act * sum as f32;
            }
        }
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::GgmlType;
    use crate::quant::idot;
    use crate::quant::testdata::{activations, row};

    /// The property this file is built around: same products, any order, same
    /// integer total — so the vector and scalar paths agree exactly.
    fn assert_matches_scalar(ty: GgmlType, blocks: usize) {
        if !available() {
            eprintln!("no AVX2 on this machine; skipping");
            return;
        }
        let (block, _) = ty.layout().expect("sized type");
        let elements = blocks * block as usize;
        let weights = row(ty, blocks);

        let mut quantized = Quantized::with_capacity(elements);
        quantized.fill(&activations(elements));

        let scalar = match ty {
            GgmlType::Q8_0 => idot::q8_0(&weights, &quantized),
            GgmlType::Q4_0 => idot::q4_0(&weights, &quantized),
            GgmlType::Q4_1 => idot::q4_1(&weights, &quantized),
            GgmlType::Q5_0 => idot::q5_0(&weights, &quantized),
            GgmlType::Q5_1 => idot::q5_1(&weights, &quantized),
            GgmlType::Q2_K => idot::q2_k(&weights, &quantized),
            GgmlType::Q3_K => idot::q3_k(&weights, &quantized),
            GgmlType::Q4_K => idot::q4_k(&weights, &quantized),
            GgmlType::Q5_K => idot::q5_k(&weights, &quantized),
            GgmlType::Q6_K => idot::q6_k(&weights, &quantized),
            GgmlType::IQ4_NL => crate::quant::iq::dot_iq4_nl(&weights, &quantized),
            GgmlType::IQ4_XS => crate::quant::iq::dot_iq4_xs(&weights, &quantized),
            GgmlType::TQ2_0 => crate::quant::bitnet::dot_tq2_0(&weights, &quantized),
            other => panic!("no scalar kernel for {}", other.name()),
        };
        let vector = unsafe {
            match ty {
                GgmlType::Q8_0 => q8_0(&weights, &quantized),
                GgmlType::Q4_0 => q4_0(&weights, &quantized),
                GgmlType::Q4_1 => q4_1(&weights, &quantized),
                GgmlType::Q5_0 => q5_0(&weights, &quantized),
                GgmlType::Q5_1 => q5_1(&weights, &quantized),
                GgmlType::Q2_K => q2_k(&weights, &quantized),
                GgmlType::Q3_K => q3_k(&weights, &quantized),
                GgmlType::Q4_K => q4_k(&weights, &quantized),
                GgmlType::Q5_K => q5_k(&weights, &quantized),
                GgmlType::Q6_K => q6_k(&weights, &quantized),
                GgmlType::IQ4_NL => iq4_nl(&weights, &quantized),
                GgmlType::IQ4_XS => iq4_xs(&weights, &quantized),
                GgmlType::TQ2_0 => tq2_0(&weights, &quantized),
                other => panic!("no vector kernel for {}", other.name()),
            }
        };

        assert_eq!(
            vector.to_bits(),
            scalar.to_bits(),
            "{}: avx2 {vector} != scalar {scalar}",
            ty.name()
        );
    }

    #[test]
    fn q8_0_matches_the_scalar_kernel_exactly() {
        assert_matches_scalar(GgmlType::Q8_0, 1);
        assert_matches_scalar(GgmlType::Q8_0, 9);
    }

    #[test]
    fn q4_0_matches_the_scalar_kernel_exactly() {
        assert_matches_scalar(GgmlType::Q4_0, 1);
        assert_matches_scalar(GgmlType::Q4_0, 9);
    }

    #[test]
    fn q4_1_q5_0_q5_1_match_scalar_exactly() {
        for ty in [GgmlType::Q4_1, GgmlType::Q5_0, GgmlType::Q5_1] {
            assert_matches_scalar(ty, 1);
            assert_matches_scalar(ty, 9);
        }
    }

    #[test]
    fn q2_k_and_q3_k_match_scalar_exactly() {
        for ty in [GgmlType::Q2_K, GgmlType::Q3_K] {
            assert_matches_scalar(ty, 1);
            assert_matches_scalar(ty, 5);
        }
    }

    #[test]
    fn iq4_nl_iq4_xs_and_tq2_0_match_scalar_exactly() {
        assert_matches_scalar(GgmlType::IQ4_NL, 1);
        assert_matches_scalar(GgmlType::IQ4_NL, 9);
        assert_matches_scalar(GgmlType::IQ4_XS, 1);
        assert_matches_scalar(GgmlType::IQ4_XS, 5);
        assert_matches_scalar(GgmlType::TQ2_0, 1);
        assert_matches_scalar(GgmlType::TQ2_0, 5);
    }

    #[test]
    fn q4_k_matches_the_scalar_kernel_exactly() {
        assert_matches_scalar(GgmlType::Q4_K, 1);
        assert_matches_scalar(GgmlType::Q4_K, 5);
    }

    #[test]
    fn q5_k_matches_the_scalar_kernel_exactly() {
        assert_matches_scalar(GgmlType::Q5_K, 1);
        assert_matches_scalar(GgmlType::Q5_K, 5);
    }

    #[test]
    fn q6_k_matches_the_scalar_kernel_exactly() {
        assert_matches_scalar(GgmlType::Q6_K, 1);
        assert_matches_scalar(GgmlType::Q6_K, 5);
    }

    #[test]
    fn detection_is_stable() {
        assert_eq!(available(), available());
        assert_eq!(available_fma_f16c(), available_fma_f16c());
    }
}

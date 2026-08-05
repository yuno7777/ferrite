//! K-quants: 256 elements per super-block, split into eight 32-element blocks
//! with their own 6-bit scale and minimum.
//!
//! The two-level scheme is why `Q4_K` holds up so much better than `Q4_0` at
//! nearly the same size: outliers cost one sub-block's range instead of the
//! whole block's.

use super::half;

/// Elements per super-block.
pub const QK_K: usize = 256;

/// Unpack the 6-bit scale and minimum for sub-block `index` out of the 12
/// packed bytes.
///
/// The first four pairs live in the low 6 bits of `scales[0..8]`. The last four
/// are split: their low 4 bits sit in `scales[8..12]`, and their top 2 bits are
/// stolen from the unused high bits of the first eight bytes.
pub(crate) fn scale_min(index: usize, scales: &[u8]) -> (u8, u8) {
    if index < 4 {
        (scales[index] & 63, scales[index + 4] & 63)
    } else {
        (
            (scales[index + 4] & 0x0F) | ((scales[index - 4] >> 6) << 4),
            (scales[index + 4] >> 4) | ((scales[index] >> 6) << 4),
        )
    }
}

/// 256 elements in 144 bytes: f16 scale, f16 minimum scale, 12 packed
/// scale/min pairs, then 128 bytes of 4-bit quants.
pub fn q4_k(block: &[u8], out: &mut [f32]) {
    let d = half::read_f16(block);
    let dmin = half::read_f16(&block[2..]);
    let scales = &block[4..16];
    let qs = &block[16..144];

    for group in 0..4 {
        let quants = &qs[group * 32..(group + 1) * 32];
        let (sc_low, min_low) = scale_min(group * 2, scales);
        let (sc_high, min_high) = scale_min(group * 2 + 1, scales);
        let (d_low, offset_low) = (d * sc_low as f32, dmin * min_low as f32);
        let (d_high, offset_high) = (d * sc_high as f32, dmin * min_high as f32);

        let base = group * 64;
        for (l, byte) in quants.iter().enumerate() {
            out[base + l] = d_low * (byte & 0x0F) as f32 - offset_low;
            out[base + 32 + l] = d_high * (byte >> 4) as f32 - offset_high;
        }
    }
}

/// 256 elements in 176 bytes: like `Q4_K` but with a fifth bit per weight kept
/// in a separate 32-byte plane, one bit per element.
pub fn q5_k(block: &[u8], out: &mut [f32]) {
    let d = half::read_f16(block);
    let dmin = half::read_f16(&block[2..]);
    let scales = &block[4..16];
    let qh = &block[16..48];
    let qs = &block[48..176];

    for group in 0..4 {
        let quants = &qs[group * 32..(group + 1) * 32];
        let (sc_low, min_low) = scale_min(group * 2, scales);
        let (sc_high, min_high) = scale_min(group * 2 + 1, scales);
        let (d_low, offset_low) = (d * sc_low as f32, dmin * min_low as f32);
        let (d_high, offset_high) = (d * sc_high as f32, dmin * min_high as f32);

        // Each group consumes two bits of the high plane per element, so the
        // masks walk left two positions per group.
        let mask_low = 1u8 << (group * 2);
        let mask_high = 2u8 << (group * 2);

        let base = group * 64;
        for (l, byte) in quants.iter().enumerate() {
            let fifth_low = if qh[l] & mask_low != 0 { 16 } else { 0 };
            let fifth_high = if qh[l] & mask_high != 0 { 16 } else { 0 };
            out[base + l] = d_low * ((byte & 0x0F) + fifth_low) as f32 - offset_low;
            out[base + 32 + l] = d_high * ((byte >> 4) + fifth_high) as f32 - offset_high;
        }
    }
}

/// 256 elements in 210 bytes: 4 low bits per weight, 2 high bits in a second
/// plane, and a full signed byte of scale per 16 elements.
///
/// The layout puts the quants first and the super-block scale last, unlike the
/// other K-quants.
pub fn q6_k(block: &[u8], out: &mut [f32]) {
    let ql = &block[0..128];
    let qh = &block[128..192];
    let scales = &block[192..208];
    let d = half::read_f16(&block[208..]);

    for half_block in 0..2 {
        let ql = &ql[half_block * 64..(half_block + 1) * 64];
        let qh = &qh[half_block * 32..(half_block + 1) * 32];
        let scales = &scales[half_block * 8..(half_block + 1) * 8];
        let base = half_block * 128;

        for l in 0..32 {
            let sub = l / 16;
            // Six bits assembled from two planes, then centred on zero.
            let q = |low: u8, shift: u32| (low | (((qh[l] >> shift) & 3) << 4)) as i8 - 32;
            let quads = [
                (0, q(ql[l] & 0x0F, 0), sub),
                (32, q(ql[l + 32] & 0x0F, 2), sub + 2),
                (64, q(ql[l] >> 4, 4), sub + 4),
                (96, q(ql[l + 32] >> 4, 6), sub + 6),
            ];
            for (offset, quant, scale_index) in quads {
                out[base + offset + l] = d * scales[scale_index] as i8 as f32 * quant as f32;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// d = 1.0, dmin = 0.0, so weights come out as the raw quant values.
    fn q4_k_block(scales: [u8; 12], quants: [u8; 128]) -> Vec<u8> {
        let mut block = Vec::with_capacity(144);
        block.extend_from_slice(&0x3C00u16.to_le_bytes()); // 1.0
        block.extend_from_slice(&0x0000u16.to_le_bytes()); // 0.0
        block.extend_from_slice(&scales);
        block.extend_from_slice(&quants);
        block
    }

    #[test]
    fn q4_k_expands_nibbles_per_sub_block() {
        // Scales 1 for the first four sub-blocks, 0 for the rest: with only the
        // low 6 bits set, the last four unpack to zero.
        let mut scales = [0u8; 12];
        scales[..4].copy_from_slice(&[1, 1, 1, 1]);
        // Low nibble 5, high nibble 2, everywhere.
        let block = q4_k_block(scales, [0x25; 128]);

        let mut out = [0.0; QK_K];
        q4_k(&block, &mut out);

        // Sub-blocks 0..4 cover the first 128 elements, alternating 32 at a
        // time between the low and high nibbles.
        assert!(out[..32].iter().all(|v| *v == 5.0));
        assert!(out[32..64].iter().all(|v| *v == 2.0));
        assert!(out[64..96].iter().all(|v| *v == 5.0));
        assert!(out[96..128].iter().all(|v| *v == 2.0));
        // Scales 4..8 are zero, so the second half is zero.
        assert!(out[128..].iter().all(|v| *v == 0.0));
    }

    #[test]
    fn q4_k_subtracts_the_minimum() {
        let mut scales = [0u8; 12];
        scales[0] = 1; // scale for sub-block 0
        scales[4] = 3; // minimum for sub-block 0
        let mut block = q4_k_block(scales, [0x07; 128]);
        block[2..4].copy_from_slice(&0x3C00u16.to_le_bytes()); // dmin = 1.0

        let mut out = [0.0; QK_K];
        q4_k(&block, &mut out);
        assert!(out[..32].iter().all(|v| *v == 7.0 - 3.0));
    }

    #[test]
    fn q5_k_adds_the_fifth_bit_from_the_high_plane() {
        let mut block = Vec::with_capacity(176);
        block.extend_from_slice(&0x3C00u16.to_le_bytes()); // d = 1.0
        block.extend_from_slice(&0x0000u16.to_le_bytes()); // dmin = 0.0
        let mut scales = [0u8; 12];
        scales[0] = 1;
        block.extend_from_slice(&scales);
        // Bit 0 set for the first element only: group 0's low half reads bit 0.
        let mut qh = [0u8; 32];
        qh[0] = 0b0000_0001;
        block.extend_from_slice(&qh);
        block.extend_from_slice(&[0x03; 128]); // low nibble 3

        let mut out = [0.0; QK_K];
        q5_k(&block, &mut out);
        assert_eq!(out[0], 19.0, "3 + 16 from the high plane");
        assert_eq!(out[1], 3.0, "no high bit set");
    }

    #[test]
    fn q6_k_centres_quants_on_zero() {
        let mut block = vec![0u8; 210];
        block[192..208].fill(1); // every scale = 1
        block[208..210].copy_from_slice(&0x3C00u16.to_le_bytes()); // d = 1.0

        let mut out = [0.0; QK_K];
        q6_k(&block, &mut out);
        // All quant bits zero means 0 - 32 across the board.
        assert!(out.iter().all(|v| *v == -32.0), "got {:?}", &out[..4]);

        // Low nibble of the first byte to 0xF lifts element 0 to 15 - 32.
        block[0] = 0x0F;
        q6_k(&block, &mut out);
        assert_eq!(out[0], -17.0);

        // Two high bits for element 0 add 48 -> 63 - 32.
        block[128] = 0b0000_0011;
        q6_k(&block, &mut out);
        assert_eq!(out[0], 31.0);
    }

    #[test]
    fn high_sub_blocks_borrow_their_top_bits() {
        // Sub-block 4's scale is the low nibble of scales[8] plus the top two
        // bits of scales[0]; its minimum is the high nibble of scales[8] plus
        // the top two bits of scales[4].
        let mut scales = [0u8; 12];
        scales[0] = 0b1100_0000; // contributes 0b11 -> 48
        scales[8] = 0b0000_0010; // contributes 2
        assert_eq!(scale_min(4, &scales).0, 50);

        scales[4] = 0b0100_0000; // contributes 0b01 -> 16
        scales[8] = 0b0011_0010; // high nibble 3
        assert_eq!(scale_min(4, &scales).1, 19);
    }
}

//! The original block formats: 32 elements sharing one scale.
//!
//! Superseded by the K-quants for quality per bit, but still what you get from
//! `Q8_0` and `Q4_0` files, and `Q8_0` in particular stays useful as the
//! highest-fidelity quantization anyone actually ships.

use super::half;

/// 32 elements: f16 scale, then 16 bytes of packed 4-bit quants centred on 8.
///
/// The packing is not sequential — byte `j` holds element `j` in its low nibble
/// and element `j + 16` in its high nibble.
pub fn q4_0(block: &[u8], out: &mut [f32]) {
    let d = half::read_f16(block);
    for (j, byte) in block[2..18].iter().enumerate() {
        out[j] = d * ((byte & 0x0F) as f32 - 8.0);
        out[j + 16] = d * ((byte >> 4) as f32 - 8.0);
    }
}

/// 32 elements: f16 scale, f16 minimum, then 16 bytes of 4-bit quants. Unlike
/// `Q4_0` these are unsigned, with the offset carried by the minimum.
pub fn q4_1(block: &[u8], out: &mut [f32]) {
    let d = half::read_f16(block);
    let min = half::read_f16(&block[2..]);
    for (j, byte) in block[4..20].iter().enumerate() {
        out[j] = d * (byte & 0x0F) as f32 + min;
        out[j + 16] = d * (byte >> 4) as f32 + min;
    }
}

/// 32 elements: f16 scale, then 32 signed bytes. One multiply per weight.
pub fn q8_0(block: &[u8], out: &mut [f32]) {
    let d = half::read_f16(block);
    for (slot, byte) in out.iter_mut().zip(&block[2..34]) {
        *slot = d * (*byte as i8) as f32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// f16 1.0, so the scale drops out of the arithmetic.
    const ONE: [u8; 2] = [0x00, 0x3C];

    #[test]
    fn q8_0_is_scale_times_signed_byte() {
        let mut block = ONE.to_vec();
        block.extend((0..32).map(|i| match i {
            0 => 0u8,
            1 => 1,
            2 => 0xFF, // -1
            3 => 0x7F, // 127
            4 => 0x80, // -128
            _ => 0,
        }));
        let mut out = [0.0; 32];
        q8_0(&block, &mut out);
        assert_eq!(&out[..5], &[0.0, 1.0, -1.0, 127.0, -128.0]);

        // Scale of 2 doubles everything.
        let mut block = vec![0x00, 0x40];
        block.extend(std::iter::repeat_n(3u8, 32));
        q8_0(&block, &mut out);
        assert!(out.iter().all(|v| *v == 6.0));
    }

    #[test]
    fn q4_0_splits_nibbles_across_the_block() {
        let mut block = ONE.to_vec();
        // Low nibble 0xA (10 -> +2), high nibble 0x3 (3 -> -5).
        block.extend(std::iter::repeat_n(0x3Au8, 16));
        let mut out = [0.0; 32];
        q4_0(&block, &mut out);
        assert!(out[..16].iter().all(|v| *v == 2.0), "low nibbles first");
        assert!(out[16..].iter().all(|v| *v == -5.0), "then high nibbles");
    }

    #[test]
    fn q4_1_offsets_by_the_stored_minimum() {
        let mut block = ONE.to_vec();
        block.extend_from_slice(&0xC000u16.to_le_bytes()); // min = -2.0
        block.extend(std::iter::repeat_n(0x21u8, 16));
        let mut out = [0.0; 32];
        q4_1(&block, &mut out);
        // low nibble 1 -> 1 - 2 = -1, high nibble 2 -> 2 - 2 = 0
        assert!(out[..16].iter().all(|v| *v == -1.0));
        assert!(out[16..].iter().all(|v| *v == 0.0));
    }
}

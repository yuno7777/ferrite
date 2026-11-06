//! Half-precision conversion.
//!
//! Every quantized block stores its scale as an IEEE binary16, so this runs
//! once per 32 or 256 weights — often enough to matter, simple enough not to
//! need a lookup table.

/// IEEE 754 binary16 to f32. Exact: every f16 is representable in f32.
#[inline(always)]
pub fn f16(bits: u16) -> f32 {
    let sign = (bits as u32 & 0x8000) << 16;
    let exponent = (bits >> 10) & 0x1F;
    let mantissa = (bits & 0x03FF) as u32;

    match exponent {
        // Zero or subnormal. A subnormal f16 is mantissa * 2^-24, which f32
        // represents normally, so no bit surgery is needed.
        0 => {
            let magnitude = mantissa as f32 / 16_777_216.0;
            f32::from_bits(sign) + if sign == 0 { magnitude } else { -magnitude }
        }
        // Infinity or NaN: exponent all ones, mantissa carried over.
        0x1F => f32::from_bits(sign | 0x7F80_0000 | (mantissa << 13)),
        // Normal: rebias the exponent from 15 to 127.
        _ => f32::from_bits(sign | ((exponent as u32 + 112) << 23) | (mantissa << 13)),
    }
}

/// Alias for [`f16`] used across specialized microkernel tables.
#[inline(always)]
pub fn f16_to_f32(bits: u16) -> f32 {
    f16(bits)
}

/// bfloat16 to f32 — the top 16 bits of an f32, so this is a shift.
#[inline(always)]
pub fn bf16(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Read a little-endian f16 from the front of `bytes`.
///
/// Returns 0.0 on a short slice rather than panicking: callers have already
/// validated block extents, and a scale of zero yields a zero weight, which is
/// the least destructive way to fail.
#[inline(always)]
pub fn read_f16(bytes: &[u8]) -> f32 {
    match bytes.first_chunk::<2>() {
        Some(pair) => f16(u16::from_le_bytes(*pair)),
        None => 0.0,
    }
}

/// Convert f32 to IEEE 754 binary16 bits.
pub fn f32_to_f16(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp32 = ((bits >> 23) & 0xFF) as i32;
    let mant32 = bits & 0x007F_FFFF;

    if exp32 == 255 {
        if mant32 != 0 {
            sign | 0x7E00
        } else {
            sign | 0x7C00
        }
    } else if exp32 > 142 {
        sign | 0x7C00
    } else if exp32 < 113 {
        if exp32 < 103 {
            sign
        } else {
            let shift = 113 - exp32;
            let full_mant = mant32 | 0x0080_0000;
            let mant16 = (full_mant >> (shift + 13)) as u16;
            sign | mant16
        }
    } else {
        let exp16 = ((exp32 - 112) as u16) << 10;
        let mant16 = (mant32 >> 13) as u16;
        sign | exp16 | mant16
    }
}

/// Write a little-endian f16 to a 2-byte slice.
pub fn write_f16(out: &mut [u8], val: f32) {
    let bits = f32_to_f16(val);
    out[0..2].copy_from_slice(&bits.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_bit_patterns() {
        assert_eq!(f16(0x0000), 0.0);
        assert_eq!(f16(0x3C00), 1.0);
        assert_eq!(f16(0xBC00), -1.0);
        assert_eq!(f16(0x4000), 2.0);
        assert_eq!(f16(0xC000), -2.0);
        assert_eq!(f16(0x3800), 0.5);
        assert_eq!(f16(0x3555), 0.333_251_95); // nearest f16 to 1/3
        assert_eq!(f16(0x7BFF), 65504.0); // largest finite f16
    }

    #[test]
    fn signed_zero_and_subnormals() {
        assert_eq!(f16(0x8000), 0.0);
        assert!(f16(0x8000).is_sign_negative(), "-0.0 must keep its sign");
        // Smallest positive subnormal is 2^-24.
        assert_eq!(f16(0x0001), 5.960_464_5e-8);
        assert_eq!(f16(0x8001), -5.960_464_5e-8);
        // Largest subnormal, just below the smallest normal.
        assert_eq!(f16(0x03FF), 1023.0 / 16_777_216.0);
    }

    #[test]
    fn infinities_and_nan() {
        assert_eq!(f16(0x7C00), f32::INFINITY);
        assert_eq!(f16(0xFC00), f32::NEG_INFINITY);
        assert!(f16(0x7E00).is_nan());
    }

    #[test]
    fn bfloat_is_the_top_half_of_an_f32() {
        assert_eq!(bf16(0x3F80), 1.0);
        assert_eq!(bf16(0xC000), -2.0);
        assert_eq!(bf16(0x0000), 0.0);
    }

    #[test]
    fn reading_a_short_slice_yields_zero() {
        assert_eq!(read_f16(&[0x00, 0x3C]), 1.0);
        assert_eq!(read_f16(&[0x00]), 0.0);
        assert_eq!(read_f16(&[]), 0.0);
    }
}

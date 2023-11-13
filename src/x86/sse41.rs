use super::*;
use crate::into;

intrinsic! {
    fn _mm_blend_epi16<const IMM8: i32>(a: __m128i, b: __m128i) -> __m128i { blend!(IMM8, i16x8, a, b) }

    #[testvals(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)]
    fn _mm_blend_ps<const IMM8: i32>(a: __m128,  b: __m128 ) -> __m128 { blend!(IMM8, f32x4, a, b) }

    #[testvals(0, 1, 2, 3)]
    fn _mm_blend_pd<const IMM8: i32>(a: __m128d, b: __m128d) -> __m128d { blend!(IMM8, f64x2, a, b) }

    fn _mm_dp_pd<const IMM8: i32>(a: __m128d, b: __m128d) -> __m128d {
        let a: f64x2 = a.into();
        let b: f64x2 = b.into();
        let zero = f64x2::splat(0.);
        let prod = a * b;
        let sum = mask64x2::from_array([IMM8 & 0x10 != 0, IMM8 & 0x20 != 0])
            .select(prod, zero);

        let sum = simd_swizzle!(sum.deinterleave(zero).0 + sum.deinterleave(zero).1, [0, 0]);

        mask64x2::from_array([IMM8 & 0x1 != 0, IMM8 & 0x2 != 0])
            .select(sum, zero).into()
    }

    fn _mm_dp_ps<const IMM8: i32>(a: __m128, b: __m128) -> __m128 {
        let a: f32x4 = a.into();
        let b: f32x4 = b.into();
        let zero = f32x4::splat(0.);
        let prod = a * b;
        let sum = mask32x4::from_array([IMM8 & 0x10 != 0, IMM8 & 0x20 != 0, IMM8 & 0x40 != 0, IMM8 & 0x80 != 0])
            .select(prod, zero);

        let sum = sum.deinterleave(zero).0 + sum.deinterleave(zero).1;
        let sum = sum.deinterleave(zero).0 + sum.deinterleave(zero).1;
        let sum = simd_swizzle!(sum, [0, 0, 0, 0]);

        mask32x4::from_array([
            IMM8 & 0x1 != 0,
            IMM8 & 0x2 != 0,
            IMM8 & 0x4 != 0,
            IMM8 & 0x8 != 0,
        ]).select(sum, zero).into()
    }
}

intrinsic! {
    fn _mm_blendv_epi8(a: __m128i, b: __m128i, mask: __m128i) -> __m128i { blendv! (i8x16, a, b, mask) }
    fn _mm_blendv_ps  (a: __m128,  b: __m128,  mask: __m128 ) -> __m128  { fblendv!(f32x4, a, b, mask) }
    fn _mm_blendv_pd  (a: __m128d, b: __m128d, mask: __m128d) -> __m128d { fblendv!(f64x2, a, b, mask) }

    fn _mm_cmpeq_epi64(a: __m128i, b: __m128i) -> __m128i { into!(cmpeq!, i64x2, a, b) }

    fn _mm_max_epi32(a: __m128i, b: __m128i) -> __m128i { into!(SimdOrd::simd_max, i32x4, a, b) }
    fn _mm_max_epi8 (a: __m128i, b: __m128i) -> __m128i { into!(SimdOrd::simd_max, i8x16, a, b) }
    fn _mm_max_epu16(a: __m128i, b: __m128i) -> __m128i { into!(SimdOrd::simd_max, u16x8, a, b) }
    fn _mm_max_epu32(a: __m128i, b: __m128i) -> __m128i { into!(SimdOrd::simd_max, u32x4, a, b) }
    fn _mm_min_epi32(a: __m128i, b: __m128i) -> __m128i { into!(SimdOrd::simd_min, i32x4, a, b) }
    fn _mm_min_epi8 (a: __m128i, b: __m128i) -> __m128i { into!(SimdOrd::simd_min, i8x16, a, b) }
    fn _mm_min_epu16(a: __m128i, b: __m128i) -> __m128i { into!(SimdOrd::simd_min, u16x8, a, b) }
    fn _mm_min_epu32(a: __m128i, b: __m128i) -> __m128i { into!(SimdOrd::simd_min, u32x4, a, b) }

    fn _mm_minpos_epu16(a: __m128i) -> __m128i {
        use core::simd::ToBitMask;
        let a: u16x8 = a.into();
        let min = a.reduce_min();
        let min_lanes = u16x8::splat(min).simd_eq(a);
        let index = min_lanes.to_bitmask().trailing_zeros();
        u16x8::from_array([min, index as u16, 0, 0, 0, 0, 0, 0]).into()
    }
}

intrinsic! {
    #[testvals(0, 1, 2, 3)]
    fn _mm_extract_epi32<const IMM8: i32>(a: __m128i) -> i32 {
        let a: i32x4 = a.into();
        a[(IMM8 & 0b11) as usize]
    }

    #[testvals(0, 1)]
    fn _mm_extract_epi64<const IMM8: i32>(a: __m128i) -> i64 {
        let a: i64x2 = a.into();
        a[(IMM8 & 0b1) as usize]
    }

    #[testvals(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)]
    fn _mm_extract_epi8<const IMM8: i32>(a: __m128i) -> i32 {
        let a: u8x16 = a.into();
        a[(IMM8 & 0b1111) as usize] as i32
    }

    #[testvals(0, 1, 2, 3)]
    fn _mm_extract_ps<const IMM8: i32>(a: __m128) -> i32 {
        let a: i32x4 = a.into();
        a[(IMM8 & 0b11) as usize]
    }

    #[testvals(0, 1, 2, 3)]
    fn _mm_insert_epi32<const IMM8: i32>(a: __m128i, i: i32) -> __m128i {
        let mut a: i32x4 = a.into();
        a[(IMM8 & 0b11) as usize] = i;
        a.into()
    }

    #[testvals(0, 1)]
    fn _mm_insert_epi64<const IMM8: i32>(a: __m128i, i: i64) -> __m128i {
        let mut a: i64x2 = a.into();
        a[(IMM8 & 0b1) as usize] = i;
        a.into()
    }

    #[testvals(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)]
    fn _mm_insert_epi8<const IMM8: i32>(a: __m128i, i: i32) -> __m128i {
        let mut a: i8x16 = a.into();
        a[(IMM8 & 0b1111) as usize] = (i & 0xff) as i8;
        a.into()
    }

    #[testvals(0, 1, 2, 3)]
    fn _mm_insert_ps<const IMM8: i32>(a: __m128, b: __m128) -> __m128 {
        let a: f32x4 = a.into();
        let b: f32x4 = b.into();
        let mut tmp2 = a;
        tmp2[((IMM8 >> 4) & 0b11) as usize] = b[((IMM8 >> 6) & 0b11) as usize];
        mask32x4::from_array([IMM8 & 0x1 != 0, IMM8 & 0x2 != 0, IMM8 & 0x4 != 0, IMM8 & 0x8 != 0])
            .select(f32x4::splat(0.), tmp2).into()
    }
}

// TODO ceil, floor
// TODO cvt

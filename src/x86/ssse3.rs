use super::*;
use crate::into;

intrinsic! {
    fn _mm_abs_epi8 (a: __m128i) -> __m128i { into!(SimdInt::abs, i8x16, a) }
    fn _mm_abs_epi16(a: __m128i) -> __m128i { into!(SimdInt::abs, i16x8, a) }
    fn _mm_abs_epi32(a: __m128i) -> __m128i { into!(SimdInt::abs, i32x4, a) }
    #[notest()] fn _mm_abs_pi8  (a: __m64) -> __m64 { into!(SimdInt::abs, i8x8,  a) }
    #[notest()] fn _mm_abs_pi16 (a: __m64) -> __m64 { into!(SimdInt::abs, i16x4, a) }
    #[notest()] fn _mm_abs_pi32 (a: __m64) -> __m64 { into!(SimdInt::abs, i32x2, a) }

    #[notest()] fn _mm_mulhrs_pi16 (a: __m64, b: __m64) -> __m64 { into!(mulhrs, i16x4, a, b) }
    fn _mm_mulhrs_epi16(a: __m128i, b: __m128i) -> __m128i { into!(mulhrs, i16x8, a, b) }

    fn _mm_hadd_epi16 (a: __m128i, b: __m128i) -> __m128i { into!(hadd!,  i16x8, a, b) }
    fn _mm_hadd_epi32 (a: __m128i, b: __m128i) -> __m128i { into!(hadd!,  i32x4, a, b) }
    fn _mm_hadds_epi16(a: __m128i, b: __m128i) -> __m128i { into!(hadds!, i16x8, a, b) }
    fn _mm_hsub_epi16 (a: __m128i, b: __m128i) -> __m128i { into!(hsub!,  i16x8, a, b) }
    fn _mm_hsub_epi32 (a: __m128i, b: __m128i) -> __m128i { into!(hsub!,  i32x4, a, b) }
    fn _mm_hsubs_epi16(a: __m128i, b: __m128i) -> __m128i { into!(hsubs!, i16x8, a, b) }
    #[notest()] fn _mm_hadd_pi16  (a: __m64, b: __m64) -> __m64 { into!(hadd!,  i16x4, a, b) }
    #[notest()] fn _mm_hadd_pi32  (a: __m64, b: __m64) -> __m64 { into!(hadd!,  i32x2, a, b) }
    #[notest()] fn _mm_hadds_pi16 (a: __m64, b: __m64) -> __m64 { into!(hadds!, i16x4, a, b) }
    #[notest()] fn _mm_hsub_pi16  (a: __m64, b: __m64) -> __m64 { into!(hsub!,  i16x4, a, b) }
    #[notest()] fn _mm_hsub_pi32  (a: __m64, b: __m64) -> __m64 { into!(hsub!,  i32x2, a, b) }
    #[notest()] fn _mm_hsubs_pi16 (a: __m64, b: __m64) -> __m64 { into!(hsubs!, i16x4, a, b) }

    fn _mm_sign_epi8 (a: __m128i, b: __m128i) -> __m128i { into!(sign!, i8x16, a, b) }
    fn _mm_sign_epi16(a: __m128i, b: __m128i) -> __m128i { into!(sign!, i16x8, a, b) }
    fn _mm_sign_epi32(a: __m128i, b: __m128i) -> __m128i { into!(sign!, i32x4, a, b) }
    #[notest()] fn _mm_sign_pi8 (a: __m64, b: __m64) -> __m64 { into!(sign!, i8x8, a, b) }
    #[notest()] fn _mm_sign_pi16(a: __m64, b: __m64) -> __m64 { into!(sign!, i16x4, a, b) }
    #[notest()] fn _mm_sign_pi32(a: __m64, b: __m64) -> __m64 { into!(sign!, i32x2, a, b) }
}

intrinsic! {
    #[notest()] fn _mm_maddubs_pi16(a: __m64, b: __m64) -> __m64 {
        let a: u8x8 = a.into();
        let b: i8x8 = b.into();
        let a: i16x8 = a.cast();
        let b: i16x8 = b.cast();
        let prod = a * b;
        let (first, second) = prod.deinterleave(prod);
        simd_swizzle!(first.saturating_add(second), [0, 1, 2, 3]).into()
    }

    fn _mm_maddubs_epi16(a: __m128i, b: __m128i) -> __m128i {
        let a: u8x16 = a.into();
        let b: i8x16 = b.into();
        let a: i16x16 = a.cast();
        let b: i16x16 = b.cast();
        let prod = a * b;
        let (first, second) = prod.deinterleave(prod);
        simd_swizzle!(first.saturating_add(second), [0, 1, 2, 3, 4, 5, 6, 7]).into()
    }
}

intrinsic! {
    fn _mm_alignr_pi8<const IMM8: usize>(a: __m64, b: __m64) -> __m64 {
        fn select<const IMM8: usize>() -> [bool; 8] {
            let mut mask = [true; 8];
            let mut i = 0;
            while i < IMM8 {
                mask[i] = false;
                i += 1;
            }
            mask
        }

        fn mask<const IMM8: usize>() -> [bool; 8] {
            if IMM8 <= 8 {
                [true; 8]
            } else if IMM8 < 16 {
                let mut mask = [false; 8];
                let mut i = 0;
                while i < 16 - IMM8 {
                    mask[i] = true;
                    i += 1;
                }
                mask
            } else {
                [false; 8]
            }
        }

        let a: i8x8 = a.into();
        let b: i8x8 = b.into();
        mask8x8::from_array(mask::<IMM8>())
            .select(
                mask8x8::from_array(select::<IMM8>())
                    .select(a.rotate_lanes_left::<IMM8>(), b.rotate_lanes_right::<IMM8>()),
                i8x8::splat(0),
            ).into()
    }

    fn _mm_alignr_epi8<const IMM8: usize>(a: __m128i, b: __m128i) -> __m128i {
        fn select<const IMM8: usize>() -> [bool; 16] {
            let mut mask = [true; 16];
            let mut i = 0;
            while i < IMM8 {
                mask[i] = false;
                i += 1;
            }
            mask
        }

        fn mask<const IMM8: usize>() -> [bool; 16] {
            if IMM8 <= 16 {
                [true; 16]
            } else if IMM8 < 32 {
                let mut mask = [false; 16];
                let mut i = 0;
                while i < 32 - IMM8 {
                    mask[i] = true;
                    i += 1;
                }
                mask
            } else {
                [false; 16]
            }
        }

        let a: i8x16 = a.into();
        let b: i8x16 = b.into();
        mask8x16::from_array(mask::<IMM8>())
            .select(
                mask8x16::from_array(select::<IMM8>())
                    .select(a.rotate_lanes_left::<IMM8>(), b.rotate_lanes_right::<IMM8>()),
                i8x16::splat(0),
            ).into()
    }
}

// TODO _mm_shuffle_epi8, _mm_shuffle_pi8

use super::*;
use crate::into;

intrinsic! {
    fn _mm_addsub_ps(a: __m128,  b: __m128)  -> __m128  { into!(addsub, f32x4, a, b) }
    fn _mm_addsub_pd(a: __m128d, b: __m128d) -> __m128d { into!(addsub, f64x2, a, b) }
    fn _mm_hadd_ps(a: __m128,  b: __m128)  -> __m128  { into!(hadd!, f32x4, a, b) }
    fn _mm_hadd_pd(a: __m128d, b: __m128d) -> __m128d { into!(hadd!, f64x2, a, b) }
    fn _mm_hsub_ps(a: __m128,  b: __m128)  -> __m128  { into!(hsub!, f32x4, a, b) }
    fn _mm_hsub_pd(a: __m128d, b: __m128d) -> __m128d { into!(hsub!, f64x2, a, b) }
}

intrinsic! {
    unsafe fn _mm_lddqu_si128(mem_addr: *const __m128i) -> __m128i {
        mem_addr.read_unaligned()
    }

    unsafe fn _mm_loaddup_pd(mem_addr: *const f64) -> __m128d {
        f64x2::splat(mem_addr.read()).into()
    }
}

intrinsic! {
    fn _mm_movedup_pd(a: __m128d) -> __m128d {
        let a: f64x2 = a.into();
        simd_swizzle!(a, [0, 0]).into()
    }

    fn _mm_movehdup_ps(a: __m128) -> __m128 {
        let a: f32x4 = a.into();
        simd_swizzle!(a, [1, 1, 3, 3]).into()
    }

    fn _mm_moveldup_ps(a: __m128) -> __m128 {
        let a: f32x4 = a.into();
        simd_swizzle!(a, [0, 0, 2, 2]).into()
    }
}

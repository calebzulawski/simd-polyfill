use super::*;

binary! {
    _mm_addsub_ps, addsub, __m128 as f32x4;
    _mm_addsub_pd, addsub, __m128d as f64x2;
}

binary! {
    _mm_hadd_ps, macro hadd, __m128 as f32x4;
    _mm_hadd_pd, macro hadd, __m128d as f64x2;
    _mm_hsub_ps, macro hsub, __m128 as f32x4;
    _mm_hsub_pd, macro hsub, __m128d as f64x2;
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

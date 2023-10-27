use super::*;

binary! {
    _mm_add_pi8, core::ops::Add::add, __m64 as i8x8;
    _mm_add_pi16, core::ops::Add::add, __m64 as i16x4;
    _mm_add_pi32, core::ops::Add::add, __m64 as i32x2;

    _mm_adds_pi8, SimdInt::saturating_add, __m64 as i8x8;
    _mm_adds_pi16, SimdInt::saturating_add, __m64 as i16x4;
    _mm_adds_pu8, SimdUint::saturating_add, __m64 as u8x8;
    _mm_adds_pu16, SimdUint::saturating_add, __m64 as u16x4;

    _mm_and_si64, core::ops::BitAnd::bitand, __m64 as u8x8;
}

binary! {
    _mm_andnot_si64, macro andnot, __m64 as u8x8;

    _mm_cmpeq_pi8, macro cmpeq, __m64 as i8x8;
    _mm_cmpeq_pi16, macro cmpeq, __m64 as i16x4;
    _mm_cmpeq_pi32, macro cmpeq, __m64 as i32x2;

    _mm_cmpgt_pi8, macro cmpgt, __m64 as i8x8;
    _mm_cmpgt_pi16, macro cmpgt, __m64 as i16x4;
    _mm_cmpgt_pi32, macro cmpgt, __m64 as i32x2;


}

intrinsic! {
    fn _mm_cvtm64_si64(a: __m64) -> i64 {
        unsafe { core::mem::transmute(a) }
    }

    fn _mm_cvtsi32_si64(a: i32) -> __m64 {
        unsafe { core::mem::transmute(a as i64) }
    }

    fn _mm_cvtsi64_si64(a: i64) -> __m64 {
        unsafe { core::mem::transmute(a) }
    }

    fn _mm_cvtsi64_si32(a: __m64) -> i32 {
        let a: i64 = unsafe { core::mem::transmute(a) };
        (a & 0xFFFFFFFF) as i32
    }

    fn _m_empty() {}
    fn _mm_empty() {}

    fn _m_from_int(a: i32) -> __m64 {
        _mm_cvtsi32_si64(a)
    }

    fn _m_from_int64(a: i64) -> __m64 {
        _mm_cvtsi64_si64(a)
    }
}

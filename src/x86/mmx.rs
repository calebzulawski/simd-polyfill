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
    _mm_or_si64, core::ops::BitOr::bitor, __m64 as u8x8;
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

    fn _mm_madd_pi16(a: __m64, b: __m64) -> __m64 {
        let a: i16x4 = a.into();
        let a: i32x4 = a.cast();
        let b: i16x4 = b.into();
        let b: i32x4 = b.cast();
        let mul = a * b;
        (simd_swizzle!(mul, [0, 2]) + simd_swizzle!(mul, [1, 3])).into()
    }

    fn _mm_mulhi_pi16(a: __m64, b: __m64) -> __m64 {
        let a: i16x4 = a.into();
        let a: i32x4 = a.cast();
        let b: i16x4 = b.into();
        let b: i32x4 = b.cast();
        let mul: i16x8 = unsafe { core::mem::transmute(a * b) };
        simd_swizzle!(mul, [1, 3, 5, 7]).into()
    }

    fn _mm_mullo_pi16(a: __m64, b: __m64) -> __m64 {
        let a: i16x4 = a.into();
        let a: i32x4 = a.cast();
        let b: i16x4 = b.into();
        let b: i32x4 = b.cast();
        let mul: i16x8 = unsafe { core::mem::transmute(a * b) };
        simd_swizzle!(mul, [0, 2, 4, 6]).into()
    }

    fn _mm_packs_pi16(a: __m64, b: __m64) -> __m64 {
        packs4! { i8, Into::<i16x4>::into(a), b.into() }.into()
    }

    fn _mm_packs_pi32(a: __m64, b: __m64) -> __m64 {
        packs2! { i16, Into::<i32x2>::into(a), b.into() }.into()
    }

    fn _mm_packs_pu16(a: __m64, b: __m64) -> __m64 {
        packs4! { u8, Into::<i16x4>::into(a), b.into() }.into()
    }

    fn _m_packssdw(a: __m64, b: __m64) -> __m64 {
        _mm_packs_pi32(a, b)
    }

    fn _m_packsswb(a: __m64, b: __m64) -> __m64 {
        _mm_packs_pi16(a, b)
    }

    fn _m_packuswb(a: __m64, b: __m64) -> __m64 {
        _mm_packs_pu16(a, b)
    }
}

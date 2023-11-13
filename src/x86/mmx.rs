use super::*;
use crate::into;

intrinsic! {
    #[notest()] fn _mm_add_pi8 (a: __m64, b: __m64) -> __m64 { into!(Add::add, i8x8, a, b) }
    #[notest()] fn _mm_add_pi16(a: __m64, b: __m64) -> __m64 { into!(Add::add, i16x4, a, b) }
    #[notest()] fn _mm_add_pi32(a: __m64, b: __m64) -> __m64 { into!(Add::add, i32x2, a, b) }

    #[notest()] fn _mm_sub_pi8 (a: __m64, b: __m64) -> __m64 { into!(Sub::sub, i8x8, a, b) }
    #[notest()] fn _mm_sub_pi16(a: __m64, b: __m64) -> __m64 { into!(Sub::sub, i16x4, a, b) }
    #[notest()] fn _mm_sub_pi32(a: __m64, b: __m64) -> __m64 { into!(Sub::sub, i32x2, a, b) }

    #[notest()] fn _mm_adds_pi8 (a: __m64, b: __m64) -> __m64 { into!(SimdInt::saturating_add,  i8x8, a, b) }
    #[notest()] fn _mm_adds_pi16(a: __m64, b: __m64) -> __m64 { into!(SimdInt::saturating_add,  i16x4, a, b) }
    #[notest()] fn _mm_adds_pu8 (a: __m64, b: __m64) -> __m64 { into!(SimdUint::saturating_add, u8x8, a, b) }
    #[notest()] fn _mm_adds_pu16(a: __m64, b: __m64) -> __m64 { into!(SimdUint::saturating_add, u16x4, a, b) }

    #[notest()] fn _mm_subs_pi8 (a: __m64, b: __m64) -> __m64 { into!(SimdInt::saturating_sub,  i8x8, a, b) }
    #[notest()] fn _mm_subs_pi16(a: __m64, b: __m64) -> __m64 { into!(SimdInt::saturating_sub,  i16x4, a, b) }
    #[notest()] fn _mm_subs_pu8 (a: __m64, b: __m64) -> __m64 { into!(SimdUint::saturating_sub, u8x8, a, b) }
    #[notest()] fn _mm_subs_pu16(a: __m64, b: __m64) -> __m64 { into!(SimdUint::saturating_sub, u16x4, a, b) }

    #[notest()] fn _mm_and_si64(a: __m64, b: __m64) -> __m64 { into!(core::ops::BitAnd::bitand, u8x8, a, b) }
    #[notest()] fn _mm_or_si64 (a: __m64, b: __m64) -> __m64 { into!(core::ops::BitOr::bitor,   u8x8, a, b) }
    #[notest()] fn _mm_xor_si64(a: __m64, b: __m64) -> __m64 { into!(core::ops::BitXor::bitxor, u8x8, a, b) }

    #[notest()] fn _mm_andnot_si64(a: __m64, b: __m64) -> __m64 { into!(andnot!, u8x8, a, b) }

    #[notest()] fn _mm_cmpeq_pi8 (a: __m64, b: __m64) -> __m64 { into!(cmpeq!, i8x8, a, b) }
    #[notest()] fn _mm_cmpeq_pi16(a: __m64, b: __m64) -> __m64 { into!(cmpeq!, i16x4, a, b) }
    #[notest()] fn _mm_cmpeq_pi32(a: __m64, b: __m64) -> __m64 { into!(cmpeq!, i32x2, a, b) }

    #[notest()] fn _mm_cmpgt_pi8 (a: __m64, b: __m64) -> __m64 { into!(cmpgt!, i8x8, a, b) }
    #[notest()] fn _mm_cmpgt_pi16(a: __m64, b: __m64) -> __m64 { into!(cmpgt!, i16x4, a, b) }
    #[notest()] fn _mm_cmpgt_pi32(a: __m64, b: __m64) -> __m64 { into!(cmpgt!, i32x2, a, b) }

    #[notest()] fn _mm_unpackhi_pi8 (a: __m64, b: __m64) -> __m64 { into!(unpackhi!, i8x8, a, b) }
    #[notest()] fn _mm_unpackhi_pi16(a: __m64, b: __m64) -> __m64 { into!(unpackhi!, i16x4, a, b) }
    #[notest()] fn _mm_unpackhi_pi32(a: __m64, b: __m64) -> __m64 { into!(unpackhi!, i32x2, a, b) }

    #[notest()] fn _mm_unpacklo_pi8 (a: __m64, b: __m64) -> __m64 { into!(unpacklo!, i8x8, a, b) }
    #[notest()] fn _mm_unpacklo_pi16(a: __m64, b: __m64) -> __m64 { into!(unpacklo!, i16x4, a, b) }
    #[notest()] fn _mm_unpacklo_pi32(a: __m64, b: __m64) -> __m64 { into!(unpacklo!, i32x2, a, b) }
}

intrinsic! {
    #[notest()]
    fn _mm_cvtm64_si64(a: __m64) -> i64 {
        unsafe { core::mem::transmute(a) }
    }

    #[notest()]
    fn _mm_cvtsi32_si64(a: i32) -> __m64 {
        unsafe { core::mem::transmute(a as i64) }
    }

    #[notest()]
    fn _mm_cvtsi64_si64(a: i64) -> __m64 {
        unsafe { core::mem::transmute(a) }
    }

    #[notest()]
    fn _mm_cvtsi64_si32(a: __m64) -> i32 {
        let a: i64 = unsafe { core::mem::transmute(a) };
        (a & 0xFFFFFFFF) as i32
    }

    #[notest()]
    fn _m_empty() {}
    #[notest()]
    fn _mm_empty() {}

    #[notest()]
    fn _m_to_int(a: __m64) -> i32 {
        _mm_cvtsi64_si32(a)
    }

    #[notest()]
    fn _m_to_int64(a: __m64) -> i64 {
        _mm_cvtm64_si64(a)
    }

    #[notest()]
    fn _m_from_int(a: i32) -> __m64 {
        _mm_cvtsi32_si64(a)
    }

    #[notest()]
    fn _m_from_int64(a: i64) -> __m64 {
        _mm_cvtsi64_si64(a)
    }

    #[notest()]
    fn _mm_madd_pi16(a: __m64, b: __m64) -> __m64 {
        let a: i16x4 = a.into();
        let a: i32x4 = a.cast();
        let b: i16x4 = b.into();
        let b: i32x4 = b.cast();
        let mul = a * b;
        (simd_swizzle!(mul, [0, 2]) + simd_swizzle!(mul, [1, 3])).into()
    }

    #[notest()]
    fn _mm_mulhi_pi16(a: __m64, b: __m64) -> __m64 {
        let a: i16x4 = a.into();
        let a: i32x4 = a.cast();
        let b: i16x4 = b.into();
        let b: i32x4 = b.cast();
        let mul: i16x8 = unsafe { core::mem::transmute(a * b) };
        simd_swizzle!(mul, [1, 3, 5, 7]).into()
    }

    #[notest()]
    fn _mm_mullo_pi16(a: __m64, b: __m64) -> __m64 {
        let a: i16x4 = a.into();
        let a: i32x4 = a.cast();
        let b: i16x4 = b.into();
        let b: i32x4 = b.cast();
        let mul: i16x8 = unsafe { core::mem::transmute(a * b) };
        simd_swizzle!(mul, [0, 2, 4, 6]).into()
    }

    #[notest()]
    fn _mm_packs_pi16(a: __m64, b: __m64) -> __m64 {
        packs4! { i8, Into::<i16x4>::into(a), b.into() }.into()
    }

    #[notest()]
    fn _mm_packs_pi32(a: __m64, b: __m64) -> __m64 {
        packs2! { i16, Into::<i32x2>::into(a), b.into() }.into()
    }

    #[notest()]
    fn _mm_packs_pu16(a: __m64, b: __m64) -> __m64 {
        packs4! { u8, Into::<i16x4>::into(a), b.into() }.into()
    }
}

intrinsic! {
    #[notest()]
    fn _mm_set_pi8(e7: i8, e6: i8, e5: i8, e4: i8, e3: i8, e2: i8, e1: i8, e0: i8) -> __m64 {
        i8x8::from_array([e0, e1, e2, e3, e4, e5, e6, e7]).into()
    }

    #[notest()]
    fn _mm_set_pi16(e3: i16, e2: i16, e1: i16, e0: i16) -> __m64 {
        i16x4::from_array([e0, e1, e2, e3]).into()
    }

    #[notest()]
    fn _mm_set_pi32(e1: i32, e0: i32) -> __m64 {
        i32x2::from_array([e0, e1]).into()
    }

    #[notest()]
    fn _mm_setr_pi8(e7: i8, e6: i8, e5: i8, e4: i8, e3: i8, e2: i8, e1: i8, e0: i8) -> __m64 {
        i8x8::from_array([e7, e6, e5, e4, e3, e2, e1, e0]).into()
    }

    #[notest()]
    fn _mm_setr_pi16(e3: i16, e2: i16, e1: i16, e0: i16) -> __m64 {
        i16x4::from_array([e3, e2, e1, e0]).into()
    }

    #[notest()]
    fn _mm_setr_pi32(e1: i32, e0: i32) -> __m64 {
        i32x2::from_array([e1, e0]).into()
    }

    #[notest()]
    fn _mm_set1_pi32(a: i32) -> __m64 {
        i32x2::splat(a).into()
    }

    #[notest()]
    fn _mm_set1_pi16(a: i16) -> __m64 {
        i16x4::splat(a).into()
    }

    #[notest()]
    fn _mm_set1_pi8(a: i8) -> __m64 {
        i8x8::splat(a).into()
    }

    #[notest()]
    fn _mm_setzero_si64() -> __m64 {
        u8x8::splat(0).into()
    }
}

intrinsic! {
    #[notest()] fn _mm_slli_epi16<const IMM8: i32>(a: __m64) -> __m64 { shift_logical!(Shl::shl, i16x4, a, IMM8) }
    #[notest()] fn _mm_slli_epi32<const IMM8: i32>(a: __m64) -> __m64 { shift_logical!(Shl::shl, i32x2, a, IMM8) }
    #[notest()] fn _mm_slli_epi64<const IMM8: i32>(a: __m64) -> __m64 { shift_logical!(Shl::shl, i64x1, a, IMM8) }
    #[notest()] fn _mm_srli_epi16<const IMM8: i32>(a: __m64) -> __m64 { shift_logical!(Shr::shr, u16x4, a, IMM8) }
    #[notest()] fn _mm_srli_epi32<const IMM8: i32>(a: __m64) -> __m64 { shift_logical!(Shr::shr, u32x2, a, IMM8) }
    #[notest()] fn _mm_srli_epi64<const IMM8: i32>(a: __m64) -> __m64 { shift_logical!(Shr::shr, u64x1, a, IMM8) }
    #[notest()] fn _mm_srai_epi16<const IMM8: i32>(a: __m64) -> __m64 { shift_right!(i16x4, a, IMM8) }
    #[notest()] fn _mm_srai_epi32<const IMM8: i32>(a: __m64) -> __m64 { shift_right!(i32x2, a, IMM8) }
    #[notest()] fn _m_psllwi<const IMM8: i32>(a: __m64) -> __m64 { shift_logical!(Shl::shl, i16x4, a, IMM8) }
    #[notest()] fn _m_pslldi<const IMM8: i32>(a: __m64) -> __m64 { shift_logical!(Shl::shl, i32x2, a, IMM8) }
    #[notest()] fn _m_psllqi<const IMM8: i32>(a: __m64) -> __m64 { shift_logical!(Shl::shl, i64x1, a, IMM8) }
    #[notest()] fn _m_psrlwi<const IMM8: i32>(a: __m64) -> __m64 { shift_logical!(Shr::shr, u16x4, a, IMM8) }
    #[notest()] fn _m_psrldi<const IMM8: i32>(a: __m64) -> __m64 { shift_logical!(Shr::shr, u32x2, a, IMM8) }
    #[notest()] fn _m_psrlqi<const IMM8: i32>(a: __m64) -> __m64 { shift_logical!(Shr::shr, u64x1, a, IMM8) }
    #[notest()] fn _m_psrawi<const IMM8: i32>(a: __m64) -> __m64 { shift_right!(i16x4, a, IMM8) }
    #[notest()] fn _m_psradi<const IMM8: i32>(a: __m64) -> __m64 { shift_right!(i32x2, a, IMM8) }
}

intrinsic! {
    #[notest()] fn _mm_sll_pi16(a: __m64, b: __m64) -> __m64 { sxl!(Shl::shl, i16x4, u64x1, a, b) }
    #[notest()] fn _mm_sll_pi32(a: __m64, b: __m64) -> __m64 { sxl!(Shl::shl, i32x2, u64x1, a, b) }
    #[notest()] fn _mm_sll_pi64(a: __m64, b: __m64) -> __m64 { sxl!(Shl::shl, i64x1, u64x1, a, b) }
    #[notest()] fn _mm_srl_pi16(a: __m64, b: __m64) -> __m64 { sxl!(Shr::shr, u16x4, u64x1, a, b) }
    #[notest()] fn _mm_srl_pi32(a: __m64, b: __m64) -> __m64 { sxl!(Shr::shr, u32x2, u64x1, a, b) }
    #[notest()] fn _mm_srl_pi64(a: __m64, b: __m64) -> __m64 { sxl!(Shr::shr, u64x1, u64x1, a, b) }
    #[notest()] fn _mm_sra_pi16(a: __m64, b: __m64) -> __m64 { sra!(i16x4, u64x1, a, b) }
    #[notest()] fn _mm_sra_pi32(a: __m64, b: __m64) -> __m64 { sra!(i32x2, u64x1, a, b) }
}

macro_rules! alias {
    { $($alias:ident = $name:ident;)* } => {
        intrinsic! {
        $(
            #[notest()]
            fn $alias(a: __m64, b: __m64) -> __m64 {
                $name(a, b)
            }
        )*
        }
    }
}

alias! {
    _m_packssdw = _mm_packs_pi32;
    _m_packsswb = _mm_packs_pi16;
    _m_packuswb = _mm_packs_pu16;
    _m_paddb = _mm_add_pi8;
    _m_paddd = _mm_add_pi32;
    _m_paddsb = _mm_adds_pi8;
    _m_paddsw = _mm_adds_pi16;
    _m_paddusb = _mm_adds_pu8;
    _m_paddusw = _mm_adds_pu16;
    _m_paddw = _mm_add_pi16;
    _m_pand = _mm_and_si64;
    _m_pandn = _mm_andnot_si64;
    _m_pcmpeqb = _mm_cmpeq_pi8;
    _m_pcmpeqd = _mm_cmpeq_pi32;
    _m_pcmpeqw = _mm_cmpeq_pi16;
    _m_pcmpgtb = _mm_cmpgt_pi8;
    _m_pcmpgtd = _mm_cmpgt_pi32;
    _m_pcmpgtw = _mm_cmpgt_pi16;
    _m_pmaddwd = _mm_madd_pi16;
    _m_pmulhw = _mm_mulhi_pi16;
    _m_pmullw = _mm_mullo_pi16;
    _m_por = _mm_or_si64;
    _m_psubb = _mm_sub_pi8;
    _m_psubd = _mm_sub_pi32;
    _m_psubsb = _mm_subs_pi8;
    _m_psubsw = _mm_subs_pi16;
    _m_psubusb = _mm_subs_pu8;
    _m_psubusw = _mm_subs_pu16;
    _m_psubw = _mm_sub_pi16;
    _m_punpckhbw = _mm_unpackhi_pi8;
    _m_punpckhdq = _mm_unpackhi_pi32;
    _m_punpckhwd = _mm_unpackhi_pi16;
    _m_punpcklbw = _mm_unpacklo_pi8;
    _m_punpckldq = _mm_unpacklo_pi32;
    _m_punpcklwd = _mm_unpacklo_pi16;
    _m_pxor = _mm_xor_si64;
    _m_psllw = _mm_sll_pi16;
    _m_pslld = _mm_sll_pi32;
    _m_psslq = _mm_sll_pi64;
    _m_psrlw = _mm_srl_pi16;
    _m_psrld = _mm_srl_pi32;
    _m_psrlq = _mm_srl_pi64;
    _m_psraw = _mm_sra_pi16;
    _m_psrad = _mm_sra_pi32;
}

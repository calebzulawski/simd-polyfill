use super::*;

binary! {
    _mm_add_pi8, core::ops::Add::add, __m64 as i8x8;
    _mm_add_pi16, core::ops::Add::add, __m64 as i16x4;
    _mm_add_pi32, core::ops::Add::add, __m64 as i32x2;

    _mm_sub_pi8, core::ops::Sub::sub, __m64 as i8x8;
    _mm_sub_pi16, core::ops::Sub::sub, __m64 as i16x4;
    _mm_sub_pi32, core::ops::Sub::sub, __m64 as i32x2;

    _mm_adds_pi8, SimdInt::saturating_add, __m64 as i8x8;
    _mm_adds_pi16, SimdInt::saturating_add, __m64 as i16x4;
    _mm_adds_pu8, SimdUint::saturating_add, __m64 as u8x8;
    _mm_adds_pu16, SimdUint::saturating_add, __m64 as u16x4;

    _mm_subs_pi8, SimdInt::saturating_sub, __m64 as i8x8;
    _mm_subs_pi16, SimdInt::saturating_sub, __m64 as i16x4;
    _mm_subs_pu8, SimdUint::saturating_sub, __m64 as u8x8;
    _mm_subs_pu16, SimdUint::saturating_sub, __m64 as u16x4;

    _mm_and_si64, core::ops::BitAnd::bitand, __m64 as u8x8;
    _mm_or_si64, core::ops::BitOr::bitor, __m64 as u8x8;
    _mm_xor_si64, core::ops::BitXor::bitxor, __m64 as u8x8;
}

binary! {
    _mm_andnot_si64, macro andnot, __m64 as u8x8;

    _mm_cmpeq_pi8, macro cmpeq, __m64 as i8x8;
    _mm_cmpeq_pi16, macro cmpeq, __m64 as i16x4;
    _mm_cmpeq_pi32, macro cmpeq, __m64 as i32x2;

    _mm_cmpgt_pi8, macro cmpgt, __m64 as i8x8;
    _mm_cmpgt_pi16, macro cmpgt, __m64 as i16x4;
    _mm_cmpgt_pi32, macro cmpgt, __m64 as i32x2;

    _mm_unpackhi_pi8, macro unpackhi, __m64 as i8x8;
    _mm_unpackhi_pi16, macro unpackhi, __m64 as i16x4;
    _mm_unpackhi_pi32, macro unpackhi, __m64 as i32x2;

    _mm_unpacklo_pi8, macro unpacklo, __m64 as i8x8;
    _mm_unpacklo_pi16, macro unpacklo, __m64 as i16x4;
    _mm_unpacklo_pi32, macro unpacklo, __m64 as i32x2;
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

    fn _m_to_int(a: __m64) -> i32 {
        _mm_cvtsi64_si32(a)
    }

    fn _m_to_int64(a: __m64) -> i64 {
        _mm_cvtm64_si64(a)
    }

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
}

intrinsic! {
    fn _mm_set_pi8(e7: i8, e6: i8, e5: i8, e4: i8, e3: i8, e2: i8, e1: i8, e0: i8) -> __m64 {
        i8x8::from_array([e0, e1, e2, e3, e4, e5, e6, e7]).into()
    }

    fn _mm_set_pi16(e3: i16, e2: i16, e1: i16, e0: i16) -> __m64 {
        i16x4::from_array([e0, e1, e2, e3]).into()
    }

    fn _mm_set_pi32(e1: i32, e0: i32) -> __m64 {
        i32x2::from_array([e0, e1]).into()
    }

    fn _mm_setr_pi8(e7: i8, e6: i8, e5: i8, e4: i8, e3: i8, e2: i8, e1: i8, e0: i8) -> __m64 {
        i8x8::from_array([e7, e6, e5, e4, e3, e2, e1, e0]).into()
    }

    fn _mm_setr_pi16(e3: i16, e2: i16, e1: i16, e0: i16) -> __m64 {
        i16x4::from_array([e3, e2, e1, e0]).into()
    }

    fn _mm_setr_pi32(e1: i32, e0: i32) -> __m64 {
        i32x2::from_array([e1, e0]).into()
    }

    fn _mm_set1_pi32(a: i32) -> __m64 {
        i32x2::splat(a).into()
    }

    fn _mm_set1_pi16(a: i16) -> __m64 {
        i16x4::splat(a).into()
    }

    fn _mm_set1_pi8(a: i8) -> __m64 {
        i8x8::splat(a).into()
    }

    fn _mm_setzero_si64() -> __m64 {
        u8x8::splat(0).into()
    }
}

macro_rules! alias {
    { $($alias:ident = $name:ident;)* } => {
        intrinsic! {
        $(
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
}

binary! {
    _mm_sll_pi16, core::ops::Shl::shl, __m64 as u16x4;
    _mm_sll_pi32, core::ops::Shl::shl, __m64 as u32x2;
    _mm_sll_si64, core::ops::Shl::shl, __m64 as u64x1;
    _mm_sra_pi16, core::ops::Shr::shr, __m64 as i16x4;
    _mm_sra_pi32, core::ops::Shr::shr, __m64 as i32x2;
    _mm_srl_pi16, core::ops::Shr::shr, __m64 as u16x4;
    _mm_srl_pi32, core::ops::Shr::shr, __m64 as u32x2;
    _mm_srl_si64, core::ops::Shr::shr, __m64 as u64x1;

    _m_psllw,  core::ops::Shl::shl, __m64 as u16x4;
    _m_pslld,  core::ops::Shl::shl, __m64 as u32x2;
    _m_psllq,  core::ops::Shl::shl, __m64 as u64x1;
    _m_psraw,  core::ops::Shr::shr, __m64 as i16x4;
    _m_psrad,  core::ops::Shr::shr, __m64 as i32x2;
    _m_psrlw,  core::ops::Shr::shr, __m64 as u16x4;
    _m_psrld,  core::ops::Shr::shr, __m64 as u32x2;
    _m_psrlq,  core::ops::Shr::shr, __m64 as u64x1;
}

macro_rules! shift {
    { $($imm:ident, $func:path, $ty:ty;)* } => {
        intrinsic! {
            $(
            fn $imm(a: __m64, imm8: i32) -> __m64 {
                let a: $ty = a.into();
                $func(a, <$ty>::splat(imm8 as _)).into()
            }
            )*
        }
    }
}

shift! {
    _mm_slli_pi16, core::ops::Shl::shl, u16x4;
    _mm_slli_pi32, core::ops::Shl::shl, u32x2;
    _mm_slli_si64, core::ops::Shl::shl, u64x1;
    _mm_srai_pi16, core::ops::Shr::shr, i16x4;
    _mm_srai_pi32, core::ops::Shr::shr, i32x2;
    _mm_srli_pi16, core::ops::Shr::shr, u16x4;
    _mm_srli_pi32, core::ops::Shr::shr, u32x2;
    _mm_srli_si64, core::ops::Shr::shr, u64x1;

    _m_psllwi, core::ops::Shl::shl, u16x4;
    _m_pslldi, core::ops::Shl::shl, u32x2;
    _m_psllqi, core::ops::Shl::shl, u64x1;
    _m_psrawi, core::ops::Shr::shr, i16x4;
    _m_psradi, core::ops::Shr::shr, i32x2;
    _m_psrlwi, core::ops::Shr::shr, u16x4;
    _m_psrldi, core::ops::Shr::shr, u32x2;
    _m_psrlqi, core::ops::Shr::shr, u64x1;
}

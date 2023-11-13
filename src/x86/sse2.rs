use super::*;
use crate::{into, into_first};
use core::simd::ToBitMask;

intrinsic! {
    fn _mm_add_epi8 (a: __m128i, b: __m128i) -> __m128i { into!(Add::add, i8x16, a, b) }
    fn _mm_add_epi16(a: __m128i, b: __m128i) -> __m128i { into!(Add::add, i16x8, a, b) }
    fn _mm_add_epi32(a: __m128i, b: __m128i) -> __m128i { into!(Add::add, i32x4, a, b) }
    fn _mm_add_epi64(a: __m128i, b: __m128i) -> __m128i { into!(Add::add, i64x2, a, b) }
    fn _mm_sub_epi8 (a: __m128i, b: __m128i) -> __m128i { into!(Sub::sub, i8x16, a, b) }
    fn _mm_sub_epi16(a: __m128i, b: __m128i) -> __m128i { into!(Sub::sub, i16x8, a, b) }
    fn _mm_sub_epi32(a: __m128i, b: __m128i) -> __m128i { into!(Sub::sub, i32x4, a, b) }
    fn _mm_sub_epi64(a: __m128i, b: __m128i) -> __m128i { into!(Sub::sub, i64x2, a, b) }

    #[notest()] fn _mm_add_si64(a: __m64, b: __m64) -> __m64 { into!(Add::add, i64x1, a, b) }
    #[notest()] fn _mm_sub_si64(a: __m64, b: __m64) -> __m64 { into!(Sub::sub, i64x1, a, b) }

    fn _mm_add_pd(a: __m128d, b: __m128d) -> __m128d { into!(Add::add, f64x2, a, b) }
    fn _mm_sub_pd(a: __m128d, b: __m128d) -> __m128d { into!(Sub::sub, f64x2, a, b) }
    fn _mm_mul_pd(a: __m128d, b: __m128d) -> __m128d { into!(Mul::mul, f64x2, a, b) }
    fn _mm_div_pd(a: __m128d, b: __m128d) -> __m128d { into!(Div::div, f64x2, a, b) }

    fn _mm_adds_epi8 (a: __m128i, b: __m128i) -> __m128i { into!(SimdInt::saturating_add,  i8x16, a, b) }
    fn _mm_adds_epi16(a: __m128i, b: __m128i) -> __m128i { into!(SimdInt::saturating_add,  i16x8, a, b) }
    fn _mm_adds_epu8 (a: __m128i, b: __m128i) -> __m128i { into!(SimdUint::saturating_add, u8x16, a, b) }
    fn _mm_adds_epu16(a: __m128i, b: __m128i) -> __m128i { into!(SimdUint::saturating_add, u16x8, a, b) }
    fn _mm_subs_epi8 (a: __m128i, b: __m128i) -> __m128i { into!(SimdInt::saturating_sub,  i8x16, a, b) }
    fn _mm_subs_epi16(a: __m128i, b: __m128i) -> __m128i { into!(SimdInt::saturating_sub,  i16x8, a, b) }
    fn _mm_subs_epu8 (a: __m128i, b: __m128i) -> __m128i { into!(SimdUint::saturating_sub, u8x16, a, b) }
    fn _mm_subs_epu16(a: __m128i, b: __m128i) -> __m128i { into!(SimdUint::saturating_sub, u16x8, a, b) }

    fn _mm_and_pd   (a: __m128d, b: __m128d) -> __m128d { into!(BitAnd::bitand, i64x2, a, b) }
    fn _mm_or_pd    (a: __m128d, b: __m128d) -> __m128d { into!(BitOr::bitor,   i64x2, a, b) }
    fn _mm_xor_pd   (a: __m128d, b: __m128d) -> __m128d { into!(BitXor::bitxor, i64x2, a, b) }
    fn _mm_and_si128(a: __m128i, b: __m128i) -> __m128i { into!(BitAnd::bitand, i64x2, a, b) }
    fn _mm_or_si128 (a: __m128i, b: __m128i) -> __m128i { into!(BitOr::bitor,   i64x2, a, b) }
    fn _mm_xor_si128(a: __m128i, b: __m128i) -> __m128i { into!(BitXor::bitxor, i64x2, a, b) }

    fn _mm_avg_epu8 (a: __m128i, b: __m128i) -> __m128i { into!(pavgb, u8x16, a, b) }
    fn _mm_avg_epu16(a: __m128i, b: __m128i) -> __m128i { into!(pavgw, u16x8, a, b) }

    fn _mm_max_epi16(a: __m128i, b: __m128i) -> __m128i { into!(SimdOrd::simd_max, i16x8, a, b) }
    fn _mm_max_epu8 (a: __m128i, b: __m128i) -> __m128i { into!(SimdOrd::simd_max, u8x16, a, b) }
    fn _mm_min_epi16(a: __m128i, b: __m128i) -> __m128i { into!(SimdOrd::simd_min, i16x8, a, b) }
    fn _mm_min_epu8 (a: __m128i, b: __m128i) -> __m128i { into!(SimdOrd::simd_min, u8x16, a, b) }

    fn _mm_max_pd(a: __m128d, b: __m128d) -> __m128d { into!(float_max!, f64x2, a, b) }
    fn _mm_min_pd(a: __m128d, b: __m128d) -> __m128d { into!(float_min!, f64x2, a, b) }

    fn _mm_andnot_pd   (a: __m128d, b: __m128d) -> __m128d { into!(andnot!, i64x2, a, b) }
    fn _mm_andnot_si128(a: __m128i, b: __m128i) -> __m128i { into!(andnot!, i64x2, a, b) }

    fn _mm_cmpeq_epi8 (a: __m128i, b: __m128i) -> __m128i { into!(cmpeq!, i8x16, a, b) }
    fn _mm_cmpeq_epi16(a: __m128i, b: __m128i) -> __m128i { into!(cmpeq!, i16x8, a, b) }
    fn _mm_cmpeq_epi32(a: __m128i, b: __m128i) -> __m128i { into!(cmpeq!, i32x4, a, b) }
    fn _mm_cmpgt_epi8 (a: __m128i, b: __m128i) -> __m128i { into!(cmpgt!, i8x16, a, b) }
    fn _mm_cmpgt_epi16(a: __m128i, b: __m128i) -> __m128i { into!(cmpgt!, i16x8, a, b) }
    fn _mm_cmpgt_epi32(a: __m128i, b: __m128i) -> __m128i { into!(cmpgt!, i32x4, a, b) }
    fn _mm_cmplt_epi8 (a: __m128i, b: __m128i) -> __m128i { into!(cmplt!, i8x16, a, b) }
    fn _mm_cmplt_epi16(a: __m128i, b: __m128i) -> __m128i { into!(cmplt!, i16x8, a, b) }
    fn _mm_cmplt_epi32(a: __m128i, b: __m128i) -> __m128i { into!(cmplt!, i32x4, a, b) }

    fn _mm_cmpeq_pd(a: __m128d, b: __m128d) -> __m128d { into!(cmpeq!, f64x2, a, b) }
    fn _mm_cmpge_pd(a: __m128d, b: __m128d) -> __m128d { into!(cmpge!, f64x2, a, b) }
    fn _mm_cmpgt_pd(a: __m128d, b: __m128d) -> __m128d { into!(cmpgt!, f64x2, a, b) }
    fn _mm_cmple_pd(a: __m128d, b: __m128d) -> __m128d { into!(cmple!, f64x2, a, b) }
    fn _mm_cmplt_pd(a: __m128d, b: __m128d) -> __m128d { into!(cmplt!, f64x2, a, b) }
    fn _mm_cmpneq_pd(a: __m128d, b: __m128d) -> __m128d { into!(cmpneq!, f64x2, a, b) }
    fn _mm_cmpnge_pd(a: __m128d, b: __m128d) -> __m128d { into!(cmpnge!, f64x2, a, b) }
    fn _mm_cmpngt_pd(a: __m128d, b: __m128d) -> __m128d { into!(cmpngt!, f64x2, a, b) }
    fn _mm_cmpnle_pd(a: __m128d, b: __m128d) -> __m128d { into!(cmpnle!, f64x2, a, b) }
    fn _mm_cmpnlt_pd(a: __m128d, b: __m128d) -> __m128d { into!(cmpnlt!, f64x2, a, b) }
    fn _mm_cmpord_pd(a: __m128d, b: __m128d) -> __m128d { into!(cmpord!, f64x2, a, b) }
    fn _mm_cmpunord_pd(a: __m128d, b: __m128d) -> __m128d { into!(cmpunord!, f64x2, a, b) }

    fn _mm_unpackhi_epi8 (a: __m128i, b: __m128i) -> __m128i { into!(unpackhi!, i8x16, a, b) }
    fn _mm_unpackhi_epi16(a: __m128i, b: __m128i) -> __m128i { into!(unpackhi!, i16x8, a, b) }
    fn _mm_unpackhi_epi32(a: __m128i, b: __m128i) -> __m128i { into!(unpackhi!, i32x4, a, b) }
    fn _mm_unpackhi_epi64(a: __m128i, b: __m128i) -> __m128i { into!(unpackhi!, i64x2, a, b) }
    fn _mm_unpacklo_epi8 (a: __m128i, b: __m128i) -> __m128i { into!(unpacklo!, i8x16, a, b) }
    fn _mm_unpacklo_epi16(a: __m128i, b: __m128i) -> __m128i { into!(unpacklo!, i16x8, a, b) }
    fn _mm_unpacklo_epi32(a: __m128i, b: __m128i) -> __m128i { into!(unpacklo!, i32x4, a, b) }
    fn _mm_unpacklo_epi64(a: __m128i, b: __m128i) -> __m128i { into!(unpacklo!, i64x2, a, b) }

    fn _mm_unpacklo_pd(a: __m128d, b: __m128d) -> __m128d { into!(unpacklo!, f64x2, a, b) }
    fn _mm_unpackhi_pd(a: __m128d, b: __m128d) -> __m128d { into!(unpackhi!, f64x2, a, b) }

    fn _mm_max_sd(a: __m128d, b: __m128d) -> __m128d { into_first!(float_max!, f64x2, a, b) }
    fn _mm_min_sd(a: __m128d, b: __m128d) -> __m128d { into_first!(float_min!, f64x2, a, b) }

    fn _mm_cmpeq_sd(a: __m128d, b: __m128d) -> __m128d { into_first!(fcmpeq!, f64x2, a, b) }
    fn _mm_cmpge_sd(a: __m128d, b: __m128d) -> __m128d { into_first!(fcmpge!, f64x2, a, b) }
    fn _mm_cmpgt_sd(a: __m128d, b: __m128d) -> __m128d { into_first!(fcmpgt!, f64x2, a, b) }
    fn _mm_cmple_sd(a: __m128d, b: __m128d) -> __m128d { into_first!(fcmple!, f64x2, a, b) }
    fn _mm_cmplt_sd(a: __m128d, b: __m128d) -> __m128d { into_first!(fcmplt!, f64x2, a, b) }
    fn _mm_cmpneq_sd(a: __m128d, b: __m128d) -> __m128d { into_first!(fcmpneq!, f64x2, a, b) }
    fn _mm_cmpnge_sd(a: __m128d, b: __m128d) -> __m128d { into_first!(fcmpnge!, f64x2, a, b) }
    fn _mm_cmpngt_sd(a: __m128d, b: __m128d) -> __m128d { into_first!(fcmpngt!, f64x2, a, b) }
    fn _mm_cmpnle_sd(a: __m128d, b: __m128d) -> __m128d { into_first!(fcmpnle!, f64x2, a, b) }
    fn _mm_cmpnlt_sd(a: __m128d, b: __m128d) -> __m128d { into_first!(fcmpnlt!, f64x2, a, b) }
    fn _mm_cmpord_sd(a: __m128d, b: __m128d) -> __m128d { into_first!(fcmpord!, f64x2, a, b) }
    fn _mm_cmpunord_sd(a: __m128d, b: __m128d) -> __m128d { into_first!(fcmpunord!, f64x2, a, b) }

    fn _mm_add_sd(a: __m128d, b: __m128d) -> __m128d { into_first!(Add::add, f64x2, a, b) }
    fn _mm_sub_sd(a: __m128d, b: __m128d) -> __m128d { into_first!(Sub::sub, f64x2, a, b) }
    fn _mm_mul_sd(a: __m128d, b: __m128d) -> __m128d { into_first!(Mul::mul, f64x2, a, b) }
    fn _mm_div_sd(a: __m128d, b: __m128d) -> __m128d { into_first!(Div::div, f64x2, a, b) }

    fn _mm_castpd_ps(a: __m128d) -> __m128 {
        unsafe { core::mem::transmute(a) }
    }

    fn _mm_castpd_si128(a: __m128d) -> __m128i {
        unsafe { core::mem::transmute(a) }
    }

    fn _mm_castps_pd(a: __m128) -> __m128d {
        unsafe { core::mem::transmute(a) }
    }

    fn _mm_castps_si128(a: __m128) -> __m128i {
        unsafe { core::mem::transmute(a) }
    }

    fn _mm_castsi128_pd(a: __m128i) -> __m128d {
        unsafe { core::mem::transmute(a) }
    }

    fn _mm_castsi128_ps(a: __m128i) -> __m128 {
        unsafe { core::mem::transmute(a) }
    }
}

intrinsic! {
    #[testvals(0, 1, 2, 3, 4, 5, 6, 7)]
    fn _mm_extract_epi16<const IMM8: i32>(a: __m128i) -> i32 {
        let a: i16x8 = a.into();
        a[(IMM8 & 0b111) as usize] as u16 as i32
    }

    #[testvals(0, 1, 2, 3, 4, 5, 6, 7)]
    fn _mm_insert_epi16<const IMM8: i32>(a: __m128i, i: i32) -> __m128i {
        let mut a: i16x8 = a.into();
        a[(IMM8 & 0b111) as usize] = (i & 0xffff) as i16;
        a.into()
    }
}

intrinsic! {
    fn _mm_madd_epi16(a: __m128i, b: __m128i) -> __m128i {
        let a: i16x8 = a.into();
        let b: i16x8 = b.into();
        let a: i32x8 = a.cast();
        let b: i32x8 = b.cast();
        let prod = a * b;
        let (first, second) = prod.deinterleave(prod);
        simd_swizzle!(first + second, [0, 1, 2, 3]).into()
    }

    fn _mm_move_epi64(a: __m128i) -> __m128i {
        let mut a: i64x2 = a.into();
        a[1] = 0;
        a.into()
    }

    fn _mm_move_sd(a: __m128d, b: __m128d) -> __m128d {
        let a: f64x2 = a.into();
        let b: f64x2 = b.into();
        mask64x2::from_array([false, true]).select(a, b).into()
    }

    fn _mm_movemask_epi8(a: __m128i) -> i32 {
        let a: i8x16 = a.into();
        a.simd_lt(i8x16::splat(0)).to_bitmask() as i32
    }

    fn _mm_movemask_pd(a: __m128d) -> i32 {
        let a: i64x2 = a.into();
        a.simd_lt(i64x2::splat(0)).to_bitmask() as i32
    }

    #[notest()]
    fn _mm_movepi64_pi64(a: __m128i) -> __m64 {
        i64x1::splat(Into::<i64x2>::into(a)[0]).into()
    }

    #[notest()]
    fn _mm_movpi64_epi64(a: __m64) -> __m128i {
        i64x2::from_array([Into::<i64x1>::into(a)[0], 0]).into()
    }

    fn _mm_mul_epu32(a: __m128i, b: __m128i) -> __m128i {
        let a: u32x4 = a.into();
        let b: u32x4 = b.into();
        let a: u64x4 = a.cast();
        let b: u64x4 = b.cast();
        simd_swizzle!(a * b, [0, 2]).into()
    }

    #[notest()]
    fn _mm_mul_su32(a: __m64, b: __m64) -> __m64 {
        let a: u32x2 = a.into();
        let b: u32x2 = b.into();
        let a: u64x2 = a.cast();
        let b: u64x2 = b.cast();
        simd_swizzle!(a * b, [0]).into()
    }

    fn _mm_mulhi_epi16(a: __m128i, b: __m128i) -> __m128i {
        let a: i16x8 = a.into();
        let b: i16x8 = b.into();
        let a: i32x8 = a.cast();
        let b: i32x8 = b.cast();
        let r: i16x8 = ((a * b) >> i32x8::splat(16)).cast();
        r.into()
    }

    fn _mm_mulhi_epu16(a: __m128i, b: __m128i) -> __m128i {
        let a: u16x8 = a.into();
        let b: u16x8 = b.into();
        let a: u32x8 = a.cast();
        let b: u32x8 = b.cast();
        let r: u16x8 = ((a * b) >> u32x8::splat(16)).cast();
        r.into()
    }

    fn _mm_mullo_epi16(a: __m128i, b: __m128i) -> __m128i {
        let a: i16x8 = a.into();
        let b: i16x8 = b.into();
        let a: i32x8 = a.cast();
        let b: i32x8 = b.cast();
        let r: i16x8 = ((a * b) & i32x8::splat(0xffff)).cast();
        r.into()
    }

    fn _mm_packs_epi16(a: __m128i, b: __m128i) -> __m128i {
        let a: i16x8 = a.into();
        let b: i16x8 = b.into();
        packs8!{ i8, a, b }.into()
    }

    fn _mm_packs_epi32(a: __m128i, b: __m128i) -> __m128i {
        let a: i32x4 = a.into();
        let b: i32x4 = b.into();
        packs4! { i16, a, b }.into()
    }

    fn _mm_packus_epi16(a: __m128i, b: __m128i) -> __m128i {
        let a: i16x8 = a.into();
        let b: i16x8 = b.into();
        packs8!{ u8, a, b }.into()
    }

    fn _mm_pause() {
        core::hint::spin_loop()
    }

    fn _mm_sad_epu8(a: __m128i, b: __m128i) -> __m128i {
        let a: u8x16 = a.into();
        let b: u8x16 = b.into();
        let diff: u16x16 = (a.simd_max(b) - a.simd_min(b)).cast();
        let first = simd_swizzle!(diff, [0, 1, 2, 3, 4, 5, 6, 7]);
        let second = simd_swizzle!(diff, [8, 9, 10, 11, 12, 13, 14, 15]);
        u16x8::from_array([
            first.reduce_sum(),
            0,
            0,
            0,
            second.reduce_sum(),
            0,
            0,
            0,
        ]).into()
    }

    #[homogenous()]
    fn _mm_set_epi8(e15: i8, e14: i8, e13: i8, e12: i8, e11: i8, e10: i8, e9: i8, e8: i8, e7: i8, e6: i8, e5: i8, e4: i8, e3: i8, e2: i8, e1: i8, e0: i8) -> __m128i {
        i8x16::from_array([e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15]).into()
    }

    #[homogenous()]
    fn _mm_setr_epi8(e15: i8, e14: i8, e13: i8, e12: i8, e11: i8, e10: i8, e9: i8, e8: i8, e7: i8, e6: i8, e5: i8, e4: i8, e3: i8, e2: i8, e1: i8, e0: i8) -> __m128i {
        i8x16::from_array([e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15]).reverse().into()
    }

    fn _mm_set_epi16(e7: i16, e6: i16, e5: i16, e4: i16, e3: i16, e2: i16, e1: i16, e0: i16) -> __m128i {
        i16x8::from_array([e0, e1, e2, e3, e4, e5, e6, e7]).into()
    }

    fn _mm_setr_epi16(e7: i16, e6: i16, e5: i16, e4: i16, e3: i16, e2: i16, e1: i16, e0: i16) -> __m128i {
        i16x8::from_array([e0, e1, e2, e3, e4, e5, e6, e7]).reverse().into()
    }

    fn _mm_set_epi32(e3: i32, e2: i32, e1: i32, e0: i32) -> __m128i {
        i32x4::from_array([e0, e1, e2, e3]).into()
    }

    fn _mm_setr_epi32(e3: i32, e2: i32, e1: i32, e0: i32) -> __m128i {
        i32x4::from_array([e0, e1, e2, e3]).reverse().into()
    }

    #[notest()]
    fn _mm_set_epi64(e1: __m64, e0: __m64) -> __m128i {
        let e1: i64x1 = e1.into();
        let e0: i64x1 = e0.into();
        i64x2::from_array([e0[0], e1[0]]).into()
    }

    #[notest()]
    fn _mm_setr_epi64(e1: __m64, e0: __m64) -> __m128i {
        let e1: i64x1 = e1.into();
        let e0: i64x1 = e0.into();
        i64x2::from_array([e1[0], e0[0]]).into()
    }

    fn _mm_set_epi64x(e1: i64, e0: i64) -> __m128i {
        i64x2::from_array([e0, e1]).into()
    }

    fn _mm_set_pd(e1: f64, e0: f64) -> __m128d {
        f64x2::from_array([e0, e1]).into()
    }

    fn _mm_setr_pd(e1: f64, e0: f64) -> __m128d {
        f64x2::from_array([e1, e0]).into()
    }

    fn _mm_set_pd1(a: f64) -> __m128d {
        f64x2::splat(a).into()
    }

    fn _mm_set_sd(a: f64) -> __m128d {
        f64x2::from_array([a, 0.]).into()
    }

    fn _mm_set1_epi8(a: i8) -> __m128i {
        i8x16::splat(a).into()
    }

    fn _mm_set1_epi16(a: i16) -> __m128i {
        i16x8::splat(a).into()
    }

    fn _mm_set1_epi32(a: i32) -> __m128i {
        i32x4::splat(a).into()
    }

    fn _mm_set1_epi64x(a: i64) -> __m128i {
        i64x2::splat(a).into()
    }

    #[notest()]
    fn _mm_set1_epi64(a: __m64) -> __m128i {
        i64x2::splat(Into::<i64x1>::into(a)[0]).into()
    }

    fn _mm_set1_pd(a: f64) -> __m128d {
        f64x2::splat(a).into()
    }

    fn _mm_setzero_pd() -> __m128d {
        f64x2::splat(0.).into()
    }

    fn _mm_setzero_si128() -> __m128i {
        i64x2::splat(0).into()
    }

    fn _mm_undefined_pd() -> __m128d {
        _mm_setzero_pd()
    }

    fn _mm_undefined_si128() -> __m128i {
        _mm_setzero_si128()
    }
}

intrinsic! {
    unsafe fn _mm_load_pd(mem_addr: *const f64) -> __m128d {
        (mem_addr as *const __m128d).read()
    }

    unsafe fn _mm_load_pd1(mem_addr: *const f64) -> __m128d {
        f64x2::splat(mem_addr.read()).into()
    }

    unsafe fn _mm_load_sd(mem_addr: *const f64) -> __m128d {
        f64x2::from_array([mem_addr.read_unaligned(), 0.]).into()
    }

    unsafe fn _mm_load1_pd(mem_addr: *const f64) -> __m128d {
        _mm_load_pd1(mem_addr)
    }

    unsafe fn _mm_load_si128(mem_addr: *const __m128i) -> __m128i {
        mem_addr.read()
    }

    unsafe fn _mm_loadh_pd(a: __m128d, mem_addr: *const f64) -> __m128d {
        let mut a: f64x2 = a.into();
        a[1] = mem_addr.read_unaligned();
        a.into()
    }

    unsafe fn _mm_loadl_pd(a: __m128d, mem_addr: *const f64) -> __m128d {
        let mut a: f64x2 = a.into();
        a[0] = mem_addr.read_unaligned();
        a.into()
    }

    unsafe fn _mm_loadl_epi64(mem_addr: *const __m128i) -> __m128i {
        let mut a = i64x2::splat(0);
        a[0] = (mem_addr as *const i64).read_unaligned();
        a.into()
    }

    unsafe fn _mm_loadr_pd(mem_addr: *const f64) -> __m128d {
        Into::<f64x2>::into(_mm_load_pd(mem_addr)).reverse().into()
    }

    unsafe fn _mm_loadu_pd(mem_addr: *const f64) -> __m128d {
        (mem_addr as *const __m128d).read_unaligned()
    }

    unsafe fn _mm_loadu_si128(mem_addr: *const __m128i) -> __m128i {
        mem_addr.read_unaligned()
    }

    unsafe fn _mm_loadu_si16(mem_addr: *const ()) -> __m128i {
        let mut a = i16x8::splat(0);
        a[0] = (mem_addr as *const i16).read_unaligned();
        a.into()
    }

    unsafe fn _mm_loadu_si32(mem_addr: *const ()) -> __m128i {
        let mut a = i32x4::splat(0);
        a[0] = (mem_addr as *const i32).read_unaligned();
        a.into()
    }

    unsafe fn _mm_loadu_si64(mem_addr: *const ()) -> __m128i {
        let mut a = i64x2::splat(0);
        a[0] = (mem_addr as *const i64).read_unaligned();
        a.into()
    }

    unsafe fn _mm_store_pd(mem_addr: *mut f64, a: __m128d) {
        (mem_addr as *mut __m128d).write(a)
    }

    unsafe fn _mm_store_pd1(mem_addr: *mut f64, a: __m128d) {
        let a: f64x2 = a.into();
        _mm_store_pd(mem_addr, simd_swizzle!(a, [0, 0]).into())
    }

    unsafe fn _mm_store_sd(mem_addr: *mut f64, a: __m128d) {
        let a: f64x2 = a.into();
        mem_addr.write_unaligned(a[0])
    }

    unsafe fn _mm_store_si128(mem_addr: *mut __m128i, a: __m128i) {
        mem_addr.write(a)
    }

    unsafe fn _mm_store1_pd(mem_addr: *mut f64, a: __m128d) {
        _mm_store_pd1(mem_addr, a)
    }

    unsafe fn _mm_storeh_pd(mem_addr: *mut f64, a: __m128d) {
        let a: f64x2 = a.into();
        mem_addr.write(a[1]);
    }

    unsafe fn _mm_storel_pd(mem_addr: *mut f64, a: __m128d) {
        let a: f64x2 = a.into();
        mem_addr.write(a[0]);
    }

    unsafe fn _mm_storel_epi64(mem_addr: *mut __m128i, a: __m128i) {
        let a: i64x2 = a.into();
        (mem_addr as *mut i64).write(a[0])
    }

    unsafe fn _mm_storer_pd(mem_addr: *mut f64, a: __m128d) {
        (mem_addr as *mut f64x2).write(Into::<f64x2>::into(a).reverse())
    }

    unsafe fn _mm_storeu_pd(mem_addr: *mut f64, a: __m128d) {
        (mem_addr as *mut __m128d).write_unaligned(a)
    }

    unsafe fn _mm_storeu_si128(mem_addr: *mut __m128i, a: __m128i) {
        mem_addr.write_unaligned(a)
    }

    unsafe fn _mm_storeu_si16(mem_addr: *mut (), a: __m128i) {
        (mem_addr as *mut i16).write_unaligned(Into::<i16x8>::into(a)[0])
    }

    unsafe fn _mm_storeu_si32(mem_addr: *mut (), a: __m128i) {
        (mem_addr as *mut i32).write_unaligned(Into::<i32x4>::into(a)[0])
    }

    unsafe fn _mm_storeu_si64(mem_addr: *mut (), a: __m128i) {
        (mem_addr as *mut i64).write_unaligned(Into::<i64x2>::into(a)[0])
    }
}

intrinsic! {
    fn _mm_shuffle_epi32<const IMM8: i32>(a: __m128i) -> __m128i {
        let a: i32x4 = a.into();
        shuffle4! { IMM8, a }.into()
    }
}

macro_rules! comi {
    { $f:ident, $a:expr, $b:expr } => {
        {
            let a: f64x2 = $a.into();
            let b: f64x2 = $b.into();
            a[0].$f(&b[0]) as i32
        }
    }
}

intrinsic! {
    fn _mm_comieq_sd(a: __m128d, b: __m128d) -> i32 { comi!(eq, a, b) }
    fn _mm_comige_sd(a: __m128d, b: __m128d) -> i32 { comi!(ge, a, b) }
    fn _mm_comigt_sd(a: __m128d, b: __m128d) -> i32 { comi!(gt, a, b) }
    fn _mm_comile_sd(a: __m128d, b: __m128d) -> i32 { comi!(le, a, b) }
    fn _mm_comilt_sd(a: __m128d, b: __m128d) -> i32 { comi!(lt, a, b) }
    fn _mm_comineq_sd(a: __m128d, b: __m128d) -> i32 { comi!(ne, a, b) }
    fn _mm_ucomieq_sd(a: __m128d, b: __m128d) -> i32 { comi!(eq, a, b) }
    fn _mm_ucomige_sd(a: __m128d, b: __m128d) -> i32 { comi!(ge, a, b) }
    fn _mm_ucomigt_sd(a: __m128d, b: __m128d) -> i32 { comi!(gt, a, b) }
    fn _mm_ucomile_sd(a: __m128d, b: __m128d) -> i32 { comi!(le, a, b) }
    fn _mm_ucomilt_sd(a: __m128d, b: __m128d) -> i32 { comi!(lt, a, b) }
    fn _mm_ucomineq_sd(a: __m128d, b: __m128d) -> i32 { comi!(ne, a, b) }
}

intrinsic! {
    fn _mm_slli_epi16<const IMM8: i32>(a: __m128i) -> __m128i { shift_logical!(Shl::shl, i16x8, a, IMM8) }
    fn _mm_slli_epi32<const IMM8: i32>(a: __m128i) -> __m128i { shift_logical!(Shl::shl, i32x4, a, IMM8) }
    fn _mm_slli_epi64<const IMM8: i32>(a: __m128i) -> __m128i { shift_logical!(Shl::shl, i64x2, a, IMM8) }
    fn _mm_srli_epi16<const IMM8: i32>(a: __m128i) -> __m128i { shift_logical!(Shr::shr, u16x8, a, IMM8) }
    fn _mm_srli_epi32<const IMM8: i32>(a: __m128i) -> __m128i { shift_logical!(Shr::shr, u32x4, a, IMM8) }
    fn _mm_srli_epi64<const IMM8: i32>(a: __m128i) -> __m128i { shift_logical!(Shr::shr, u64x2, a, IMM8) }
    fn _mm_srai_epi16<const IMM8: i32>(a: __m128i) -> __m128i { shift_right!(i16x8, a, IMM8) }
    fn _mm_srai_epi32<const IMM8: i32>(a: __m128i) -> __m128i { shift_right!(i32x4, a, IMM8) }
}

intrinsic! {
    fn _mm_sll_epi16(a: __m128i, b: __m128i) -> __m128i { sxl!(Shl::shl, i16x8, u64x2, a, b) }
    fn _mm_sll_epi32(a: __m128i, b: __m128i) -> __m128i { sxl!(Shl::shl, i32x4, u64x2, a, b) }
    fn _mm_sll_epi64(a: __m128i, b: __m128i) -> __m128i { sxl!(Shl::shl, i64x2, u64x2, a, b) }
    fn _mm_srl_epi16(a: __m128i, b: __m128i) -> __m128i { sxl!(Shr::shr, u16x8, u64x2, a, b) }
    fn _mm_srl_epi32(a: __m128i, b: __m128i) -> __m128i { sxl!(Shr::shr, u32x4, u64x2, a, b) }
    fn _mm_srl_epi64(a: __m128i, b: __m128i) -> __m128i { sxl!(Shr::shr, u64x2, u64x2, a, b) }
    fn _mm_sra_epi16(a: __m128i, b: __m128i) -> __m128i { sra!(i16x8, u64x2, a, b) }
    fn _mm_sra_epi32(a: __m128i, b: __m128i) -> __m128i { sra!(i32x4, u64x2, a, b) }
}

// TODO cvt
// TODO bslli, bsrli, slli_si128
// TODO maskmove
// TODO shuffle_pd, shufflehi, shufflelo
// TODO sqrt
// TODO stream

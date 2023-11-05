use super::*;
use core::simd::ToBitMask;

binary! {
    _mm_add_epi8, Add::add, __m128i as i8x16;
    _mm_add_epi16, Add::add, __m128i as i16x8;
    _mm_add_epi32, Add::add, __m128i as i32x4;
    _mm_add_epi64, Add::add, __m128i as i64x2;

    _mm_add_si64, Add::add, __m64 as i64x1;

    _mm_add_pd, Add::add, __m128d as f64x2;
    _mm_sub_pd, Sub::sub, __m128d as f64x2;
    _mm_mul_pd, Mul::mul, __m128d as f64x2;
    _mm_div_pd, Div::div, __m128d as f64x2;

    _mm_adds_epi8, SimdInt::saturating_add, __m128i as i8x16;
    _mm_adds_epi16, SimdInt::saturating_add, __m128i as i16x8;
    _mm_adds_epu8, SimdUint::saturating_add, __m128i as u8x16;
    _mm_adds_epu16, SimdUint::saturating_add, __m128i as u16x8;

    _mm_and_pd, BitAnd::bitand, __m128d as i64x2;
    _mm_or_pd, BitOr::bitor, __m128d as i64x2;
    _mm_and_si128, BitAnd::bitand, __m128i as i64x2;
    _mm_or_si128, BitOr::bitor, __m128i as i64x2;

    _mm_avg_epu8, pavgb, __m128i as u8x16;
    _mm_avg_epu16, pavgw, __m128i as u16x8;

    _mm_max_epi16, SimdOrd::simd_max, __m128i as i16x8;
    _mm_max_epu8, SimdOrd::simd_max, __m128i as u8x16;
    _mm_min_epi16, SimdOrd::simd_min, __m128i as i16x8;
    _mm_min_epu8, SimdOrd::simd_min, __m128i as u8x16;
}

binary! {
    _mm_max_pd, macro float_max, __m128d as f64x2;
    _mm_min_pd, macro float_min, __m128d as f64x2;

    _mm_andnot_pd, macro andnot, __m128d as i64x2;
    _mm_andnot_si128, macro andnot, __m128i as i64x2;

    _mm_cmpeq_epi8, macro cmpeq, __m128i as i8x16;
    _mm_cmpeq_epi16, macro cmpeq, __m128i as i16x8;
    _mm_cmpeq_epi32, macro cmpeq, __m128i as i32x4;
    _mm_cmpgt_epi8, macro cmpgt, __m128i as i8x16;
    _mm_cmpgt_epi16, macro cmpgt, __m128i as i16x8;
    _mm_cmpgt_epi32, macro cmpgt, __m128i as i32x4;
    _mm_cmplt_epi8, macro cmplt, __m128i as i8x16;
    _mm_cmplt_epi16, macro cmplt, __m128i as i16x8;
    _mm_cmplt_epi32, macro cmplt, __m128i as i32x4;

    _mm_cmpeq_pd, macro cmpeq, __m128d as f64x2;
    _mm_cmpge_pd, macro cmpge, __m128d as f64x2;
    _mm_cmpgt_pd, macro cmpgt, __m128d as f64x2;
    _mm_cmple_pd, macro cmple, __m128d as f64x2;
    _mm_cmplt_pd, macro cmplt, __m128d as f64x2;
    _mm_cmpneq_pd, macro cmpneq, __m128d as f64x2;
    _mm_cmpnge_pd, macro cmpnge, __m128d as f64x2;
    _mm_cmpngt_pd, macro cmpngt, __m128d as f64x2;
    _mm_cmpnle_pd, macro cmpnle, __m128d as f64x2;
    _mm_cmpnlt_pd, macro cmpnlt, __m128d as f64x2;
    _mm_cmpord_pd, macro cmpord, __m128d as f64x2;
    _mm_cmpunord_pd, macro cmpunord, __m128d as f64x2;
}

binary_one_element! {
    _mm_max_sd, macro float_max, __m128d as f64x2;
    _mm_min_sd, macro float_min, __m128d as f64x2;

    _mm_cmpeq_sd, macro cmpeq, __m128d as f64x2;
    _mm_cmpge_sd, macro cmpge, __m128d as f64x2;
    _mm_cmpgt_sd, macro cmpgt, __m128d as f64x2;
    _mm_cmple_sd, macro cmple, __m128d as f64x2;
    _mm_cmplt_sd, macro cmplt, __m128d as f64x2;
    _mm_cmpneq_sd, macro cmpneq, __m128d as f64x2;
    _mm_cmpnge_sd, macro cmpnge, __m128d as f64x2;
    _mm_cmpngt_sd, macro cmpngt, __m128d as f64x2;
    _mm_cmpnle_sd, macro cmpnle, __m128d as f64x2;
    _mm_cmpnlt_sd, macro cmpnlt, __m128d as f64x2;
    _mm_cmpord_sd, macro cmpord, __m128d as f64x2;
    _mm_cmpunord_sd, macro cmpunord, __m128d as f64x2;
}

binary_one_element! {
    _mm_add_sd, Add::add, __m128d as f64x2;
    _mm_sub_sd, Sub::sub, __m128d as f64x2;
    _mm_mul_sd, Mul::mul, __m128d as f64x2;
    _mm_div_sd, Div::div, __m128d as f64x2;
}

intrinsic! {
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

    fn _mm_extract_epi16(a: __m128i, imm8: i32) -> i32 {
        let a: i16x8 = a.into();
        a[(imm8 & 0b111) as usize] as i32
    }

    fn _mm_insert_epi16(a: __m128i, i: i32, imm8: i32) -> __m128i {
        let mut a: i16x8 = a.into();
        a[(imm8 & 0x111) as usize] = i as i16;
        a.into()
    }

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

    fn _mm_movepi64_pi64(a: __m128i) -> __m64 {
        i64x1::splat(Into::<i64x2>::into(a)[0]).into()
    }

    fn _mm_movpi64_epi64(a: __m64) -> __m128i {
        i64x2::from_array([Into::<i64x1>::into(a)[0], 0]).into()
    }

    fn _mm_mul_epi32(a: __m128i, b: __m128i) -> __m128i {
        let a: u32x4 = a.into();
        let b: u32x4 = b.into();
        let a: u64x4 = a.cast();
        let b: u64x4 = b.cast();
        simd_swizzle!(a * b, [0, 1]).into()
    }

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
}

macro_rules! comi {
    { $($name:ident, $func:ident;)* } => {
        intrinsic! {
            $(
            fn $name(a: __m128d, b: __m128d) -> i32 {
                let a: f64x2 = a.into();
                let b: f64x2 = b.into();
                a[0].$func(&b[0]) as i32
            }
            )*
        }
    }
}

comi! {
    _mm_comieq_sd, eq;
    _mm_comige_sd, ge;
    _mm_comigt_sd, gt;
    _mm_comile_sd, le;
    _mm_comilt_sd, lt;
    _mm_comineq_sd, ne;
}

// TODO cvt
// TODO bslli, bsrli
// TODO maskmove

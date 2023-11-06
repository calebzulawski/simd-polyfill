use super::*;
use core::simd::ToBitMask;

binary! {
    _mm_add_epi8, Add::add, __m128i as i8x16;
    _mm_add_epi16, Add::add, __m128i as i16x8;
    _mm_add_epi32, Add::add, __m128i as i32x4;
    _mm_add_epi64, Add::add, __m128i as i64x2;
    _mm_sub_epi8, Sub::sub, __m128i as i8x16;
    _mm_sub_epi16, Sub::sub, __m128i as i16x8;
    _mm_sub_epi32, Sub::sub, __m128i as i32x4;
    _mm_sub_epi64, Sub::sub, __m128i as i64x2;

    _mm_add_si64, Add::add, __m64 as i64x1;
    _mm_sub_si64, Sub::sub, __m64 as i64x1;

    _mm_add_pd, Add::add, __m128d as f64x2;
    _mm_sub_pd, Sub::sub, __m128d as f64x2;
    _mm_mul_pd, Mul::mul, __m128d as f64x2;
    _mm_div_pd, Div::div, __m128d as f64x2;

    _mm_adds_epi8, SimdInt::saturating_add, __m128i as i8x16;
    _mm_adds_epi16, SimdInt::saturating_add, __m128i as i16x8;
    _mm_adds_epu8, SimdUint::saturating_add, __m128i as u8x16;
    _mm_adds_epu16, SimdUint::saturating_add, __m128i as u16x8;
    _mm_subs_epi8, SimdInt::saturating_sub, __m128i as i8x16;
    _mm_subs_epi16, SimdInt::saturating_sub, __m128i as i16x8;
    _mm_subs_epu8, SimdUint::saturating_sub, __m128i as u8x16;
    _mm_subs_epu16, SimdUint::saturating_sub, __m128i as u16x8;

    _mm_and_pd, BitAnd::bitand, __m128d as i64x2;
    _mm_or_pd, BitOr::bitor, __m128d as i64x2;
    _mm_xor_pd, BitXor::bitxor, __m128d as i64x2;
    _mm_and_si128, BitAnd::bitand, __m128i as i64x2;
    _mm_or_si128, BitOr::bitor, __m128i as i64x2;
    _mm_xor_si128, BitXor::bitxor, __m128i as i64x2;

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

    _mm_unpackhi_epi8, macro unpackhi, __m128i as i8x16;
    _mm_unpackhi_epi16, macro unpackhi, __m128i as i16x8;
    _mm_unpackhi_epi32, macro unpackhi, __m128i as i32x4;
    _mm_unpackhi_epi64, macro unpackhi, __m128i as i64x2;
    _mm_unpacklo_epi8, macro unpacklo, __m128i as i8x16;
    _mm_unpacklo_epi16, macro unpacklo, __m128i as i16x8;
    _mm_unpacklo_epi32, macro unpacklo, __m128i as i32x4;
    _mm_unpacklo_epi64, macro unpacklo, __m128i as i64x2;

    _mm_unpacklo_pd, macro unpacklo, __m128d as f64x2;
    _mm_unpackhi_pd, macro unpackhi, __m128d as f64x2;
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

    fn _mm_set_epi8(e15: i8, e14: i8, e13: i8, e12: i8, e11: i8, e10: i8, e9: i8, e8: i8, e7: i8, e6: i8, e5: i8, e4: i8, e3: i8, e2: i8, e1: i8, e0: i8) -> __m128i {
        i8x16::from_array([e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15]).into()
    }

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

    fn _mm_set_epi64(e1: __m64, e0: __m64) -> __m128i {
        let e1: i64x1 = e1.into();
        let e0: i64x1 = e0.into();
        i64x2::from_array([e0[0], e1[0]]).into()
    }

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

    _mm_ucomieq_sd, eq;
    _mm_ucomige_sd, ge;
    _mm_ucomigt_sd, gt;
    _mm_ucomile_sd, le;
    _mm_ucomilt_sd, lt;
    _mm_ucomineq_sd, ne;
}

macro_rules! shift {
    { $($imm:ident, $func:path, $ty:ty;)* } => {
        intrinsic! {
            $(
            fn $imm(a: __m128i, imm8: i32) -> __m128i {
                let a: $ty = a.into();
                $func(a, <$ty>::splat(imm8 as _)).into()
            }
            )*
        }
    }
}

shift! {
    _mm_slli_epi16, Shl::shl, i16x8;
    _mm_slli_epi32, Shl::shl, i32x4;
    _mm_slli_epi64, Shl::shl, i64x2;
    _mm_srai_epi16, Shr::shr, i16x8;
    _mm_srai_epi32, Shr::shr, i32x4;
    _mm_srai_epi64, Shr::shr, i64x2;
    _mm_srli_epi16, Shr::shr, u16x8;
    _mm_srli_epi32, Shr::shr, u32x4;
    _mm_srli_epi64, Shr::shr, u64x2;
}

binary! {
    _mm_sll_epi16, Shl::shl, __m128i as i16x8;
    _mm_sll_epi32, Shl::shl, __m128i as i32x4;
    _mm_sll_epi64, Shl::shl, __m128i as i64x2;
    _mm_sra_epi16, Shr::shr, __m128i as i16x8;
    _mm_sra_epi32, Shr::shr, __m128i as i32x4;
    _mm_sra_epi64, Shr::shr, __m128i as i64x2;
    _mm_srl_epi16, Shr::shr, __m128i as u16x8;
    _mm_srl_epi32, Shr::shr, __m128i as u32x4;
    _mm_srl_epi64, Shr::shr, __m128i as u64x2;
}

// TODO cvt
// TODO bslli, bsrli, slli_si128
// TODO maskmove
// TODO shuffle_pd, shufflehi, shufflelo
// TODO sqrt
// TODO stream

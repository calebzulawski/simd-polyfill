use super::*;
use std::simd::ToBitMask;

binary! {
    _mm_add_ps, Add::add, __m128 as f32x4;
    _mm_div_ps, Div::div, __m128 as f32x4;
    _mm_mul_ps, Mul::mul, __m128 as f32x4;
    _mm_sub_ps, Sub::sub, __m128 as f32x4;

    _mm_max_pu8, SimdOrd::simd_max, __m64 as u8x8;
    _mm_max_pi16, SimdOrd::simd_max, __m64 as i16x4;
    _mm_min_pu8, SimdOrd::simd_min, __m64 as u8x8;
    _mm_min_pi16, SimdOrd::simd_min, __m64 as i16x4;

    _mm_and_ps, BitAnd::bitand, __m128 as i32x4;
    _mm_or_ps, BitOr::bitor, __m128 as i32x4;
    _mm_xor_ps, BitXor::bitxor, __m128 as i32x4;

    _mm_avg_pu16, pavgw, __m64 as u16x4;
    _mm_avg_pu8, pavgb, __m64 as u8x8;
}

binary! {
    _mm_andnot_ps, macro andnot, __m128 as i32x4;

    _mm_max_ps, macro float_max, __m128 as f32x4;
    _mm_min_ps, macro float_min, __m128 as f32x4;

    _mm_unpackhi_ps, macro unpackhi, __m128 as f32x4;
    _mm_unpacklo_ps, macro unpacklo, __m128 as f32x4;

    _mm_cmpeq_ps, macro cmpeq, __m128 as f32x4;
    _mm_cmpge_ps, macro cmpge, __m128 as f32x4;
    _mm_cmpgt_ps, macro cmpgt, __m128 as f32x4;
    _mm_cmple_ps, macro cmple, __m128 as f32x4;
    _mm_cmplt_ps, macro cmplt, __m128 as f32x4;
    _mm_cmpneq_ps, macro cmpneq, __m128 as f32x4;
    _mm_cmpnge_ps, macro cmpnge, __m128 as f32x4;
    _mm_cmpngt_ps, macro cmpngt, __m128 as f32x4;
    _mm_cmpnle_ps, macro cmpnle, __m128 as f32x4;
    _mm_cmpnlt_ps, macro cmpnlt, __m128 as f32x4;
    _mm_cmpord_ps, macro cmpord, __m128 as f32x4;
    _mm_cmpunord_ps, macro cmpunord, __m128 as f32x4;
}

binary_one_element! {
    _mm_add_ss, Add::add, __m128 as f32x4;
    _mm_div_ss, Div::div, __m128 as f32x4;
    _mm_mul_ss, Mul::mul, __m128 as f32x4;
    _mm_sub_ss, Sub::sub, __m128 as f32x4;
}

binary_one_element! {
    _mm_max_ss, macro float_max, __m128 as f32x4;
    _mm_min_ss, macro float_min, __m128 as f32x4;

    _mm_cmpeq_ss, macro cmpeq, __m128 as f32x4;
    _mm_cmpge_ss, macro cmpge, __m128 as f32x4;
    _mm_cmpgt_ss, macro cmpgt, __m128 as f32x4;
    _mm_cmple_ss, macro cmple, __m128 as f32x4;
    _mm_cmplt_ss, macro cmplt, __m128 as f32x4;
    _mm_cmpneq_ss, macro cmpneq, __m128 as f32x4;
    _mm_cmpnge_ss, macro cmpnge, __m128 as f32x4;
    _mm_cmpngt_ss, macro cmpngt, __m128 as f32x4;
    _mm_cmpnle_ss, macro cmpnle, __m128 as f32x4;
    _mm_cmpnlt_ss, macro cmpnlt, __m128 as f32x4;
    _mm_cmpord_ss, macro cmpord, __m128 as f32x4;
    _mm_cmpunord_ss, macro cmpunord, __m128 as f32x4;
}

macro_rules! comi {
    { $($name:ident, $func:ident;)* } => {
        intrinsic! {
            $(
            fn $name(a: __m128, b: __m128) -> i32 {
                let a: f32x4 = a.into();
                let b: f32x4 = b.into();
                a[0].$func(&b[0]) as i32
            }
            )*
        }
    }
}

comi! {
    _mm_comieq_ss, eq;
    _mm_comige_ss, ge;
    _mm_comigt_ss, gt;
    _mm_comile_ss, le;
    _mm_comilt_ss, lt;
    _mm_comineq_ss, ne;

    _mm_ucomieq_ss, eq;
    _mm_ucomige_ss, ge;
    _mm_ucomigt_ss, gt;
    _mm_ucomile_ss, le;
    _mm_ucomilt_ss, lt;
    _mm_ucomineq_ss, ne;
}

intrinsic! {
    fn _mm_extract_pi16(a: __m64, imm8: i32) -> i32 {
        let a: i16x4 = a.into();
        a[(imm8 & 0x3) as usize] as i32
    }

    fn _mm_insert_pi16(a: __m64, i: i32, imm8: i32) -> __m64 {
        let mut a: i16x4 = a.into();
        a[(imm8 & 0x3) as usize] = i as i16;
        a.into()
    }

    fn _mm_move_ss(a: __m128, b: __m128) -> __m128 {
        let a: f32x4 = a.into();
        let b: f32x4 = b.into();
        mask32x4::from_array([false, true, true, true]).select(a, b).into()
    }

    fn _mm_movehl_ps(a: __m128, b: __m128) -> __m128 {
        let a: f32x4 = a.into();
        let b: f32x4 = b.into();
        use core::simd::Which::*;
        simd_swizzle!(a, b, [Second(2), Second(3), First(2), First(3)]).into()
    }

    fn _mm_movelh_ps(a: __m128, b: __m128) -> __m128 {
        let a: f32x4 = a.into();
        let b: f32x4 = b.into();
        use core::simd::Which::*;
        simd_swizzle!(a, b, [First(0), First(1), Second(0), Second(1)]).into()
    }

    fn _mm_movemask_pi8(a: __m64) -> i32 {
        let a: i8x8 = a.into();
        a.simd_lt(i8x8::splat(0)).to_bitmask() as i32
    }

    fn _mm_movemask_ps(a: __m128) -> i32 {
        let a: f32x4 = a.into();
        let a: i32x4 = unsafe { core::mem::transmute(a) };
        a.simd_lt(i32x4::splat(0)).to_bitmask() as i32
    }

    fn _mm_mulhi_pu16(a: __m64, b: __m64) -> __m64 {
        let a: u16x4 = a.into();
        let b: u16x4 = b.into();
        let a: u32x4 = a.cast();
        let b: u32x4 = b.cast();
        let r: u16x4 = ((a * b) >> u32x4::splat(16)).cast();
        r.into()
    }

    fn _mm_sad_pu8(a: __m64, b: __m64) -> __m64 {
        let a: u8x8 = a.into();
        let b: u8x8 = b.into();
        let diff: u16x8 = (a.simd_max(b) - a.simd_min(b)).cast();
        u16x4::from_array([
            diff.reduce_sum(),
            0,
            0,
            0,
        ]).into()
    }

    fn _mm_set_ps(e3: f32, e2: f32, e1: f32, e0: f32) -> __m128 {
        f32x4::from_array([e0, e1, e2, e3]).into()
    }

    fn _mm_set_ps1(a: f32) -> __m128 {
        f32x4::splat(a).into()
    }

    fn _mm_set_ss(a: f32) -> __m128 {
        _mm_set_ps(0., 0., 0., a).into()
    }

    fn _mm_set1_ps(a: f32) -> __m128 {
        f32x4::splat(a).into()
    }

    fn _mm_setr_ps(e3: f32, e2: f32, e1: f32, e0: f32) -> __m128 {
        f32x4::from_array([e3, e2, e1, e0]).into()
    }

    fn _mm_setzero_ps() -> __m128 {
        f32x4::splat(0.).into()
    }

    fn _mm_undefined_ps() -> __m128 {
        // Rust can't have undefined values anyway
        _mm_setzero_ps()
    }
}

intrinsic! {
    #[intrinsic = _mm_shuffle_pi16]
    pub fn _mm_shuffle_pi16<const IMM8: i32>(a: __m64) -> __m64 {
        let a: i16x4 = a.into();
        use core::simd::Swizzle;
        struct Shuffle<const IMM8: i32>;
        impl<const IMM8: i32> Swizzle<4, 4> for Shuffle<IMM8> {
            const INDEX: [usize; 4] = [
                (IMM8 as usize >> 0) & 0x3,
                (IMM8 as usize >> 2) & 0x3,
                (IMM8 as usize >> 4) & 0x3,
                (IMM8 as usize >> 8) & 0x3,
            ];
        }
        Shuffle::<IMM8>::swizzle(a).into()
    }

    #[intrinsic = _mm_shuffle_ps]
    pub fn _mm_shuffle_ps<const IMM8: i32>(a: __m128) -> __m128 {
        let a: f32x4 = a.into();
        use core::simd::Swizzle;
        struct Shuffle<const IMM8: i32>;
        impl<const IMM8: i32> Swizzle<4, 4> for Shuffle<IMM8> {
            const INDEX: [usize; 4] = [
                (IMM8 as usize >> 0) & 0x3,
                (IMM8 as usize >> 2) & 0x3,
                (IMM8 as usize >> 4) & 0x3,
                (IMM8 as usize >> 8) & 0x3,
            ];
        }
        Shuffle::<IMM8>::swizzle(a).into()
    }
}

intrinsic! {
    unsafe fn _mm_load_ps(mem_addr: *const f32) -> __m128 {
        (mem_addr as *const __m128).read()
    }

    unsafe fn _mm_load_ps1(mem_addr: *const f32) -> __m128 {
        f32x4::splat(mem_addr.read()).into()
    }

    unsafe fn _mm_load_ss(mem_addr: *const f32) -> __m128 {
        f32x4::from_array([mem_addr.read(), 0., 0., 0.]).into()
    }

    unsafe fn _mm_load1_ps(mem_addr: *const f32) -> __m128 {
        _mm_load_ps1(mem_addr)
    }

    unsafe fn _mm_loadl_pi(a: __m128, mem_addr: *const __m64) -> __m128 {
        let mut a: f32x4 = a.into();
        let b: [f32; 2] = (mem_addr as *const [f32; 2]).read_unaligned();
        a[2..].copy_from_slice(b.as_ref());
        a.into()
    }

    unsafe fn _mm_loadh_pi(a: __m128, mem_addr: *const __m64) -> __m128 {
        let mut a: f32x4 = a.into();
        let b: [f32; 2] = (mem_addr as *const [f32; 2]).read_unaligned();
        a[..2].copy_from_slice(b.as_ref());
        a.into()
    }

    unsafe fn _mm_loadr_ps(mem_addr: *const f32) -> __m128 {
        Into::<f32x4>::into(_mm_load_ps(mem_addr)).reverse().into()
    }

    unsafe fn _mm_loadu_ps(mem_addr: *const f32) -> __m128 {
        (mem_addr as *const __m128).read_unaligned()
    }

    unsafe fn _mm_store_ps(mem_addr: *mut f32, a: __m128) {
        (mem_addr as *mut __m128).write(a)
    }

    unsafe fn _mm_store_ps1(mem_addr: *mut f32, a: __m128) {
        let a: f32x4 = a.into();
        _mm_store_ps(mem_addr, simd_swizzle!(a, [0, 0, 0, 0]).into())
    }

    unsafe fn _mm_store_ss(mem_addr: *mut f32, a: __m128) {
        let a: f32x4 = a.into();
        mem_addr.write_unaligned(a[0])
    }

    unsafe fn _mm_store1_ps(mem_addr: *mut f32, a: __m128) {
        _mm_store_ps1(mem_addr, a)
    }

    unsafe fn _mm_storeh_pi(mem_addr: *mut __m64, a: __m128) {
        let a: i32x4 = a.into();
        (mem_addr as *mut [i32; 2]).write_unaligned([a[2], a[3]])
    }

    unsafe fn _mm_storel_pi(mem_addr: *mut __m64, a: __m128) {
        let a: i32x4 = a.into();
        (mem_addr as *mut [i32; 2]).write_unaligned([a[0], a[1]])
    }

    unsafe fn _mm_storer_ps(mem_addr: *mut f32, a: __m128) {
        _mm_store_ps(mem_addr, Into::<f32x4>::into(a).reverse().into())
    }

    unsafe fn _mm_storeu_ps(mem_addr: *mut f32, a: __m128) {
        (mem_addr as *mut __m128).write_unaligned(a)
    }
}

// TODO cvt
// TODO malloc/free
// TODO maskmov
// TODO rcp/rsqrt/sqrt
// TODO stream

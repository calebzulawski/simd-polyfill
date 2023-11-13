use super::*;
use crate::{into, into_first};
use core::simd::ToBitMask;

macro_rules! comi {
    { $f:ident, $a:expr, $b:expr } => {
        {
            let a: f32x4 = $a.into();
            let b: f32x4 = $b.into();
            a[0].$f(&b[0]) as i32
        }
    }
}

intrinsic! {
    fn _mm_add_ps(a: __m128, b: __m128) -> __m128 { into!(Add::add, f32x4, a, b) }
    fn _mm_div_ps(a: __m128, b: __m128) -> __m128 { into!(Div::div, f32x4, a, b) }
    fn _mm_mul_ps(a: __m128, b: __m128) -> __m128 { into!(Mul::mul, f32x4, a, b) }
    fn _mm_sub_ps(a: __m128, b: __m128) -> __m128 { into!(Sub::sub, f32x4, a, b) }

    #[notest()] fn _mm_max_pu8 (a: __m64, b: __m64) -> __m64 { into!(SimdOrd::simd_max, u8x8,  a, b) }
    #[notest()] fn _mm_max_pi16(a: __m64, b: __m64) -> __m64 { into!(SimdOrd::simd_max, i16x4, a, b) }
    #[notest()] fn _mm_min_pu8 (a: __m64, b: __m64) -> __m64 { into!(SimdOrd::simd_min, u8x8,  a, b) }
    #[notest()] fn _mm_min_pi16(a: __m64, b: __m64) -> __m64 { into!(SimdOrd::simd_min, i16x4, a, b) }

    fn _mm_and_ps(a: __m128, b: __m128) -> __m128 { into!(BitAnd::bitand, i32x4, a, b) }
    fn _mm_or_ps (a: __m128, b: __m128) -> __m128 { into!(BitOr::bitor,   i32x4, a, b) }
    fn _mm_xor_ps(a: __m128, b: __m128) -> __m128 { into!(BitXor::bitxor, i32x4, a, b) }
    fn _mm_andnot_ps(a: __m128, b: __m128) -> __m128 { into!(andnot!, i32x4, a, b) }

    #[notest()] fn _mm_avg_pu16(a: __m64, b: __m64) -> __m64 { into!(pavgw, u16x4, a, b) }
    #[notest()] fn _mm_avg_pu8 (a: __m64, b: __m64) -> __m64 { into!(pavgb, u8x8, a, b) }

    fn _mm_max_ps(a: __m128, b: __m128) -> __m128 { into!(float_max!, f32x4, a, b) }
    fn _mm_min_ps(a: __m128, b: __m128) -> __m128 { into!(float_min!, f32x4, a, b) }

    fn _mm_unpackhi_ps(a: __m128, b: __m128) -> __m128 { into!(unpackhi!, f32x4, a, b) }
    fn _mm_unpacklo_ps(a: __m128, b: __m128) -> __m128 { into!(unpacklo!, f32x4, a, b) }

    fn _mm_cmpeq_ps(a: __m128, b: __m128) -> __m128 { into!(cmpeq!, f32x4, a, b) }
    fn _mm_cmpge_ps(a: __m128, b: __m128) -> __m128 { into!(cmpge!, f32x4, a, b) }
    fn _mm_cmpgt_ps(a: __m128, b: __m128) -> __m128 { into!(cmpgt!, f32x4, a, b) }
    fn _mm_cmple_ps(a: __m128, b: __m128) -> __m128 { into!(cmple!, f32x4, a, b) }
    fn _mm_cmplt_ps(a: __m128, b: __m128) -> __m128 { into!(cmplt!, f32x4, a, b) }
    fn _mm_cmpneq_ps(a: __m128, b: __m128) -> __m128 { into!(cmpneq!, f32x4, a, b) }
    fn _mm_cmpnge_ps(a: __m128, b: __m128) -> __m128 { into!(cmpnge!, f32x4, a, b) }
    fn _mm_cmpngt_ps(a: __m128, b: __m128) -> __m128 { into!(cmpngt!, f32x4, a, b) }
    fn _mm_cmpnle_ps(a: __m128, b: __m128) -> __m128 { into!(cmpnle!, f32x4, a, b) }
    fn _mm_cmpnlt_ps(a: __m128, b: __m128) -> __m128 { into!(cmpnlt!, f32x4, a, b) }
    fn _mm_cmpord_ps(a: __m128, b: __m128) -> __m128 { into!(cmpord!, f32x4, a, b) }
    fn _mm_cmpunord_ps(a: __m128, b: __m128) -> __m128 { into!(cmpunord!, f32x4, a, b) }

    fn _mm_add_ss(a: __m128, b: __m128) -> __m128 { into_first!(Add::add, f32x4, a, b) }
    fn _mm_div_ss(a: __m128, b: __m128) -> __m128 { into_first!(Div::div, f32x4, a, b) }
    fn _mm_mul_ss(a: __m128, b: __m128) -> __m128 { into_first!(Mul::mul, f32x4, a, b) }
    fn _mm_sub_ss(a: __m128, b: __m128) -> __m128 { into_first!(Sub::sub, f32x4, a, b) }

    fn _mm_max_ss(a: __m128, b: __m128) -> __m128 { into_first!(float_max!, f32x4, a, b) }
    fn _mm_min_ss(a: __m128, b: __m128) -> __m128 { into_first!(float_min!, f32x4, a, b) }

    fn _mm_cmpeq_ss(a: __m128, b: __m128) -> __m128 { into_first!(fcmpeq!, f32x4, a, b) }
    fn _mm_cmpge_ss(a: __m128, b: __m128) -> __m128 { into_first!(fcmpge!, f32x4, a, b) }
    fn _mm_cmpgt_ss(a: __m128, b: __m128) -> __m128 { into_first!(fcmpgt!, f32x4, a, b) }
    fn _mm_cmple_ss(a: __m128, b: __m128) -> __m128 { into_first!(fcmple!, f32x4, a, b) }
    fn _mm_cmplt_ss(a: __m128, b: __m128) -> __m128 { into_first!(fcmplt!, f32x4, a, b) }
    fn _mm_cmpneq_ss(a: __m128, b: __m128) -> __m128 { into_first!(fcmpneq!, f32x4, a, b) }
    fn _mm_cmpnge_ss(a: __m128, b: __m128) -> __m128 { into_first!(fcmpnge!, f32x4, a, b) }
    fn _mm_cmpngt_ss(a: __m128, b: __m128) -> __m128 { into_first!(fcmpngt!, f32x4, a, b) }
    fn _mm_cmpnle_ss(a: __m128, b: __m128) -> __m128 { into_first!(fcmpnle!, f32x4, a, b) }
    fn _mm_cmpnlt_ss(a: __m128, b: __m128) -> __m128 { into_first!(fcmpnlt!, f32x4, a, b) }
    fn _mm_cmpord_ss(a: __m128, b: __m128) -> __m128 { into_first!(fcmpord!, f32x4, a, b) }
    fn _mm_cmpunord_ss(a: __m128, b: __m128) -> __m128 { into_first!(fcmpunord!, f32x4, a, b) }

    fn _mm_comieq_ss(a: __m128, b: __m128) -> i32 { comi!(eq, a, b) }
    fn _mm_comige_ss(a: __m128, b: __m128) -> i32 { comi!(ge, a, b) }
    fn _mm_comigt_ss(a: __m128, b: __m128) -> i32 { comi!(gt, a, b) }
    fn _mm_comile_ss(a: __m128, b: __m128) -> i32 { comi!(le, a, b) }
    fn _mm_comilt_ss(a: __m128, b: __m128) -> i32 { comi!(lt, a, b) }
    fn _mm_comineq_ss(a: __m128, b: __m128) -> i32 { comi!(ne, a, b) }
    fn _mm_ucomieq_ss(a: __m128, b: __m128) -> i32 { comi!(eq, a, b) }
    fn _mm_ucomige_ss(a: __m128, b: __m128) -> i32 { comi!(ge, a, b) }
    fn _mm_ucomigt_ss(a: __m128, b: __m128) -> i32 { comi!(gt, a, b) }
    fn _mm_ucomile_ss(a: __m128, b: __m128) -> i32 { comi!(le, a, b) }
    fn _mm_ucomilt_ss(a: __m128, b: __m128) -> i32 { comi!(lt, a, b) }
    fn _mm_ucomineq_ss(a: __m128, b: __m128) -> i32 { comi!(ne, a, b) }
}

intrinsic! {
    #[notest()]
    fn _mm_extract_pi16<const IMM8: i32>(a: __m64) -> i32 {
        let a: i16x4 = a.into();
        a[(IMM8 & 0x3) as usize] as i32
    }

    #[notest()]
    fn _mm_insert_pi16<const IMM8: i32>(a: __m64, i: i32) -> __m64 {
        let mut a: i16x4 = a.into();
        a[(IMM8 & 0x3) as usize] = i as i16;
        a.into()
    }
}

intrinsic! {
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

    #[notest()]
    fn _mm_movemask_pi8(a: __m64) -> i32 {
        let a: i8x8 = a.into();
        a.simd_lt(i8x8::splat(0)).to_bitmask() as i32
    }

    fn _mm_movemask_ps(a: __m128) -> i32 {
        let a: i32x4 = a.into();
        a.simd_lt(i32x4::splat(0)).to_bitmask() as i32
    }

    #[notest()]
    fn _mm_mulhi_pu16(a: __m64, b: __m64) -> __m64 {
        let a: u16x4 = a.into();
        let b: u16x4 = b.into();
        let a: u32x4 = a.cast();
        let b: u32x4 = b.cast();
        let r: u16x4 = ((a * b) >> u32x4::splat(16)).cast();
        r.into()
    }

    #[notest()]
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
        _mm_set_ps(0., 0., 0., a)
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
    #[notest()]
    fn _mm_shuffle_pi16<const IMM8: i32>(a: __m64) -> __m64 {
        let a: i16x4 = a.into();
        shuffle4! { IMM8, a }.into()
    }

    fn _mm_shuffle_ps<const IMM8: i32>(a: __m128, b: __m128) -> __m128 {
        let a: f32x4 = a.into();
        let b: f32x4 = b.into();
        use core::simd::{Swizzle2, Which};
        struct Shuffle<const IMM8: i32>;
        impl<const IMM8: i32> Swizzle2<4, 4> for Shuffle<IMM8> {
            const INDEX: [Which; 4] = [
                Which::First((IMM8 as usize) & 0x3),
                Which::First((IMM8 as usize >> 2) & 0x3),
                Which::Second((IMM8 as usize >> 4) & 0x3),
                Which::Second((IMM8 as usize >> 6) & 0x3),
            ];
        }
        Shuffle::<IMM8>::swizzle2(a, b).into()
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
        f32x4::from_array([mem_addr.read_unaligned(), 0., 0., 0.]).into()
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

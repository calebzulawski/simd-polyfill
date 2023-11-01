use super::*;
use core::ops::*;

binary! {
    _mm_add_ps, Add::add, __m128 as f32x4;

    // _mm_and_ps
    // _mm_andnot_ps

    // _mm_avg_pu16
    // _mm_avg_pu8
}

binary! {
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
}

binary_one_element! {
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
}

use crate::vector;
use core::simd::prelude::*;

/// MMX instruction set
pub mod mmx;

vector! {
    pub struct __m64(u8x8) from u8x8, u16x4, u32x2, u64x1, i8x8, i16x4, i32x2, i64x1;
}

vector! {
    pub struct __m128i(u8x16) from u8x16, u16x8, u32x4, u64x2, i8x16, i16x8, i32x4, i64x2;
}

vector! {
    pub struct __m128(f32x4) from f32x4;
}

vector! {
    pub struct __m128d(f64x2) from f64x2;
}

vector! {
    pub struct __m256i(u8x32) from u8x32, u16x16, u32x8, u64x4, i8x32, i16x16, i32x8, i64x4;
}

vector! {
    pub struct __m256(f32x8) from f32x8;
}

vector! {
    pub struct __m256d(f64x4) from f64x4;
}

macro_rules! intrinsic {
    {
        $(fn $name:ident $args:tt $(-> $ret:ty)? { $($body:tt)* })*
    } => {
        $(
        #[doc = concat!("[reference ↗](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=", stringify!($name), ")")]
        #[inline]
        pub fn $name $args $(-> $ret)* { $($body)* }
        )*
    }
}

macro_rules! binary {
    {
        $(
            $name:ident, $func:expr, $ty1:ty as $inner1:ty, $ty2:ty as $inner2:ty, $ret:ty;
        )*
    } => {
        $(
        intrinsic! {
            fn $name(a: $ty1, b: $ty2) -> $ret {
                let a: $inner1 = a.into();
                let b: $inner2 = b.into();
                $func(a, b).into()
            }
        }
        )*
    };
    {
        $(
            $name:ident, $func:expr, $ty:ty as $inner:ty;
        )*
    } => {
        $(
        binary! { $name, $func, $ty as $inner, $ty as $inner, $ty; }
        )*
    };
    {
        $(
            $name:ident, macro $func:path, $ty1:ty as $inner1:ty, $ty2:ty as $inner2:ty, $ret:ty;
        )*
    } => {
        $(
        intrinsic! {
            fn $name(a: $ty1, b: $ty2) -> $ret {
                let a: $inner1 = a.into();
                let b: $inner2 = b.into();
                $func!(a, b).into()
            }
        }
        )*
    };
    {
        $(
            $name:ident, macro $func:path, $ty:ty as $inner:ty;
        )*
    } => {
        $(
        binary! { $name, macro $func, $ty as $inner, $ty as $inner, $ty; }
        )*
    }
}

macro_rules! andnot {
    { $a:expr, $b:expr } => {
        !($a & $b)
    }
}

macro_rules! cmpeq {
    { $a:expr, $b:expr } => {
        $a.simd_eq($b).to_int()
    }
}

macro_rules! cmpgt {
    { $a:expr, $b:expr } => {
        $a.simd_gt($b).to_int()
    }
}

macro_rules! packs2 {
    { $out:ty, $a:expr, $b:expr } => {
        simd_swizzle!($a, $b, [
            core::simd::Which::First(0),
            core::simd::Which::First(1),
            core::simd::Which::Second(0),
            core::simd::Which::Second(1),
        ])
        .simd_clamp(
            Simd::splat(<$out>::MIN as _),
            Simd::splat(<$out>::MAX as _),
        )
        .cast::<$out>()
    }
}

macro_rules! packs4 {
    { $out:ty, $a:expr, $b:expr } => {
        simd_swizzle!($a, $b, [
            core::simd::Which::First(0),
            core::simd::Which::First(1),
            core::simd::Which::First(2),
            core::simd::Which::First(3),
            core::simd::Which::Second(0),
            core::simd::Which::Second(1),
            core::simd::Which::Second(2),
            core::simd::Which::Second(3),
        ])
        .simd_clamp(
            Simd::splat(<$out>::MIN as _),
            Simd::splat(<$out>::MAX as _),
        )
        .cast::<$out>()
    }
}

macro_rules! unpackhi {
    { $a:expr, $b:expr } => {
        $a.interleave($b).1
    }
}

macro_rules! unpacklo {
    { $a:expr, $b:expr } => {
        $a.interleave($b).0
    }
}

pub(crate) use andnot;
pub(crate) use binary;
pub(crate) use cmpeq;
pub(crate) use cmpgt;
pub(crate) use intrinsic;
pub(crate) use packs2;
pub(crate) use packs4;
pub(crate) use unpackhi;
pub(crate) use unpacklo;

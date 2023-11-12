//! x86 and x86-64 intrinsics

use crate::vector;
use core::{
    ops::*,
    simd::{prelude::*, LaneCount, SimdElement, SupportedLaneCount},
};

#[cfg(test)]
pub(crate) mod test;

/// MMX instruction set
pub mod mmx;

/// SSE instruction set
pub mod sse;

/// SSE2 instruction set
pub mod sse2;

/// SSE3 instruction set
pub mod sse3;

/// SSSE3 instruction set
pub mod ssse3;

vector! {
    pub struct __m64(u32x2) from u8x8, u16x4, u32x2, u64x1, i8x8, i16x4, i32x2, i64x1;
}

vector! {
    pub struct __m128i(u32x4) from u8x16, u16x8, u32x4, u64x2, i8x16, i16x8, i32x4, i64x2;
}

vector! {
    pub struct __m128(f32x4) from f32x4, u32x4, i32x4;
}

vector! {
    pub struct __m128d(f64x2) from f64x2, u64x2, i64x2;
}

vector! {
    pub struct __m256i(u32x8) from u8x32, u16x16, u32x8, u64x4, i8x32, i16x16, i32x8, i64x4;
}

vector! {
    pub struct __m256(f32x8) from f32x8, u32x8, i32x8;
}

vector! {
    pub struct __m256d(f64x4) from f64x4, u64x4, i64x4;
}

macro_rules! intrinsic {
    {
        $(#[intrinsic = $name:ident] $func:item)*
    } => {
        $(
        #[doc = concat!("[reference ↗](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=", stringify!($name), ")")]
        #[inline]
        $func
        )*
    };
    {
        $(fn $name:ident<const $imm:ident: $immty:ty> $args:tt $(-> $ret:ty)? { $($body:tt)* })*
    } => {
        $(
        #[doc = concat!("[reference ↗](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=", stringify!($name), ")")]
        #[inline]
        pub fn $name <const $imm: $immty> $args $(-> $ret)* { $($body)* }
        )*
    };
    {
        $(
        $(#[notest $notest:tt])?
        $(#[homogenous $homogenous:tt])?
        fn $name:ident $args:tt $(-> $ret:ty)? { $($body:tt)* }
        )*
    } => {
        $(
        #[doc = concat!("[reference ↗](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=", stringify!($name), ")")]
        #[inline]
        pub fn $name $args $(-> $ret)* { $($body)* }

        #[cfg(test)]
        crate::x86::test::make_test! {
            $(#[notest $notest])?
            $(#[homogenous $homogenous])?
            $name $args $(-> $ret)?
        }
        )*
    };
    {
        $(unsafe fn $name:ident $args:tt $(-> $ret:ty)? { $($body:tt)* })*
    } => {
        $(
        #[doc = concat!("[reference ↗](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=", stringify!($name), ")")]
        #[inline]
        pub unsafe fn $name $args $(-> $ret)* { $($body)* }
        )*
    };
}

macro_rules! andnot {
    { $a:expr, $b:expr } => {
        (!$a) & $b
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

macro_rules! cmpge {
    { $a:expr, $b:expr } => {
        $a.simd_ge($b).to_int()
    }
}

macro_rules! cmplt {
    { $a:expr, $b:expr } => {
        $a.simd_lt($b).to_int()
    }
}

macro_rules! cmple {
    { $a:expr, $b:expr } => {
        $a.simd_le($b).to_int()
    }
}

macro_rules! cmpneq {
    { $a:expr, $b:expr } => {
        (!$a.simd_eq($b)).to_int()
    }
}

macro_rules! cmpngt {
    { $a:expr, $b:expr } => {
        (!$a.simd_gt($b)).to_int()
    }
}

macro_rules! cmpnge {
    { $a:expr, $b:expr } => {
        (!$a.simd_ge($b)).to_int()
    }
}

macro_rules! cmpnlt {
    { $a:expr, $b:expr } => {
        (!$a.simd_lt($b)).to_int()
    }
}

macro_rules! cmpnle {
    { $a:expr, $b:expr } => {
        (!$a.simd_le($b)).to_int()
    }
}

macro_rules! cmpord {
    { $a:expr, $b:expr } => {
        (!$a.is_nan() & !$b.is_nan()).to_int()
    }
}

macro_rules! cmpunord {
    { $a:expr, $b:expr } => {
        ($a.is_nan() | $b.is_nan()).to_int()
    }
}

macro_rules! fcmp {
    { $($fcmp:ident: $cmp:ident,)* } => {
        $(
        macro_rules! $fcmp {
            { $a:expr, $b:expr } => { SimdFloat::from_bits($cmp!($a, $b).cast()) }
        }

        pub(crate) use $fcmp;
        )*
    }
}

fcmp! {
    fcmpeq: cmpeq,
    fcmpgt: cmpgt,
    fcmpge: cmpge,
    fcmplt: cmplt,
    fcmple: cmple,
    fcmpord: cmpord,
    fcmpneq: cmpneq,
    fcmpngt: cmpngt,
    fcmpnge: cmpnge,
    fcmpnlt: cmpnlt,
    fcmpnle: cmpnle,
    fcmpunord: cmpunord,
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

macro_rules! packs8 {
    { $out:ty, $a:expr, $b:expr } => {
        simd_swizzle!($a, $b, [
            core::simd::Which::First(0),
            core::simd::Which::First(1),
            core::simd::Which::First(2),
            core::simd::Which::First(3),
            core::simd::Which::First(4),
            core::simd::Which::First(5),
            core::simd::Which::First(6),
            core::simd::Which::First(7),
            core::simd::Which::Second(0),
            core::simd::Which::Second(1),
            core::simd::Which::Second(2),
            core::simd::Which::Second(3),
            core::simd::Which::Second(4),
            core::simd::Which::Second(5),
            core::simd::Which::Second(6),
            core::simd::Which::Second(7),
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

macro_rules! float_max {
    { $a:expr, $b:expr } => {
        $a.simd_gt($b).select($a, $b)
    }
}

macro_rules! float_min {
    { $a:expr, $b:expr } => {
        $a.simd_lt($b).select($a, $b)
    }
}

macro_rules! hadd {
    { $a:expr, $b:expr } => {
        {
            let (first, second) = $a.deinterleave($b);
            first + second
        }
    }
}

macro_rules! hsub {
    { $a:expr, $b:expr } => {
        {
            let (first, second) = $a.deinterleave($b);
            first - second
        }
    }
}

macro_rules! hadds {
    { $a:expr, $b:expr } => {
        {
            let (first, second) = $a.deinterleave($b);
            first.saturating_add(second)
        }
    }
}

macro_rules! hsubs {
    { $a:expr, $b:expr } => {
        {
            let (first, second) = $a.deinterleave($b);
            first.saturating_sub(second)
        }
    }
}

macro_rules! sign {
    { $a:expr, $b:expr } => {
        $b.simd_eq(Simd::splat(0)).select(
            Simd::splat(0),
            $b.simd_lt(Simd::splat(0)).select(-$a, $a)
        )
    }
}

macro_rules! shuffle4 {
    { $IMM8:expr, $a:expr } => {
        {
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
            Shuffle::<$IMM8>::swizzle($a)
        }
    }
}

pub(crate) fn mulhrs<const N: usize>(a: Simd<i16, N>, b: Simd<i16, N>) -> Simd<i16, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    let a: Simd<i32, N> = a.cast();
    let b: Simd<i32, N> = b.cast();
    ((((a * b) >> Simd::splat(14)) + Simd::splat(1)) >> Simd::splat(1)).cast()
}

pub(crate) fn addsub<T, const N: usize>(a: Simd<T, N>, b: Simd<T, N>) -> Simd<T, N>
where
    T: SimdElement,
    LaneCount<N>: SupportedLaneCount,
    Simd<T, N>: Add<Output = Simd<T, N>> + Sub<Output = Simd<T, N>>,
{
    const fn alternate<const N: usize>() -> [bool; N] {
        let mut mask = [false; N];
        let mut i = 0;
        while i < N {
            mask[i] = (i & 1) == 0;
            i += 1;
        }
        mask
    }

    Mask::from_array(alternate::<N>()).select(a - b, a + b)
}

pub(crate) fn pavgb<const N: usize>(a: Simd<u8, N>, b: Simd<u8, N>) -> Simd<u8, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    let a: Simd<u16, N> = a.cast();
    let b: Simd<u16, N> = b.cast();
    let one = Simd::<u16, N>::splat(1);
    let r = (a + b + one) >> one;
    r.cast()
}

pub(crate) fn pavgw<const N: usize>(a: Simd<u16, N>, b: Simd<u16, N>) -> Simd<u16, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    let a: Simd<u32, N> = a.cast();
    let b: Simd<u32, N> = b.cast();
    let one = Simd::<u32, N>::splat(1);
    let r = (a + b + one) >> one;
    r.cast()
}

macro_rules! shift_logical {
    { $func:path, $ty:ty, $a:expr, $b:expr } => {
        {
            let a: $ty = $a.into();
            let count = $b as u64;
            if count >= (128 / <$ty>::LANES) as u64 {
                <$ty>::splat(0).into()
            } else {
                $func(a, <$ty>::splat(count as _)).into()
            }
        }
    }
}

macro_rules! shift_right {
    { $ty:ty, $a:expr, $b:expr } => {
        {
            let a: $ty = $a.into();
            let count = $b as u64;
            (a >> <$ty>::splat(count.clamp(0, (128 / <$ty>::LANES - 1) as _) as _)).into()
        }
    }
}

macro_rules! sxl {
    { $func:path, $ty:ty, $ty2:ty, $a:expr, $b:expr } => {
        {
            let b: $ty2 = $b.into();
            shift_logical!($func, $ty, $a, b[0])
        }
    }
}

macro_rules! sra {
    { $ty:ty, $ty2:ty, $a:expr, $b:expr } => {
        {
            let b: $ty2 = $b.into();
            shift_right!($ty, $a, b[0])
        }
    }
}

pub(crate) use andnot;
pub(crate) use cmpeq;
pub(crate) use cmpge;
pub(crate) use cmpgt;
pub(crate) use cmple;
pub(crate) use cmplt;
pub(crate) use cmpneq;
pub(crate) use cmpnge;
pub(crate) use cmpngt;
pub(crate) use cmpnle;
pub(crate) use cmpnlt;
pub(crate) use cmpord;
pub(crate) use cmpunord;
pub(crate) use float_max;
pub(crate) use float_min;
pub(crate) use hadd;
pub(crate) use hadds;
pub(crate) use hsub;
pub(crate) use hsubs;
pub(crate) use intrinsic;
pub(crate) use packs2;
pub(crate) use packs4;
pub(crate) use packs8;
pub(crate) use shift_logical;
pub(crate) use shift_right;
pub(crate) use shuffle4;
pub(crate) use sign;
pub(crate) use sra;
pub(crate) use sxl;
pub(crate) use unpackhi;
pub(crate) use unpacklo;

//! SIMD polyfill for Rust.
//!
//! Provides implementations of SIMD instruction sets that work on all other SIMD instruction sets.

#![feature(portable_simd)]
#![cfg_attr(test, feature(stdsimd))]
#![allow(
    non_camel_case_types,
    non_snake_case,
    clippy::missing_safety_doc,
    clippy::too_many_arguments,
    clippy::useless_transmute
)]
#![cfg_attr(not(test), no_std)]

pub mod x86;

#[cfg(test)]
pub(crate) mod test;

macro_rules! vector {
    { $(#[doc = $doc:literal])* pub struct $name:ident($inner:ty) from $($from:ty),*; } => {
        $(#[doc = $doc])*
        #[derive(Copy, Clone, Debug, PartialEq)]
        #[repr(transparent)]
        pub struct $name($inner);

        $(
        impl From<$from> for $name {
            fn from(v: $from) -> Self {
                unsafe { core::mem::transmute(v) }
            }
        }

        impl From<$name> for $from {
            fn from(v: $name) -> Self {
                unsafe { core::mem::transmute(v) }
            }
        }
        )*
    }
}

macro_rules! into {
    { $f:path, $into:ty, $a:expr } => {
        {
            #![allow(unused_mut)]
            let mut a: $into = $a.into();
            $f(a).into()
        }
    };
    { $f:ident!, $into:ty, $a:expr } => {
        {
            #![allow(unused_mut)]
            let mut a: $into = $a.into();
            $f!(a).into()
        }
    };
    { $f:path, $into:ty, $a:expr, $b:expr } => {
        {
            #![allow(unused_mut)]
            let mut a: $into = $a.into();
            let mut b: $into = $b.into();
            $f(a, b).into()
        }
    };
    { $f:ident!, $into:ty, $a:expr, $b:expr } => {
        {
            #![allow(unused_mut)]
            let mut a: $into = $a.into();
            let mut b: $into = $b.into();
            $f!(a, b).into()
        }
    }
}

macro_rules! into_first {
    { $f:path, $into:ty, $($args:expr),+ } => {
        {
            macro_rules! inner {
                { $a:expr } => {
                    {
                        let out: $into = $f($a);
                        $a[0] = out[0];
                        $a
                    }
                };
                { $a:expr, $b:expr } => {
                    {
                        let out: $into = $f($a, $b);
                        $a[0] = out[0];
                        $a
                    }
                }
            }
            into!(inner!, $into, $($args),*)
        }
    };
    { $f:ident!, $into:ty, $($args:expr),+ } => {
        {
            macro_rules! inner {
                { $a:expr } => {
                    {
                        let out: $into = $f!($a);
                        $a[0] = out[0];
                        $a
                    }
                };
                { $a:expr, $b:expr } => {
                    {
                        let out: $into = $f!($a, $b);
                        $a[0] = out[0];
                        $a
                    }
                }
            }
            into!(inner!, $into, $($args),*)
        }
    };
}

pub(crate) use into;
pub(crate) use into_first;
pub(crate) use vector;

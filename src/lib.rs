#![feature(portable_simd)]
#![allow(non_camel_case_types)]

pub mod x86;

macro_rules! vector {
    { pub struct $name:ident($inner:ty) from $($from:ty),*; } => {
        #[derive(Copy, Clone, Debug)]
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

pub(crate) use vector;

use super::*;
use crate::test::{strategy_type, Equalish};
strategy_type! { __m64, i32, 2 }
strategy_type! { __m128, f32, 4 }
strategy_type! { __m128d, f64, 2 }
strategy_type! { __m128i, i32, 4 }

impl Equalish for __m64 {}
impl Equalish for __m128i {}
impl Equalish for __m128 {
    fn equalish(self, other: Self) -> bool {
        let a: f32x4 = self.into();
        let b: f32x4 = other.into();
        (a.simd_eq(b) | (a.is_nan() & b.is_nan())).all()
    }
}
impl Equalish for __m128d {
    fn equalish(self, other: Self) -> bool {
        let a: f64x2 = self.into();
        let b: f64x2 = other.into();
        (a.simd_eq(b) | (a.is_nan() & b.is_nan())).all()
    }
}

macro_rules! supported_target {
    {} => {
        match module_path!().split("::").nth(2).unwrap() {
            "sse" => is_x86_feature_detected!("sse"),
            "sse2" => is_x86_feature_detected!("sse2"),
            "sse3" => is_x86_feature_detected!("sse3"),
            "ssse3" => is_x86_feature_detected!("ssse3"),
            "sse41" => is_x86_feature_detected!("sse4.1"),
            "sse42" => is_x86_feature_detected!("sse4.2"),
            x => panic!("bad target {}", x),
        }
    }
}

macro_rules! make_test {
    { #[notest()] $name:ident $args:tt $(-> $ret:ty)? } => {};
    { #[notest()] $name:ident <const $imm:ident: $immty:ty> $args:tt $(-> $ret:ty)? } => {};
    { @imm ($testval:expr) $name:ident <const $imm:ident: $immty:ty> ($($var:ident: $ty:ty),+) -> $ret:ty } => {
        paste::paste! {
            #[test]
            fn [<test_ $testval>]() {
                if !crate::x86::test::supported_target!() {
                    return
                }

                use crate::test::DefaultStrategy;
                let mut runner = proptest::test_runner::TestRunner::default();

                runner.run(&<($($ty,)*)>::default_strategy(), |($($var,)*): ($($ty,)*)| {
                    let result = super::$name::<$testval>($($var),*);
                    let result_intrin: $ret = unsafe {
                        core::mem::transmute(
                            core::arch::x86_64::$name(
                                $(core::mem::transmute($var)),*, $testval,
                            ),
                        )
                    };
                    crate::test::assert_equalish!(result, result_intrin);
                    Ok(())
                }).unwrap();
            }
        }
    };
    { #[testvals($($testval:expr),+)] $name:ident <const $imm:ident: $immty:ty> $args:tt -> $ret:ty } => {
        mod $name {
            use super::*;
            $( crate::x86::test::make_test! { @imm ($testval) $name <const $imm: $immty> $args -> $ret } )*
        }
    };
    { $name:ident <const IMM8: $immty:ty> $args:tt -> $ret:ty } => {
        crate::x86::test::make_test! {
            #[testvals(0x00, 0x01, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0xf, 0x10, 0xf0, 0xff, 0x25, 0x28, 0x8a, 0x1c, 0xb6, 0x61, 0xb0, 0x7c)]
            $name <const IMM8: $immty> $args -> $ret
        }
    };
    { $name:ident ($($var:ident: $ty:ty),+) -> $ret:ty } => {
        mod $name {
            use super::*;

            #[test]
            fn test() {
                if !crate::x86::test::supported_target!() {
                    return
                }

                use crate::test::DefaultStrategy;
                let mut runner = proptest::test_runner::TestRunner::default();
                runner.run(&<($($ty,)*)>::default_strategy(), |($($var,)*): ($($ty,)*)| {
                    let result = super::$name($($var),*);
                    let result_intrin: $ret = unsafe {
                        core::mem::transmute(
                            core::arch::x86_64::$name(
                                $(core::mem::transmute($var)),*
                            ),
                        )
                    };
                    crate::test::assert_equalish!(result, result_intrin);
                    Ok(())
                }).unwrap()
            }
        }
    };
    { #[homogenous()] $name:ident ($($var:ident: $ty:ty),+) -> $ret:ty } => {
        mod $name {
            type Input = [crate::test::first!($($ty)*); crate::test::count!($($ty)*)];

            #[test]
            fn test() {
                if !crate::x86::test::supported_target!() {
                    return
                }

                use crate::test::DefaultStrategy;
                let mut runner = proptest::test_runner::TestRunner::default();
                runner.run(&<Input>::default_strategy(), |[$($var,)*]: Input| {
                    let result = super::$name($($var),*);
                    let result_intrin = unsafe {
                        core::mem::transmute(
                            core::arch::x86_64::$name(
                                $(core::mem::transmute($var)),*
                            ),
                        )
                    };
                    crate::test::assert_equalish!(result, result_intrin);
                    Ok(())
                }).unwrap()
            }
        }
    };
    { $name:ident() -> $ret:ty } => {
        mod $name {
            #[test]
            fn test() {
                assert_eq!(super::$name(), unsafe { core::mem::transmute(core::arch::x86_64::$name()) })
            }
        }
    };
    { $name:ident() } => {}
}

pub(crate) use make_test;
pub(crate) use supported_target;

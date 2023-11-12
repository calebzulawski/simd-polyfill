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
        let a: i32x4 = self.into();
        let b: i32x4 = other.into();
        a == b
    }
}
impl Equalish for __m128d {
    fn equalish(self, other: Self) -> bool {
        let a: i64x2 = self.into();
        let b: i64x2 = other.into();
        a == b
    }
}

macro_rules! supported_target {
    {} => {
        match module_path!().split("::").nth(2).unwrap() {
            "sse" => is_x86_feature_detected!("sse"),
            "sse2" => is_x86_feature_detected!("sse2"),
            "sse3" => is_x86_feature_detected!("sse3"),
            "ssse3" => is_x86_feature_detected!("ssse3"),
            "sse4.1" => is_x86_feature_detected!("sse4.1"),
            "sse4.2" => is_x86_feature_detected!("sse4.2"),
            x => panic!("bad target {}", x),
        }
    }
}

macro_rules! make_test {
    { #[notest()] $name:ident $args:tt $(-> $ret:ty)? } => {};
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

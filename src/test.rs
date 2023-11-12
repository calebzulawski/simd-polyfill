use proptest::strategy;

macro_rules! count {
    {} => { 0usize };
    { $x:tt $($xs:tt)* } => { 1usize + crate::test::count!($($xs)*) };
}

macro_rules! first {
    { $first:tt $($rest:tt)* } => { $first };
}

pub(crate) use count;
pub(crate) use first;

pub trait Equalish: Copy + PartialEq + core::fmt::Debug {
    fn equalish(self, other: Self) -> bool {
        self == other
    }
}

impl Equalish for i32 {}

pub trait DefaultStrategy {
    type Strategy: strategy::Strategy<Value = Self>;
    fn default_strategy() -> Self::Strategy;
}

impl<T, const N: usize> DefaultStrategy for [T; N]
where
    T: DefaultStrategy + core::fmt::Debug,
{
    type Strategy = proptest::array::UniformArrayStrategy<T::Strategy, [T; N]>;
    fn default_strategy() -> Self::Strategy {
        Self::Strategy::new(T::default_strategy())
    }
}

macro_rules! assert_equalish {
    { $a:expr, $b:expr } => {
        assert!(crate::test::Equalish::equalish($a, $b), "left = {:?}\nright = {:?}", $a, $b)
    }
}

macro_rules! strategy_num {
    { $($num:ident),* } => {
        $(
        impl DefaultStrategy for $num {
            type Strategy = proptest::num::$num::Any;
            fn default_strategy() -> Self::Strategy { proptest::num::$num::ANY }
        }
        )*
    }
}

strategy_num! { i8, i16, i32, i64, isize, u8, u16, u32, u64, usize, f32, f64 }

macro_rules! strategy_type {
    { $ty:ident, $scalar:tt, $num:literal } => {
        impl crate::test::DefaultStrategy for $ty {
            type Strategy = proptest::strategy::MapInto<
                proptest::strategy::MapInto<
                    proptest::array::UniformArrayStrategy<proptest::num::$scalar::Any, [$scalar; $num]>,
                    Simd<$scalar, $num>,
                >,
                $ty,
            >;

            fn default_strategy() -> Self::Strategy {
                use proptest::strategy::Strategy;
                proptest::array::UniformArrayStrategy::new(proptest::num::$scalar::ANY)
                    .prop_map_into()
                    .prop_map_into()
            }
        }
    }
}

macro_rules! impl_tuple {
    { $($name:ident),* } => {
        impl<$($name),*> DefaultStrategy for ($($name,)*)
        where
        $(
            $name: DefaultStrategy,
        )*
        {
            type Strategy = ($($name::Strategy,)*);
            fn default_strategy() -> Self::Strategy {
                ($($name::default_strategy(),)*)
            }
        }
    }
}

impl_tuple! { A }
impl_tuple! { A, B }
impl_tuple! { A, B, C }
impl_tuple! { A, B, C, D }
impl_tuple! { A, B, C, D, E, F, G, H }

pub(crate) use assert_equalish;
pub(crate) use strategy_type;

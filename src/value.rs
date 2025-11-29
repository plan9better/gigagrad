use num_traits::{FromPrimitive, ToPrimitive};
use std::hash::Hash;
use std::ops::Add;

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub enum Oper {
    Add,
    Sub,
    Div,
    Mul,
    Exp,
    Pow,
    Leaf,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Value<T> {
    pub value: T,
    pub grad: Option<T>,
    pub parent: Oper,
    // this should be an unordered set of arbitrary length e.g. a HashMap
    // but floats don't implement Eq (NaN != NaN) and Hash so a list for now
    pub children: Vec<Value<T>>,
}

impl<T> Value<T>
where
    T: ToPrimitive + FromPrimitive + Add<Output = T> + Copy,
{
    pub fn new(value: T) -> Self {
        Value {
            value,
            grad: None,
            parent: Oper::Leaf,
            children: Vec::new(),
        }
    }

    pub fn backward(&mut self) {
        if self.grad.is_none() {
            self.grad = Some(T::from_f64(1.0).unwrap());
        }

        let parent_grad = self.grad.unwrap();

        match self.parent {
            Oper::Add => {
                for child in self.children.iter_mut() {
                    let current_grad = child.grad.unwrap_or(T::from_f64(0.0).unwrap());
                    child.grad = Some(current_grad + parent_grad);

                    child.backward();
                }
            }
            Oper::Sub => unimplemented!(),
            Oper::Div => unimplemented!(),
            Oper::Mul => unimplemented!(),
            Oper::Exp => unimplemented!(),
            Oper::Pow => unimplemented!(),
            Oper::Leaf => {}
        }
    }
}

// Adding two value types
impl<T> Add for Value<T>
where
    T: Add<Output = T> + Copy,
{
    type Output = Value<T>;
    fn add(self, other: Self) -> Self {
        Value {
            value: self.value + other.value,
            parent: Oper::Add,
            grad: None,
            children: Vec::from([self.clone(), other.clone()]),
        }
    }
}

// Adding value to anything that implements add e.g. int
impl<T> Add<T> for Value<T>
where
    T: Add<Output = T> + Copy + FromPrimitive + ToPrimitive,
{
    type Output = Value<T>;
    fn add(self, other: T) -> Self {
        return Value {
            value: self.value + other,
            parent: Oper::Add,
            grad: None,
            children: Vec::from([self, Value::new(other)]),
        };
    }
}

// you can't define a generic operation for a rhs (would
// require you to to smth like impl Add<Value<*>> or whatever)
// You can add them if you specify the type explicitly so
// this is a macro to generate the addidion rhs operation
// for the common numeric types.
macro_rules! impl_add_for_numeric_types {
    ($($t:ty),*) => {
        $(
            impl Add<Value<$t>> for $t {
                type Output = Value<$t>;
                fn add(self, val: Value<$t>) -> Value<$t> {
                    Value {
                        value: self + val.value,
                        parent: Oper::Add,
                        grad: None,
                        children: Vec::from([val, Value::new(self)]),
                    }
                }
            }
        )*
    };
}
impl_add_for_numeric_types! {
    f32, f64,
    i8, i16, i32, i64, i128,
    u8, u16, u32, u64, u128,
    isize, usize
}

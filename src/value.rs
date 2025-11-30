use std::cell::RefCell;
use std::fmt;
use std::ops::{Add, Mul};
use std::rc::Rc;

#[derive(Debug, Clone, Copy, PartialEq)]
enum Oper {
    Add,
    Mul,
    Leaf,
}

struct ValueData {
    data: f64,
    grad: f64,
    _prev: Vec<Value>,
    _op: Oper,
}

#[derive(Clone)]
pub struct Value(Rc<RefCell<ValueData>>);

impl Value {
    pub fn new(data: f64) -> Value {
        Value(Rc::new(RefCell::new(ValueData {
            data,
            grad: 0.0,
            _prev: vec![],
            _op: Oper::Leaf,
        })))
    }

    pub fn data(&self) -> f64 {
        self.0.borrow().data
    }

    pub fn grad(&self) -> f64 {
        self.0.borrow().grad
    }
}

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Value(data={:.4}, grad={:.4})", self.data(), self.grad())
    }
}

//  &Value + &Value
impl<'a, 'b> Add<&'b Value> for &'a Value {
    type Output = Value;

    fn add(self, other: &'b Value) -> Value {
        Value(Rc::new(RefCell::new(ValueData {
            data: self.data() + other.data(),
            grad: 0.0,
            _prev: vec![self.clone(), other.clone()],
            _op: Oper::Add,
        })))
    }
}

// Value + Value
impl Add for Value {
    type Output = Value;
    fn add(self, other: Value) -> Value {
        &self + &other
    }
}

// &Value + Value
impl Add<Value> for &Value {
    type Output = Value;
    fn add(self, other: Value) -> Value {
        self + &other
    }
}

// Value + &Value
impl Add<&Value> for Value {
    type Output = Value;
    fn add(self, other: &Value) -> Value {
        &self + other
    }
}

// &Value * &Value
impl<'a, 'b> Mul<&'b Value> for &'a Value {
    type Output = Value;

    fn mul(self, other: &'b Value) -> Value {
        Value(Rc::new(RefCell::new(ValueData {
            data: self.data() * other.data(),
            grad: 0.0,
            _prev: vec![self.clone(), other.clone()],
            _op: Oper::Mul,
        })))
    }
}

// Value * Value
impl Mul for Value {
    type Output = Value;
    fn mul(self, other: Value) -> Value {
        &self * &other
    }
}

// &Value * Value
impl Mul<Value> for &Value {
    type Output = Value;
    fn mul(self, other: Value) -> Value {
        self * &other
    }
}

// Value * &Value
impl Mul<&Value> for Value {
    type Output = Value;
    fn mul(self, other: &Value) -> Value {
        &self * other
    }
}

impl Add<f64> for Value {
    type Output = Value;
    fn add(self, other: f64) -> Value {
        self + Value::new(other)
    }
}

impl Add<Value> for f64 {
    type Output = Value;
    fn add(self, other: Value) -> Value {
        Value::new(self) + other
    }
}

impl Mul<f64> for Value {
    type Output = Value;
    fn mul(self, other: f64) -> Value {
        self * Value::new(other)
    }
}

impl Mul<Value> for f64 {
    type Output = Value;
    fn mul(self, other: Value) -> Value {
        Value::new(self) * other
    }
}

use std::cell::RefCell;
use std::collections::HashSet;
use std::f64::consts;
use std::fmt;
use std::ops::{Add, Mul};
use std::rc::Rc;

use num_traits::Float;

#[derive(Debug, Clone, Copy, PartialEq)]
enum Oper {
    Add,
    Mul,
    Leaf,
    Tanh,
}

struct ValueData {
    data: f64,
    grad: Option<f64>,
    _prev: Vec<Value>,
    _op: Oper,
}

#[derive(Clone)]
pub struct Value(Rc<RefCell<ValueData>>);

impl Value {
    pub fn new(data: f64) -> Value {
        Value(Rc::new(RefCell::new(ValueData {
            data,
            grad: None,
            _prev: vec![],
            _op: Oper::Leaf,
        })))
    }

    pub fn backprop(&self) -> () {
        let mut topo = vec![];
        let mut visited = HashSet::new();
        self.build_topo(&mut visited, &mut topo);

        {
            // borrow and return
            let mut root = self.0.borrow_mut();
            if root.grad.is_none() {
                root.grad = Some(1.0);
            }
        }

        for node in topo.iter().rev() {
            let data = node.0.borrow_mut();
            match data._op {
                Oper::Add => {
                    for v in data._prev.iter() {
                        let mut child = v.0.borrow_mut();
                        child.grad = match child.grad {
                            None => data.grad,
                            Some(x) => Some(x + data.grad.unwrap()),
                        }
                    }
                }
                Oper::Mul => {
                    let a_rc = &data._prev[0];
                    let b_rc = &data._prev[1];

                    // .data() borrows quickly and returns
                    // copy so we just do it before the long
                    // borrow (mutable) since a value can
                    // only be borrowed once
                    let a_data = a_rc.data();
                    let b_data = b_rc.data();
                    let mut a = a_rc.0.borrow_mut();
                    let mut b = b_rc.0.borrow_mut();
                    a.grad = match a.grad {
                        None => Some(b_data * data.grad.unwrap()),
                        Some(x) => Some(x + b_data),
                    };
                    b.grad = match b.grad {
                        None => Some(a_data * data.grad.unwrap()),
                        Some(x) => Some((x + a_data) * data.grad.unwrap()),
                    };
                }
                Oper::Tanh => {
                    // data is already tanh(x) since
                    // we are now in the output and
                    // calculating derivatives of
                    // inputs so just ^2
                    let tanh: f64 = data.data.powi(2);
                    if tanh == 1 {
                        eprintln!("WARN[gigagrad]: While calculating gradient of {:?} the output value is 1 therefore the gradient will be 0.")
                    }
                    let d = 1.0 - tanh;
                    let mut child = data._prev[0].0.borrow_mut();
                    child.grad = match child.grad {
                        None => Some(d * data.grad.unwrap()),
                        Some(x) => Some((d * data.grad.unwrap()) + x),
                    };
                }
                Oper::Leaf => {}
            };
        }
    }

    pub fn data(&self) -> f64 {
        self.0.borrow().data
    }

    pub fn grad(&self) -> Option<f64> {
        self.0.borrow().grad
    }

    fn build_topo(&self, visited: &mut HashSet<usize>, topo: &mut Vec<Value>) {
        let id = Rc::as_ptr(&self.0) as usize;

        if !visited.contains(&id) {
            visited.insert(id);
            for child in &self.0.borrow()._prev {
                child.build_topo(visited, topo);
            }
            topo.push(self.clone());
        }
    }
    pub fn tanh(&self) -> Self {
        let data = self.data();
        Value(Rc::new(RefCell::new(ValueData {
            data: data.tanh(),
            grad: None,
            _prev: vec![self.clone()],
            _op: Oper::Tanh,
        })))
    }
}

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.grad().is_none() {
            write!(
                f,
                "Value(data={:.4}, grad=(Not set), Op={:?})",
                self.data(),
                self._op
            )
        } else {
            write!(
                f,
                "Value(data={:.4}, grad={:.4}), Op={:?}",
                self.data(),
                self.grad().unwrap()
            )
        }
    }
}

//  &Value + &Value
// unrelated scopes because the Rc will keep
// heap data alive for us so we don't have to
// worry about wether some value will outlive
// what it depends on.
impl<'a, 'b> Add<&'b Value> for &'a Value {
    type Output = Value;

    fn add(self, other: &'b Value) -> Value {
        Value(Rc::new(RefCell::new(ValueData {
            data: self.data() + other.data(),
            grad: None,
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
            grad: None,
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

impl Mul<&Value> for &f64 {
    type Output = Value;
    fn mul(self, other: &Value) -> Value {
        return Value::new(other.data() + self);
    }
}

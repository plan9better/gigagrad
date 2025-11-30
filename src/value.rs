use std::cell::RefCell;
use std::collections::HashSet;
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::rc::Rc;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Oper {
    Add,
    Sub,
    Mul,
    Pow(f64),
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

    pub fn pow(&self, exponent: f64) -> Value {
        Value(Rc::new(RefCell::new(ValueData {
            data: self.data().powf(exponent),
            grad: None,
            _prev: vec![self.clone()],
            _op: Oper::Pow(exponent),
        })))
    }

    pub fn update(&self, learning_rate: f64) -> () {
        let mut param = self.0.borrow_mut();
        param.data += -learning_rate * param.grad.unwrap()
    }

    pub fn backprop(&self) -> () {
        let mut topo = vec![];
        let mut visited = HashSet::new();
        self.build_topo(&mut visited, &mut topo);

        {
            let mut root = self.0.borrow_mut();
            if root.grad.is_none() {
                root.grad = Some(1.0);
            }
        }

        for node in topo.iter().rev() {
            let data = node.0.borrow_mut();
            let grad = data.grad.unwrap();

            match data._op {
                Oper::Add => {
                    for v in data._prev.iter() {
                        let mut child = v.0.borrow_mut();
                        child.grad = Some(child.grad.unwrap_or(0.0) + grad);
                    }
                }
                Oper::Sub => {
                    let a_rc = &data._prev[0];
                    let b_rc = &data._prev[1];

                    {
                        let mut a = a_rc.0.borrow_mut();
                        a.grad = Some(a.grad.unwrap_or(0.0) + grad);
                    }
                    {
                        let mut b = b_rc.0.borrow_mut();
                        b.grad = Some(b.grad.unwrap_or(0.0) - grad);
                    }
                }
                Oper::Mul => {
                    let a_rc = &data._prev[0];
                    let b_rc = &data._prev[1];
                    let a_data = a_rc.data();
                    let b_data = b_rc.data();

                    {
                        let mut a = a_rc.0.borrow_mut();
                        a.grad = Some(a.grad.unwrap_or(0.0) + b_data * grad);
                    }
                    {
                        let mut b = b_rc.0.borrow_mut();
                        b.grad = Some(b.grad.unwrap_or(0.0) + a_data * grad);
                    }
                }
                Oper::Pow(exp) => {
                    let child = &data._prev[0];
                    let base_data = child.data();

                    let derivative = exp * base_data.powf(exp - 1.0);

                    let mut c = child.0.borrow_mut();
                    c.grad = Some(c.grad.unwrap_or(0.0) + derivative * grad);
                }
                Oper::Tanh => {
                    let tanh = data.data; // tanh(x)
                    let d = 1.0 - tanh.powi(2); // derivative is 1 - tanh^2(x)

                    let mut child = data._prev[0].0.borrow_mut();
                    child.grad = Some(child.grad.unwrap_or(0.0) + d * grad);
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

    pub fn op(&self) -> Oper {
        self.0.borrow()._op
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
        if let Some(grad) = self.grad() {
            write!(
                f,
                "Value(data={:.4}, grad={:.4}, Op={:?})",
                self.data(),
                grad,
                self.op()
            )
        } else {
            write!(
                f,
                "Value(data={:.4}, grad=None, Op={:?})",
                self.data(),
                self.op()
            )
        }
    }
}

// Add
impl Add for Value {
    type Output = Value;
    fn add(self, other: Value) -> Value {
        &self + &other
    }
}
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

// Mul
impl Mul for Value {
    type Output = Value;
    fn mul(self, other: Value) -> Value {
        &self * &other
    }
}
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
        let tmp = Value::new(*self);
        return &tmp * other;
    }
}

// Sub
impl Sub for Value {
    type Output = Value;
    fn sub(self, other: Value) -> Value {
        &self - &other
    }
}
impl<'a, 'b> Sub<&'b Value> for &'a Value {
    type Output = Value;
    fn sub(self, other: &'b Value) -> Value {
        Value(Rc::new(RefCell::new(ValueData {
            data: self.data() - other.data(),
            grad: None,
            _prev: vec![self.clone(), other.clone()],
            _op: Oper::Sub,
        })))
    }
}
impl Sub<f64> for Value {
    type Output = Value;
    fn sub(self, other: f64) -> Value {
        self - Value::new(other)
    }
}
impl Sub<Value> for f64 {
    type Output = Value;
    fn sub(self, other: Value) -> Value {
        Value::new(self) - other
    }
}

// Div
impl Div for Value {
    type Output = Value;
    fn div(self, other: Value) -> Value {
        self * other.pow(-1.0)
    }
}
impl Div<f64> for Value {
    type Output = Value;
    fn div(self, other: f64) -> Value {
        self * Value::new(other).pow(-1.0)
    }
}
impl Div<Value> for f64 {
    type Output = Value;
    fn div(self, other: Value) -> Value {
        Value::new(self) * other.pow(-1.0)
    }
}

impl Neg for Value {
    type Output = Value;
    fn neg(self) -> Value {
        self * -1.0
    }
}

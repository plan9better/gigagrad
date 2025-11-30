mod value;
use value::Value;

fn main() {
    let a = Value::new(2.0);
    let b = Value::new(3.0);
    let c = &a * &b;
    let d = &c * &a;
    println!("Found c {:?}", c);
    println!("Found d {:?}", d);
    println!("Doing backprop on d");
    d.backprop();
    println!("a: {:?}", a);
    println!("b: {:?}", b);
    println!("c: {:?}", c);
    println!("d: {:?}", d);
}

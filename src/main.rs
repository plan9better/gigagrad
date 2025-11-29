mod value;

use value::Value;

fn main() {
    let a = Value::new(3.0);
    let b = Value::new(5.0);

    println!("{:?}", 3.0 + a.clone());
    println!("{:?}", b.clone() + a.clone());
    println!("{:?}", b == a);
    println!("b: {:?}", b);
    println!("a: {:?}", a);
    let c = b + a;
    println!("c: {:?}", c);

    println!("Hello, world!");
}

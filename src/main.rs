mod neuron;
mod value;
use neuron::MLP;
use value::Value;

fn main() {
    let mlp = MLP::new(3, vec![4, 4, 1]);
    let inputs: Vec<Value> = vec![Value::new(0.1), Value::new(0.7), Value::new(0.3)];
    println!("{:?}", mlp.forward(inputs));
}

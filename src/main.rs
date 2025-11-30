mod neuron;
mod value;
use neuron::MLP;
use std::{thread, time};
use value::Value;

fn main() {
    let inputs = vec![
        vec![Value::new(2.0), Value::new(3.0), Value::new(-1.0)],
        vec![Value::new(3.0), Value::new(-1.0), Value::new(0.5)],
        vec![Value::new(0.5), Value::new(1.0), Value::new(1.0)],
        vec![Value::new(1.0), Value::new(1.0), Value::new(-1.0)],
    ];
    let targets = vec![
        Value::new(1.0),
        Value::new(-1.0),
        Value::new(-1.0),
        Value::new(1.0),
    ];

    let mlp = MLP::new(3, vec![4, 4, 1]);
    loop {
        println!("MLP: {:?}", mlp);
        let mut outputs = Vec::with_capacity(inputs.len());
        for input in inputs.iter() {
            outputs.push(mlp.forward(input))
        }
        let first_out = outputs[0].clone();
        let tmp = &first_out[0] - &targets[0];
        let mut loss = tmp.pow(2.0);
        for (out, target) in outputs.iter().zip(targets.iter()).skip(1) {
            let tmp = &out[0] - target;
            loss = loss + tmp.pow(2.0);
        }

        if loss.data() < 0.0001 && loss.data() > -0.001 {
            break;
        }

        println!(
            "Found loss: {:?}\nSample weight:{:?}",
            loss, mlp.layers[0].neurons[0].weights[0]
        );
        println!("Doing backprop over loss");
        loss.backprop();
        mlp.descend(0.01);
        println!("Updated weight: {:?}", mlp.layers[0].neurons[0].weights[0]);
        thread::sleep(time::Duration::from_millis(1000));
    }
    println!("Found low loss, predictions: ");
    for input in inputs.iter() {
        println!("\t{:?} : {:?}", input, mlp.forward(input))
    }
}

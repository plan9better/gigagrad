mod neuron;
mod value;
use neuron::MLP;
use std::error::Error;
use std::fs::File;
use value::Value;

#[allow(dead_code)]
fn load_data(path: &str) -> Result<(Vec<Vec<Value>>, Vec<Value>), Box<dyn Error>> {
    let file = File::open(path)?;
    let mut rdr = csv::Reader::from_reader(file);

    let mut inputs = Vec::new();
    let mut targets = Vec::new();

    for result in rdr.records() {
        let record = result?;

        let x: f64 = record[0].parse()?;
        let y: f64 = record[1].parse()?;

        let label: f64 = record[2].parse()?;

        inputs.push(vec![Value::new(x), Value::new(y)]);
        targets.push(Value::new(label));
    }

    Ok((inputs, targets))
}

#[allow(dead_code)]
fn mock_inputs() -> Result<(Vec<Vec<Value>>, Vec<Value>), Box<dyn Error>> {
    Ok((
        vec![
            vec![Value::new(2.0), Value::new(3.0), Value::new(-1.0)],
            vec![Value::new(3.0), Value::new(-1.0), Value::new(0.5)],
            vec![Value::new(0.5), Value::new(1.0), Value::new(1.0)],
            vec![Value::new(1.0), Value::new(1.0), Value::new(-1.0)],
        ],
        vec![
            Value::new(1.0),
            Value::new(-1.0),
            Value::new(-1.0),
            Value::new(1.0),
        ],
    ))
}

fn main() {
    let data = load_data("dataset.csv");
    let mlp = MLP::new(2, vec![16, 16, 16, 1]);

    // let data = mock_inputs();
    // let mlp = MLP::new(3, vec![4, 4, 1]);
    let (inputs, targets) = data.expect("Failed to open file");
    loop {
        // println!("MLP: {:?}", mlp);
        let mut outputs = Vec::with_capacity(inputs.len());
        for input in inputs.iter() {
            outputs.push(mlp.forward(input))
        }
        let first_out = outputs[0].clone();
        println!("Sample prediction {:?}", first_out);
        let tmp = &first_out[0] - &targets[0];
        let mut loss = tmp.pow(2.0);
        for (out, target) in outputs.iter().zip(targets.iter()).skip(1) {
            let tmp = &out[0] - target;
            loss = loss + tmp.pow(2.0);
        }

        loss = loss / (outputs.len() as f64);

        if loss.data() < 0.0001 && loss.data() > -0.001 {
            break;
        }

        println!(
            "Found loss: {:?}\nSample weight:{:?}",
            loss, mlp.layers[0].neurons[0].weights[0]
        );
        println!("Doing backprop over loss");
        loss.backprop();
        // println!("Params: {:?}", mlp.parameters());
        mlp.descend(0.05);
        println!("Updated weight: {:?}", mlp.layers[0].neurons[0].weights[0]);
        // thread::sleep(time::Duration::from_millis(1000));
    }
    println!("Found low loss, predictions: ");
    for input in inputs.iter() {
        println!("\t{:?} : {:?}", input, mlp.forward(input))
    }
}

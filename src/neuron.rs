use crate::Value;
use rand::Rng;
use std::iter::repeat_with;

#[derive(Debug, Clone)]
pub struct Neuron {
    #[allow(dead_code)]
    pub input_size: usize,
    pub bias: Value,
    pub weights: Vec<Value>,
}

impl Neuron {
    pub fn new(size: usize) -> Self {
        let mut rng = rand::rng();
        let mut vec = Vec::with_capacity(size);
        for _ in 0..size {
            vec.push(Value::new(rng.random_range(-1.0..1.0)));
        }
        Neuron {
            input_size: size,
            bias: Value::new(rng.random_range(-1.0..=1.0)),
            weights: vec,
        }
    }
    pub fn call(&self, inputs: &Vec<Value>) -> Value {
        let mut sum = self.bias.clone();
        for (i, w) in inputs.iter().zip(&self.weights) {
            sum = sum + (w * i)
        }
        sum.tanh()
    }
}
#[derive(Debug, Clone)]
pub struct Layer {
    pub neurons: Vec<Neuron>,
    pub input_size: usize,
    pub output_size: usize,
}

impl Layer {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        let mut ns = Vec::with_capacity(output_size);
        for _ in 0..output_size {
            ns.push(Neuron::new(input_size));
        }
        Layer {
            neurons: ns,
            input_size: input_size,
            output_size: output_size,
        }
    }
    pub fn forward(&self, inputs: Vec<Value>) -> Vec<Value> {
        let mut outputs = Vec::with_capacity(self.output_size);
        for neuron in self.neurons.iter() {
            outputs.push(neuron.call(&inputs));
        }
        return outputs;
    }
}

#[derive(Debug, Clone)]
pub struct MLP {
    pub layers: Vec<Layer>,
    pub input_size: usize,
}

impl MLP {
    pub fn new(inputs_size: usize, layer_sizes: Vec<usize>) -> Self {
        // +1 for input the user is expected to specify the output
        // size themself
        let mut layers = Vec::with_capacity(layer_sizes.len() + 1);

        let mut prev_size = inputs_size;
        for size in layer_sizes {
            layers.push(Layer::new(prev_size, size));
            prev_size = size;
        }
        MLP {
            layers: layers,
            input_size: inputs_size,
        }
    }
    pub fn forward(&self, inputs: Vec<Value>) -> Vec<Value> {
        let mut previous_out = inputs.clone();
        for layer in self.layers.iter() {
            previous_out = layer.forward(previous_out);
        }

        previous_out
    }
}

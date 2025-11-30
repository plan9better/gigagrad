use crate::value::Value;
use rand::Rng;
use std::fmt;

#[derive(Clone)]
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
            bias: Value::new(rng.random_range(-1.0..1.0)),
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

    pub fn parameters(&self) -> Vec<Value> {
        let mut p = self.weights.clone();
        p.push(self.bias.clone());
        p
    }
}

impl fmt::Debug for Neuron {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Neuron(w=[")?;
        for (i, w) in self.weights.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{:.4}", w.data())?;
        }
        write!(f, "], b={:.4})", self.bias.data())
    }
}

#[derive(Clone)]
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

    pub fn parameters(&self) -> Vec<Value> {
        self.neurons.iter().flat_map(|n| n.parameters()).collect()
    }
}

impl fmt::Debug for Layer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Layer(in={}, out={}):",
            self.input_size, self.output_size
        )?;
        for (i, n) in self.neurons.iter().enumerate() {
            writeln!(f, "    {}: {:?}", i, n)?;
        }
        Ok(())
    }
}

#[derive(Clone)]
pub struct MLP {
    pub layers: Vec<Layer>,
    pub input_size: usize,
}

impl MLP {
    pub fn new(inputs_size: usize, layer_sizes: Vec<usize>) -> Self {
        let mut layers = Vec::with_capacity(layer_sizes.len());
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

    pub fn forward(&self, inputs: &Vec<Value>) -> Vec<Value> {
        let mut previous_out = inputs.clone();
        for layer in self.layers.iter() {
            previous_out = layer.forward(previous_out);
        }
        previous_out
    }

    pub fn parameters(&self) -> Vec<Value> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }
    pub fn descend(&self, learning_rate: f64) -> () {
        for p in self.parameters().iter() {
            p.update(learning_rate);
        }
    }
}

impl fmt::Debug for MLP {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "MLP ({} Layers):", self.layers.len())?;
        for (i, layer) in self.layers.iter().enumerate() {
            write!(f, "  L{}: {:?}", i, layer)?;
        }
        Ok(())
    }
}

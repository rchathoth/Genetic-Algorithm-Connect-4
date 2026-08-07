use ndarray::{Array1, Array2};
use serde::{Serialize, Deserialize};
use rand::Rng;
use rand_distr::{Normal, Distribution};

const INPUT_SIZE: usize = 42;
const HIDDEN_SIZE: usize = 16;
const OUTPUT_SIZE: usize = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Network {
    pub w1: Array2<f32>,
    pub b1: Array1<f32>,
    pub w2: Array2<f32>,
    pub b2: Array1<f32>,
}

// ReLU activation function
fn relu(x: f32) -> f32 {
    x.max(0.0)
}

// Sigmoid activation function
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

impl Network {
    pub fn new_random() -> Self {
        let mut rng = rand::thread_rng();

        let scale_w1 = (6.0 / (INPUT_SIZE + HIDDEN_SIZE) as f32).sqrt();
        let scale_w2 = (6.0 / (HIDDEN_SIZE + OUTPUT_SIZE) as f32).sqrt();

        let w1 = Array2::from_shape_fn((HIDDEN_SIZE, INPUT_SIZE), |_| {
            rng.gen_range(-scale_w1..scale_w1)
        });

        let b1 = Array1::from_shape_fn(HIDDEN_SIZE, |_| 0.0);

        let w2 = Array2::from_shape_fn((OUTPUT_SIZE, HIDDEN_SIZE), |_| {
            rng.gen_range(-scale_w2..scale_w2)
        });

        let b2 = Array1::from_shape_fn(OUTPUT_SIZE, |_| 0.0);

        Network { w1, b1, w2, b2 }
    }

    pub fn forward(&self, input: &Array1<f32>) -> f32 {
        // hidden layer multiplication + bias
        let hidden_raw = self.w1.dot(input) + &self.b1;
        let hidden_act = hidden_raw.mapv(relu);
        
        // output layer multiplication + bias
        let output_raw = self.w2.dot(&hidden_act) + &self.b2;
        let output_act = output_raw.mapv(sigmoid);

        output_act[0]
    }

    // In-place Gaussian mutation based on mutation rate and scale
    pub fn mutate(&mut self, rate: f32, scale: f32) {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, scale as f64).unwrap();

        // Mutate w1
        self.w1.mapv_inplace(|w| {
            if rng.gen_range(0.0..1.0) < rate {
                w + (normal.sample(&mut rng) as f32)
            } else {
                w
            }
        });

        // Mutate b1
        self.b1.mapv_inplace(|b| {
            if rng.gen_range(0.0..1.0) < rate {
                b + (normal.sample(&mut rng) as f32)
            } else {
                b
            }
        });

        // Mutate w2
        self.w2.mapv_inplace(|w| {
            if rng.gen_range(0.0..1.0) < rate {
                w + (normal.sample(&mut rng) as f32)
            } else {
                w
            }
        });

        // Mutate b2
        self.b2.mapv_inplace(|b| {
            if rng.gen_range(0.0..1.0) < rate {
                b + (normal.sample(&mut rng) as f32)
            } else {
                b
            }
        });
    }

    // Uniform matrix crossover producing offspring from two parent networks
    pub fn crossover(parent_a: &Network, parent_b: &Network) -> Network {
        let mut rng = rand::thread_rng();

        let w1 = Array2::from_shape_fn(parent_a.w1.dim(), |idx| {
            if rng.gen_bool(0.5) { parent_a.w1[idx] } else { parent_b.w1[idx] }
        });

        let b1 = Array1::from_shape_fn(parent_a.b1.dim(), |idx| {
            if rng.gen_bool(0.5) { parent_a.b1[idx] } else { parent_b.b1[idx] }
        });

        let w2 = Array2::from_shape_fn(parent_a.w2.dim(), |idx| {
            if rng.gen_bool(0.5) { parent_a.w2[idx] } else { parent_b.w2[idx] }
        });

        let b2 = Array1::from_shape_fn(parent_a.b2.dim(), |idx| {
            if rng.gen_bool(0.5) { parent_a.b2[idx] } else { parent_b.b2[idx] }
        });

        Network { w1, b1, w2, b2 }
    }
}








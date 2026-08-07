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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_shapes() {
        let net = Network::new_random();
        assert_eq!(net.w1.shape(), &[16, 42]);
        assert_eq!(net.b1.shape(), &[16]);
        assert_eq!(net.w2.shape(), &[1, 16]);
        assert_eq!(net.b2.shape(), &[1]);
    }

    #[test]
    fn test_forward_pass_output_range() {
        let net = Network::new_random();
        let dummy_input = Array1::zeros(42);
        let score = net.forward(&dummy_input);
        assert!((0.0..=1.0).contains(&score), "Score {} out of range [0.0, 1.0]", score);
    }

    #[test]
    fn test_forward_pass_deterministic() {
        let net = Network::new_random();
        let dummy_input = Array1::from_elem(42, 1.0);
        let score1 = net.forward(&dummy_input);
        let score2 = net.forward(&dummy_input);
        assert_eq!(score1, score2);
    }

    #[test]
    fn test_mutation_zero_rate() {
        let net = Network::new_random();
        let mut mutated = net.clone();
        mutated.mutate(0.0, 1.0);

        assert_eq!(net.w1, mutated.w1);
        assert_eq!(net.b1, mutated.b1);
        assert_eq!(net.w2, mutated.w2);
        assert_eq!(net.b2, mutated.b2);
    }

    #[test]
    fn test_mutation_full_rate() {
        let net = Network::new_random();
        let mut mutated = net.clone();
        mutated.mutate(1.0, 0.5);

        // Every weight and bias element should be modified
        assert!(net.w1 != mutated.w1);
        assert!(net.b1 != mutated.b1);
        assert!(net.w2 != mutated.w2);
        assert!(net.b2 != mutated.b2);
    }

    #[test]
    fn test_crossover() {
        let parent_a = Network::new_random();
        let parent_b = Network::new_random();
        let child = Network::crossover(&parent_a, &parent_b);

        // Check each weight in child comes from either parent_a or parent_b
        for r in 0..16 {
            for c in 0..42 {
                let val = child.w1[[r, c]];
                assert!(val == parent_a.w1[[r, c]] || val == parent_b.w1[[r, c]]);
            }
        }
    }
}






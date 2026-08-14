# Connect 4 Neuroevolution AI Engine

A high-performance Connect 4 Artificial Intelligence engine in **Rust**. The engine pairs a custom Multi-Layer Perceptron (MLP) neural network evaluation function with a Darwinian Genetic Algorithm (GA) evolutionary training engine, eliminating backpropagation calculus in favor of multi-threaded competitive natural selection.

---

## Architecture Overview

* **Game Mechanics ([`board.rs`](connect4/src/board.rs))**: Flat contiguous memory layout (`[Cell; 42]`), localized $O(1)$ directional win checking, and fast $42$-element state vectorization (`+1.0` friendly, `-1.0` enemy, `0.0` empty).
* **Neural Brain ([`nn.rs`](connect4/src/nn.rs))**: Single hidden layer MLP ($42 \to 16 \to 1$) built with `ndarray` (`f32`), Xavier weight initialization, ReLU hidden activation, and Sigmoid scalar desirability output score $\in [0.0, 1.0]$.
* **Genetics Engine ([`nn.rs`](connect4/src/nn.rs))**: Gaussian noise mutation ($\mathcal{N}(0, \text{scale})$) and uniform 50/50 matrix crossover.
* **Parallel Trainer ([`ga.rs`](connect4/src/ga.rs))**: Round-robin tournament simulation leveraged across all available CPU cores using `rayon`, simulating thousands of moves per second.
* **Model Checkpointing ([`main.rs`](connect4/src/main.rs))**: Serializes and deserializes the champion network to [`champion.json`](connect4/champion.json) via `serde` / `serde_json`.

---

## Quickstart & Usage

Navigate to the project crate:
```bash
cd connect4
```

### 1. Train the Evolutionary AI Engine
Spins up the parallel GA loop, logs generational fitness stats (`max_fitness` & `avg_fitness`), and saves the top performer to `champion.json`:
```bash
cargo run --release -- train
```

### 2. Play Against the Trained AI
Loads `champion.json` and launches an interactive ASCII terminal match (Human vs AI with 1-step lookahead):
```bash
cargo run --release -- play
```

---

## Tech Stack

* **Language**: Rust (2024 Edition)
* **Math & Tensors**: `ndarray`
* **Concurrency**: `rayon`
* **Serialization**: `serde` / `serde_json`
* **Randomization**: `rand` & `rand_distr`


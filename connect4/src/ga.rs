use crate::board::{Board, Cell, MoveResult};
use crate::nn::Network;
use rayon::prelude::*;
use rand::Rng;

#[derive(Clone)]
pub struct Bot {
    pub network: Network,
    pub fitness: f32,
}

impl Bot {
    pub fn new() -> Self {
        Self {
            network: Network::new_random(),
            fitness: 0.0,
        }
    }

    /// Evaluates all candidate moves on the board and returns the column index 
    /// with the highest neural network desirability score.
    pub fn choose_move(&self, board: &Board) -> usize {
        let valid_moves = board.get_valid_moves();
        let mut best_col = valid_moves[0];
        let mut max_score = f32::NEG_INFINITY;
        
        for i in valid_moves {
            let mut sim_board = board.clone();
            sim_board.make_move(i);
            let input_vec = sim_board.to_input_vector(board.turn);
            let score = self.network.forward(&input_vec);
            
            if score > max_score {
                max_score = score;
                best_col = i;
            }
        }
        
        best_col
    }

    /// Simulates a single match between self (P1) and opponent (P2).
    /// Returns (self_score, opponent_score).
    pub fn play_match(&self, opponent: &Bot) -> (f32, f32) {
        let mut board = Board::new();
        let mut p1_score = 0.0;
        let mut p2_score = 0.0;

        for _move_num in 0..42 {
            p1_score += 0.01;
            p2_score += 0.01;

            let col = if board.turn == Cell::P1 {
                self.choose_move(&board)
            } else {
                opponent.choose_move(&board)
            };

            match board.make_move(col) {
                MoveResult::Win(Cell::P1) => {
                    p1_score += 3.0;
                    break;
                }
                MoveResult::Win(Cell::P2) => {
                    p2_score += 3.0;
                    break;
                }
                MoveResult::Win(_) => break,
                MoveResult::Draw => {
                    p1_score += 1.0;
                    p2_score += 1.0;
                    break;
                }
                MoveResult::Ongoing => {}
                MoveResult::Invalid => break,
            }
        }

        (p1_score, p2_score)
    }
}

pub struct Population {
    pub bots: Vec<Bot>,
}

impl Population {
    pub fn new(size: usize) -> Self {
        let bots = (0..size).map(|_| Bot::new()).collect();
        Self { bots }
    }

    /// Executes pairwise round-robin matches in parallel using Rayon and updates fitness scores.
    pub fn evaluate_parallel(&mut self) {
        for bot in self.bots.iter_mut() {
            bot.fitness = 0.0;
        }

        let n = self.bots.len();
        if n <= 1 {
            return;
        }

        let pairs: Vec<(usize, usize)> = (0..n)
            .flat_map(|i| ((i + 1)..n).map(move |j| (i, j)))
            .collect();

        let results: Vec<(usize, usize, f32, f32)> = pairs
            .par_iter()
            .map(|&(i, j)| {
                let (score_i1, score_j1) = self.bots[i].play_match(&self.bots[j]);
                let (score_j2, score_i2) = self.bots[j].play_match(&self.bots[i]);

                (i, j, score_i1 + score_i2, score_j1 + score_j2)
            })
            .collect();

        for (i, j, score_i, score_j) in results {
            self.bots[i].fitness += score_i;
            self.bots[j].fitness += score_j;
        }
    }

    /// Advances the population to the next generation via selection, crossover, and mutation.
    pub fn evolve(&mut self, elite_count: usize, mutation_rate: f32, mutation_scale: f32) {
        let pop_size = self.bots.len();
        if pop_size == 0 {
            return;
        }

        self.bots.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap_or(std::cmp::Ordering::Equal));

        let num_elites = elite_count.min(pop_size);
        let mut new_bots: Vec<Bot> = Vec::with_capacity(pop_size);

        for i in 0..num_elites {
            let mut elite = self.bots[i].clone();
            elite.fitness = 0.0;
            new_bots.push(elite);
        }

        let mut rng = rand::thread_rng();
        let tournament_pool_size = (pop_size / 2).max(1);

        while new_bots.len() < pop_size {
            let parent_a_idx = rng.gen_range(0..tournament_pool_size);
            let parent_b_idx = rng.gen_range(0..tournament_pool_size);

            let parent_a = &self.bots[parent_a_idx];
            let parent_b = &self.bots[parent_b_idx];

            let mut child_network = Network::crossover(&parent_a.network, &parent_b.network);
            child_network.mutate(mutation_rate, mutation_scale);

            new_bots.push(Bot {
                network: child_network,
                fitness: 0.0,
            });
        }

        self.bots = new_bots;
    }

    pub fn get_champion(&self) -> &Bot {
        self.bots
            .iter()
            .max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap()
    }

    pub fn avg_fitness(&self) -> f32 {
        if self.bots.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.bots.iter().map(|b| b.fitness).sum();
        sum / self.bots.len() as f32
    }
}
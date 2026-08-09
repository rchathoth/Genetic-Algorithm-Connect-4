use crate::board::{Board, Cell, MoveResult};
use crate::nn::Network;

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
}
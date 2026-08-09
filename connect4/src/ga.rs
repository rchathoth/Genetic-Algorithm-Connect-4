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
        
        for i in valid_moves {
            let mut sim_board = board.clone();
            sim_board.make_move(i);
            let input_vec = sim_board.to_input_vector(board.turn);
                        
        }
        // TODO: Iterate over valid_moves
        // TODO: Clone board, execute move, convert to input vector
        // TODO: Evaluate using self.network.forward()
        // TODO: Track and return the column with the highest evaluation score
    }
}
use ndarray::Array1;

// Enum for cell state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cell {
    Empty,
    P1,
    P2,
}

// Enum representing the result of making a move
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoveResult {
    Invalid,
    Ongoing,
    Win(Cell),
    Draw,
}

// Struct of game board
#[derive(Debug, Clone)]
pub struct Board {
    pub board: [Cell; 42],
    pub turn: Cell,
}

impl Default for Board {
    fn default() -> Self {
        Self::new()
    }
}

// Implementation of the board struct
impl Board {
    // Constructor
    pub fn new() -> Self {
        Self {
            board: [Cell::Empty; 42],
            turn: Cell::P1,
        }
    }
    
    // Returns available non-full column indices (0..7)
    pub fn get_valid_moves(&self) -> Vec<usize> {
        (0..7).filter(|&c| self.board[c] == Cell::Empty).collect()
    }

    // Converts flattened board to f32 Array1 vector (+1.0 friendly, -1.0 enemy, 0.0 empty)
    pub fn to_input_vector(&self, active_player: Cell) -> Array1<f32> {
        Array1::from_shape_fn(42, |i| match self.board[i] {
            Cell::Empty => 0.0,
            cell if cell == active_player => 1.0,
            _ => -1.0,
        })
    }

    // Makes a move on the board
    pub fn make_move(&mut self, col: usize) -> MoveResult {
        // check if the move is valid
        if col >= 7 || self.board[col] != Cell::Empty {
            return MoveResult::Invalid;
        }

        // Determine the index
        let mut idx = col + 35;
        while self.board[idx] != Cell::Empty {
            idx -= 7;
        }
        
        let player = self.turn;
        // Make move
        self.board[idx] = player;

        // Switch turns
        if self.turn == Cell::P1 {
            self.turn = Cell::P2;
        } else {
            self.turn = Cell::P1;
        }

        // Check for win
        if self.check_win_at(idx) {
            return MoveResult::Win(player);
        }

        // Check for draw (all top cells are filled)
        if (0..7).all(|c| self.board[c] != Cell::Empty) {
            return MoveResult::Draw;
        }

        MoveResult::Ongoing
    }

    // Helper method to check if the last placed piece at idx results in a win
    pub fn check_win_at(&self, idx: usize) -> bool {
        let player = self.board[idx];
        if player == Cell::Empty {
            return false;
        }

        let r = (idx / 7) as isize;
        let c = (idx % 7) as isize;

        let directions = [
            (0, 1),  // Horizontal
            (1, 0),  // Vertical
            (1, 1),  // Diagonal down-right (or up-left)
            (1, -1), // Diagonal down-left (or up-right)
        ];

        for &(dr, dc) in &directions {
            let mut count = 1;

            // Go forward
            let mut nr = r + dr;
            let mut nc = c + dc;
            while nr >= 0 && nr < 6 && nc >= 0 && nc < 7 {
                let nidx = (nr * 7 + nc) as usize;
                if self.board[nidx] == player {
                    count += 1;
                    nr += dr;
                    nc += dc;
                } else {
                    break;
                }
            }

            // Go backward
            let mut nr = r - dr;
            let mut nc = c - dc;
            while nr >= 0 && nr < 6 && nc >= 0 && nc < 7 {
                let nidx = (nr * 7 + nc) as usize;
                if self.board[nidx] == player {
                    count += 1;
                    nr -= dr;
                    nc -= dc;
                } else {
                    break;
                }
            }

            if count >= 4 {
                return true;
            }
        }

        false
    }
    
    // Displays the board in the terminal
    pub fn display(&self) {
        for r in 0..6 {
            for c in 0..7 {
                print!("{} ", match self.board[r * 7 + c] {
                    Cell::Empty => '.',
                    Cell::P1 => 'X',
                    Cell::P2 => 'O',
                });
            }
            println!();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_moves_initial() {
        let board = Board::new();
        assert_eq!(board.get_valid_moves(), vec![0, 1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_input_vector() {
        let mut board = Board::new();
        board.make_move(0); // P1 move at index 35 (col 0, bottom row)
        let vec_p1 = board.to_input_vector(Cell::P1);
        let vec_p2 = board.to_input_vector(Cell::P2);

        assert_eq!(vec_p1[35], 1.0);
        assert_eq!(vec_p2[35], -1.0);
        assert_eq!(vec_p1[0], 0.0);
    }

    #[test]
    fn test_horizontal_win() {
        let mut board = Board::new();
        // P1 places in 0, 1, 2, 3 (interleaved with P2 in col 6)
        for col in 0..3 {
            assert_eq!(board.make_move(col), MoveResult::Ongoing); // P1
            assert_eq!(board.make_move(6), MoveResult::Ongoing);   // P2
        }
        let res = board.make_move(3); // P1 4th piece
        assert_eq!(res, MoveResult::Win(Cell::P1));
    }
}
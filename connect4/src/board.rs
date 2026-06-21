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
pub struct Board {
    pub board: [Cell; 42],
    pub turn: Cell,
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
}
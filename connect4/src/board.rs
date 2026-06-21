// Enum for cell state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cell {
    Empty,
    P1,
    P2,
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
    pub fn make_move(&mut self, col: usize) -> bool{
        // check if the move is valid
        if self.board[col] != Cell::Empty {
            println!("Invalid move: try again.");
            return false;
        }

        // Determine the index
        let mut idx = col + 35;
        while self.board[idx] != Cell::Empty {
            idx -= 7;
        }
        // Make move
        self.board[idx] = self.turn;

        // Switch turns
        if self.turn == Cell::P1 {
            self.turn = Cell::P2;
        }
        else {
            self.turn = Cell::P1;
        }

        // Check for win
        


        

        
    }
}

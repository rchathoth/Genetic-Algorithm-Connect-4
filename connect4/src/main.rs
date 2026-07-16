mod board;

use board::{Board, MoveResult, Cell};
use std::io::{self, Write};

fn main() {
    let mut board = Board::new();

    println!("Welcome to Connect 4!");
    println!("Columns are numbered 0 to 6.");

    loop {
        board.display();
        let active_player = match board.turn {
            Cell::P1 => "Player 1 (X)",
            Cell::P2 => "Player 2 (O)",
            Cell::Empty => unreachable!(),
        };

        print!("{}, enter column (0-6) or 'q' to quit: ", active_player);
        io::stdout().flush().unwrap();

        let mut input = String::new();
        if io::stdin().read_line(&mut input).is_err() {
            println!("Failed to read input.");
            continue;
        }

        let trimmed = input.trim();
        if trimmed.eq_ignore_ascii_case("q") {
            println!("Goodbye!");
            break;
        }

        let col: usize = match trimmed.parse() {
            Ok(num) if num < 7 => num,
            _ => {
                println!("Invalid input. Please enter a column number between 0 and 6.");
                continue;
            }
        };

        match board.make_move(col) {
            MoveResult::Invalid => {
                println!("Column is full or invalid move. Try again.");
            }
            MoveResult::Ongoing => {
                // Game continues
            }
            MoveResult::Win(player) => {
                board.display();
                match player {
                    Cell::P1 => println!("Player 1 (X) wins!"),
                    Cell::P2 => println!("Player 2 (O) wins!"),
                    Cell::Empty => unreachable!(),
                }
                break;
            }
            MoveResult::Draw => {
                board.display();
                println!("It's a draw!");
                break;
            }
        }
    }
}

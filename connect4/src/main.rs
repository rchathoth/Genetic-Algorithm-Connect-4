mod board;
mod nn;
mod ga;

use board::{Board, MoveResult, Cell};
use nn::Network;
use ga::{Bot, Population};

use std::env;
use std::fs::File;
use std::io::{self, Write, Read};

const CHAMPION_PATH: &str = "champion.json";

// ==========================================
// STORING BEST WEIGHTS
// ==========================================

/// Saves the given neural network weights to `champion.json` using serde_json.
fn save_champion(network: &Network, path: &str) -> io::Result<()> {
    let json = serde_json::to_string_pretty(network)?;
    let mut file = File::create(path)?;
    file.write_all(json.as_bytes())?;
    Ok(())
}

/// Loads a neural network from `champion.json`.
fn load_champion(path: &str) -> io::Result<Network> {
    let mut file = File::open(path)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    let network: Network = serde_json::from_str(&contents)?;
    Ok(network)
}

// ==========================================
// 2. TRAINING MODE PIPELINE
// ==========================================

/// Runs the Genetic Algorithm evolution loop across generations.
fn run_training_mode(generations: usize, pop_size: usize) {
    println!("Starting Connect 4 GA Training Pipeline...");
    println!("Population Size: {} | Generations: {}", pop_size, generations);

    let mut pop = Population::new(pop_size);

    for generation_idx in 1..=generations {
        pop.evaluate_parallel();

        let champ = pop.get_champion();
        let max_fit = champ.fitness;
        let avg_fit = pop.avg_fitness();

        println!("Gen [{}/{}] | Max Fitness: {:.2} | Avg Fitness: {:.2}", generation_idx, generations, max_fit, avg_fit);

        if let Err(e) = save_champion(&champ.network, CHAMPION_PATH) {
             eprintln!("Warning: Failed to save champion: {}", e);
         }

        pop.evolve(5, 0.05, 0.1); // 5 elites, 5% mutation rate, 0.1 mutation scale
    }

    println!("Training complete! Best model saved to {}", CHAMPION_PATH);
}

// ==========================================
// 3. HUMAN VS AI PLAY MODE
// ==========================================

/// Interactive Human vs AI play mode using the trained champion network.
fn run_play_mode() {
    println!("🎮 Connect 4: Human vs AI Mode");

    // TODO 1: Load champion network from disk, or fallback to random
    let bot = match load_champion(CHAMPION_PATH) {
        Ok(net) => {
            println!("Loaded champion model from {}", CHAMPION_PATH);
            Bot { network: net, fitness: 0.0 }
        }
        Err(_) => {
            println!("Warning: Could not load {}, using random Bot.", CHAMPION_PATH);
            Bot::new()
        }
    };

    let mut board = Board::new();
    let human_player = Cell::P1; // Human is P1 (X), AI is P2 (O)

    loop {
        board.display();

        if board.turn == human_player {
            // TODO 2: Human move input (read column 0..6 or 'q' to quit)
            print!("Your turn (Player X), enter column (0-6) or 'q': ");
            io::stdout().flush().unwrap();

            let mut input = String::new();
            io::stdin().read_line(&mut input).unwrap();
            let trimmed = input.trim();

            if trimmed.eq_ignore_ascii_case("q") {
                break;
            }

            let col: usize = match trimmed.parse() {
                Ok(c) if c < 7 => c,
                _ => {
                    println!("Invalid column!");
                    continue;
                }
            };

            match board.make_move(col) {
                MoveResult::Invalid => println!("Column full!"),
                MoveResult::Win(player) => {
                    board.display();
                    println!("🎉 {:?} Wins!", player);
                    break;
                }
                MoveResult::Draw => {
                    board.display();
                    println!("🤝 It's a Draw!");
                    break;
                }
                MoveResult::Ongoing => {}
            }
        } else {
            // TODO 3: AI move selection using bot.choose_move(&board)
            println!("🤖 AI is thinking...");
            let ai_col = bot.choose_move(&board);
            println!("AI selected column {}", ai_col);

            match board.make_move(ai_col) {
                MoveResult::Win(player) => {
                    board.display();
                    println!("🤖 {:?} Wins!", player);
                    break;
                }
                MoveResult::Draw => {
                    board.display();
                    println!("🤝 It's a Draw!");
                    break;
                }
                MoveResult::Ongoing => {}
                MoveResult::Invalid => unreachable!(),
            }
        }
    }
}

// ==========================================
// 4. MAIN ENTRY POINT & CLI PARSER
// ==========================================

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() > 1 {
        match args[1].as_str() {
            "--mode" | "-m" => {
                if args.len() > 2 && args[2] == "train" {
                    run_training_mode(100, 100); // 100 generations, 100 population
                } else {
                    run_play_mode();
                }
            }
            "train" => run_training_mode(100, 100),
            "play" => run_play_mode(),
            _ => run_play_mode(),
        }
    } else {
        // Prompt interactive choice if no CLI flag provided
        println!("Select Mode:");
        println!("1. Train AI Engine");
        println!("2. Play against Trained AI");
        print!("Choice (1 or 2): ");
        io::stdout().flush().unwrap();

        let mut choice = String::new();
        io::stdin().read_line(&mut choice).unwrap();
        match choice.trim() {
            "1" => run_training_mode(100, 100),
            _ => run_play_mode(),
        }
    }
}
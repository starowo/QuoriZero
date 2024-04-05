pub mod net;
pub mod game;
pub mod mcts_pure_parallel;
pub mod train;
pub mod mcts_a0;

#[tokio::main]
async fn main() {

    let args: Vec<String> = std::env::args().collect();
/*            if args.len() < 2 {
                println!("Usage: quorizero <server_address> <server_port>");
                return;
            }
            let server_address = &args[1];
            let server_port = &args[2]; */
    let server_address = "http://103.215.37.27";
    let server_port = "8000";
    let server_url = format!("{}:{}", server_address, server_port);
    train::train(server_url).await;
}

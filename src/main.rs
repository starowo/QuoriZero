pub mod net;
pub mod game;
pub mod mcts_pure_parallel;
pub mod train;
pub mod mcts_a0;

use std::collections::HashMap;
use std::thread;
use std::sync::mpsc;
use websocket::sync::Server;
use websocket::OwnedMessage;

#[tokio::main]
async fn main() {

    

    let server = Server::bind("127.0.0.1:8080").unwrap();

    for request in server.filter_map(Result::ok) {
        // Spawn a new thread for each connection
        thread::spawn(move || {
            // read from args
            let args: Vec<String> = std::env::args().collect();
            if args.len() < 2 {
                println!("Usage: quorizero <server_address> <server_port>");
                return;
            }
            let server_address = &args[1];
            let server_port = &args[2];
            let server_url = format!("http://{}:{}/", server_address, server_port);
            if !request.protocols().contains(&"rust-websocket".to_string()) {
                request.reject().unwrap();
                return;
            }

            let client = request.use_protocol("rust-websocket").accept().unwrap();
            let ip = client.peer_addr().unwrap();

            println!("Connection from {}", ip);

            let (mut receiver, mut sender) = client.split().unwrap();

            let (tx, rx) = mpsc::channel();

            thread::spawn(move || {
                for message in rx {
                    sender.send_message(&message).unwrap();
                }
            });

            let id = 0;

            let mut tx_map = HashMap::new();

            for message in receiver.incoming_messages() {
                
                let (in_tx, in_rx) = mpsc::channel();
                let message = message.unwrap();
                let tx = tx.clone();
                match message {
                    OwnedMessage::Close(_) => {
                        let _ = tx.send(OwnedMessage::Close(None));
                        println!("Client {} disconnected", ip);
                        return;
                    }
                    OwnedMessage::Ping(ping) => {
                        let _ = tx.send(OwnedMessage::Pong(ping));
                    }
                    OwnedMessage::Text(txt) => {
                        // Handle received text message
                        
                        println!("Received message: {}", txt);

                        // Here you can process the received message and send back game data

                        // Example: sending back an echo message
                        //let _ = tx.send(OwnedMessage::Text(txt));
                        if txt == "train" {
                            train::train(server_url.clone(), Some(tx.clone()));
                        }
                        if txt == "humanplay" {
                            train::play(Some(tx.clone()), in_rx);
                            tx_map.insert(id, in_tx);
                        }
                        if txt.starts_with("operate") {
                            let in_tx = tx_map.get(&id).unwrap();
                            let move_id = txt.split_whitespace().nth(1).unwrap().parse::<usize>().unwrap();
                            let _ = in_tx.send(move_id);
                        }
                    }
                    _ => (),
                }
            }
        });
    }
}

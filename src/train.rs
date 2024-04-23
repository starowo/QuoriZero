use ndarray::prelude::*;
use ndarray::Array3;
use rand::seq::SliceRandom;
use rand::Rng;
use reqwest::Client;
use serde::Deserialize;
use websocket::client;
use std::collections::HashSet;
use std::io;
use std::io::copy;
use std::io::Write;
use std::sync::atomic::{AtomicI32, AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{mpsc, RwLock};
use std::time::Duration;
use std::{sync::Arc, thread};
use tch::{Device, Tensor};
use websocket::OwnedMessage;


use super::game::Board;
use super::mcts_a0::MCTSPlayer;
use super::net::{self, Net};
use serde::Serialize;

pub async fn train(http_address: String, tx: Option<mpsc::Sender<OwnedMessage>>) {
    //humanplay(net::Net::new(Some("skating_best.model")), 1e-4, 2., 4000, true, 1, 1);
    //ai_suggestion(net::Net::new(Some("skating_best.model")), 1e-4, 2.0, 4000, true, 0);
    //TrainPipeline::new(http_address, tx).train().await;
    //weight_comparation(Arc::new(net::Net::new(Some("santorini_best copy.model")).into()), 1e-4, 5., 400);
    //evaluate_with_pure_mcts(Arc::new(net::Net::new(Some("skating_best.model")).into()), 1e-4, 5.0, 400, 50000, false);
    //test()
}

pub fn play(tx: Option<Sender<OwnedMessage>>, rx: Receiver<usize>) {
    //humanplay(net::Net::new(Some("latest.model")), 1e-4, 1.5, 800, true, 2, 2, tx.clone(), rx);
    thread::spawn(move || {
        humanplay(
            net::Net::new(Some("latest.model")),
            1e-4,
            4.0,
            3000,
            true,
            2,
            2,
            tx.clone(),
            rx,
        );
    });
}

pub fn showplay(tx: Option<Sender<OwnedMessage>>) {
    //humanplay(net::Net::new(Some("latest.model")), 1e-4, 1.5, 800, true, 2, 2, tx.clone(), rx);
    thread::spawn(move || {
        weight_comparation(
            Arc::new(RwLock::new(net::Net::new(Some("latest.model")))),
            1e-4,
            4.0,
            2400,
            tx.clone(),
        );
    });
}

fn test() {
    /*
    let tensor = Tensor::of_slice(&size);
    let parts = tensor.split_with_sizes(&[81, 1], 0);
    let (p, v): (&Vec<f64>, &Tensor) = (&parts[0].exp().into(), &parts[1]);
    let mut probs = vec![];
    for i in 0..81 {
        if available.contains(&i) {
            probs.push(p[i]);
        }else {
            probs.push(0.);
        }
    }
    let value: f64 = v.into();
    println!("{:?}, {}", probs, value)
     */
    let mut board = Board::new(None);
    board.init(0);
}
pub const BATCH_SIZE: usize = 512;
const BUFFER_SIZE: usize = 10000;

const SELFPLAY_PLAYOUT: usize = 800;
const SELFPLAY_TEMP: f32 = 1.0;
const SELFPLAY_CPUCT: f32 = 2.0;

struct TrainPipeline {
    http_address: String,
    tx: Option<mpsc::Sender<OwnedMessage>>,
    timestamp: u64,
    net: net::NetTrain,
    data_buffer: Vec<SingleData>,
    kl_targ: f32,
    lr: f64,
    lr_multiplier: f64,
    evaluate_playout: usize,
    win_rate: f32,
}

impl TrainPipeline {
    fn new(http_address: String, tx: Option<mpsc::Sender<OwnedMessage>>) -> Self {
        Self {
            http_address,
            tx,
            timestamp: 0,
            net: net::NetTrain::new(if std::path::Path::new("latest.model").exists() {
                Some("latest.model")
            } else {
                None
            }),
            data_buffer: Vec::new(),
            kl_targ: 0.0005,
            lr: 0.02,
            lr_multiplier: 1.0 ,
            evaluate_playout: 1200,
            win_rate: 0.8,

        }
    }

    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = SingleData>,
    {
        for e in iter {
            self.data_buffer.push(e);
            //if self.data_buffer.len() > BUFFER_SIZE {
            //    self.data_buffer.pop_front();
            //}
        }
    }

    async fn send_data_to_server(&mut self) {
        let client = Client::new();
        let data = serde_json::to_string(&self.data_buffer).unwrap();
        let res = client.post(&format!("{}/train", self.http_address))
            .body(data)
            .send()
            .await
            .unwrap();
    }

    async fn get_data_from_server(&mut self) {
        let client = Client::new();
        let res = client.get(&format!("{}/getdata", self.http_address))
            .send()
            .await
            .unwrap();
        let content = res.text().await.unwrap();
        let contents = content.split("\n");
        for content in contents {
            self.data_buffer.extend::<Vec<SingleData>>(serde_json::from_str(content).unwrap());
        }
    }

}
fn has_duplicate_values(vec: Vec<usize>) -> bool {
    let set: HashSet<_> = vec.iter().collect();
    vec.len() != set.len()
}
#[derive(Clone, Serialize, Deserialize)]
struct SingleData {
    state: (Vec<f32>, Vec<f32>, f32),
    loss: f32,
    weight: f32,
}

impl SingleData {

    fn get_state(&self) -> (Array3<f32>, Vec<f32>, f32) {
        let state = Array3::from_shape_vec((9, 19, 19), self.state.0.clone()).unwrap();
        (state, self.state.1.clone(), self.state.2)
    }
    
}

fn humanplay<'a>(
    net: net::Net,
    temp: f32,
    c_puct: f32,
    n_playout: usize,
    a0: bool,
    start: u16,
    player_: u8,
    tx: Option<Sender<OwnedMessage>>,
    rx: Receiver<usize>
) {
    let mut board = Board::new(tx.clone());
    board.init(start);
    if a0 {
        let mut player = MCTSPlayer::new(Arc::new(net.into()), c_puct, false, 4);
        let mut history = vec![];
        loop {
            history.push(board.clone());
            let current = board.current_player();
            if current == player_ {
                let (move_, nums) = player.get_action(&mut board, temp, true, n_playout, 4);
                let mut map = nums
                    .iter()
                    .enumerate()
                    .map(|(k, v)| (k, v))
                    .collect::<Vec<(usize, &f32)>>();
                map.sort_by(|(_, b), (_, a)| a.partial_cmp(b).unwrap());
                board.do_move(move_.try_into().unwrap(), true, true);
                player.mcts.update_with_move(move_, false);
                println!("move:{}", move_);
            } else {
                let p = rx.recv();
                let p = p.unwrap();
                board.do_move(p.try_into().unwrap(), true, true);
                player.mcts.update_with_move(p.try_into().unwrap(), false);
            }
            let (end, winner) = board.game_end();
            if end {
                println!("winner is {}", winner);
                return;
            }
        }
    }
}

fn weight_comparation(
    net: Arc<RwLock<Net>>,
    temp: f32,
    c_puct: f32,
    n_playout_a0: usize,
    tx: Option<Sender<OwnedMessage>>,
) -> f32 {
    let a = Arc::new(AtomicI32::new(0));
    let mut threads = vec![];
    //let time = std::time::SystemTime::now();
    let played = Arc::new(AtomicUsize::new(0));
    let best = Arc::new(RwLock::new(Net::new(Some("best.model"))));
    for i in 0..1 {
        let a1 = a.clone();
        let net = net.clone();
        let best = best.clone();
        let played = played.clone();
        let tx = tx.clone();
        let t = std::thread::Builder::new()
            .name(format!("thread {}", i))
            .spawn(move || {
                while played.load(Ordering::SeqCst) < 1 {
                    played.fetch_add(1, Ordering::SeqCst);
                    let n = net.clone();
                    let mut board = Board::new(tx.clone());
                    let b = best.clone();
                    board.init(i % 2 + 1);
                    if true {
                        let mut player = MCTSPlayer::new(n, c_puct, false, 2);
                        let mut best = MCTSPlayer::new(b, c_puct, false, 2);
                        let mut turn = 0;
                        loop {
                            let current = board.current_player();
                            if current == 1 {
                                let (move_, nums) = player.get_action(&mut board, temp, true, n_playout_a0, 4);
                                let mut map = nums
                                    .iter()
                                    .enumerate()
                                    .map(|(k, v)| (k, v))
                                    .collect::<Vec<(usize, &f32)>>();
                                map.sort_by(|(_, b), (_, a)| a.partial_cmp(b).unwrap());
                                board.do_move(move_.try_into().unwrap(), true, true);
                                player.mcts.update_with_move(move_, false);
                                best.mcts.update_with_move(move_, false);
                            } else {
                                let (move_, nums) = best.get_action(&mut board, temp, true, n_playout_a0, 4);
                                let mut map = nums
                                    .iter()
                                    .enumerate()
                                    .map(|(k, v)| (k, v))
                                    .collect::<Vec<(usize, &f32)>>();
                                map.sort_by(|(_, b), (_, a)| a.partial_cmp(b).unwrap());
                                board.do_move(move_.try_into().unwrap(), true, true);
                                player.mcts.update_with_move(move_, false);
                                best.mcts.update_with_move(move_, false);
                            }
                            let (mut end, mut winner) = board.game_end();
                            if turn >= 150 && !end {
                                end = true;
                                winner = -1;
                            }
                            turn += 1;
                            if end {
                                if winner == 1 {
                                    a1.fetch_add(2, std::sync::atomic::Ordering::SeqCst);
                                    println!("winner is current weight")
                                } else {
                                    if winner == 2 {
                                        println!("winner is older weight")
                                    } else {
                                        a1.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                                        println!("draw game")
                                    }
                                }
                                break;
                            }
                        }
                    }
                }
            })
            .unwrap();
        threads.push(t);
    }
    for t in threads {
        t.join().unwrap();
    }
    a.load(std::sync::atomic::Ordering::SeqCst) as f32 / 2.
}


fn get_equi_data(data: Vec<SingleData>) -> Vec<SingleData> {
    let mut result = vec![];
    result.extend(data.clone());
    for v in data {
        let arr = v.get_state().1;
        result.push(into_data(
            (flipud_planes(&v.get_state().0), flipud_actions(&arr), v.state.2),
            v.weight,
        ));
        result.push(into_data(
            (fliplr_planes(&v.get_state().0), fliplr_actions(&arr), v.state.2),
            v.weight,
        ));
        result.push(into_data(
            (
                flipud_planes(&fliplr_planes(&v.get_state().0)),
                flipud_actions(&fliplr_actions(&arr)),
                v.state.2,
            ),
            v.weight,
        ));
    }
    result
}

fn into_data(state: (Array3<f32>, Vec<f32>, f32), weight: f32) -> SingleData {
    SingleData {
        state: (state.0.into_raw_vec(), state.1, state.2),
        loss: 0.,
        weight,
    }
}

fn send_data(data: Vec<SingleData>, tx: Sender<OwnedMessage>) {
    let size = data.len() / 4;
    {
        let mut s: String = String::from("data");
        let sample = data[25].clone();
        for i in sample.state.0.iter() {
            s.push_str(&format!("{},", i));
        }
        for i in sample.state.1.iter() {
            s.push_str(&format!("{},", i));
        }

        tx.send(OwnedMessage::Text(s)).unwrap();
    }
    for j in 0..3 {
        let mut s: String = String::from("data");
        let sample = data[size + 25 * 3 + j].clone();
        for i in sample.state.0.iter() {
            s.push_str(&format!("{},", i));
        }
        for i in sample.state.1.iter() {
            s.push_str(&format!("{},", i));
        }

        tx.send(OwnedMessage::Text(s)).unwrap();
    }
}

fn fliplr_actions(actions: &Vec<f32>) -> Vec<f32> {
    // flip individually: 0-63 64-127 128-208
    let mut result = vec![0.; 132];
    for i in 0..64 {
        let action = actions[i];
        let x = i % 8;
        let y = i / 8;
        let new_x = 7 - x;
        let new_y = y;
        let new_i = new_y * 8 + new_x;
        result[new_i] = action;
    }
    for i in 64..128 {
        let action = actions[i];
        let old_i = i - 64;
        let x = old_i % 8;
        let y = old_i / 8;
        let new_x = 7 - x;
        let new_y = y;
        let new_i = new_y * 8 + new_x;
        result[new_i + 64] = action;
    }
    for i in 128..132 {
        let mut action = actions[i];
        if i == 128 {
            action = actions[130];
        } else if i == 130 {
            action = actions[128];
        }
        result[i] = action;
    }
    result
}

fn flipud_actions(actions: &Vec<f32>) -> Vec<f32> {
    // flip individually: 0-63 64-127 128-208
    let mut result = vec![0.; 132];
    for i in 0..64 {
        let action = actions[i];
        let x = i % 8;
        let y = i / 8;
        let new_x = x;
        let new_y = 7 - y;
        let new_i = new_y * 8 + new_x;
        result[new_i] = action;
    }
    for i in 64..128 {
        let action = actions[i];
        let old_i = i - 64;
        let x = old_i % 8;
        let y = old_i / 8;
        let new_x = x;
        let new_y = 7 - y;
        let new_i = new_y * 8 + new_x;
        result[new_i + 64] = action;
    }
    for i in 128..132 {
        let mut action = actions[i];
        if i == 129 {
            action = actions[131];
        } else if i == 131 {
            action = actions[129];
        }
        result[i] = action;
    }
    result
}

fn fliplr(matrix: &Array2<f32>) -> Array2<f32> {
    let (m, n) = matrix.dim();
    let mut result = Array::zeros((n, m));

    for i in 0..n {
        let col = matrix.column(n - i - 1);
        result.column_mut(i).assign(&col);
    }

    result
}

fn flipud(matrix: &Array2<f32>) -> Array2<f32> {
    let (m, n) = matrix.dim();
    let mut result = Array::zeros((n, m));

    for i in 0..m {
        let col = matrix.row(m - i - 1);
        result.row_mut(i).assign(&col);
    }

    result
}

fn flipud_planes(array: &Array3<f32>) -> Array3<f32> {
    let (m, n, p) = array.dim();
    let mut result = Array::zeros((m, n, p));

    for i in 0..m {
        let plane = array.slice(s![i, .., ..]).to_owned(); // Convert borrowed view to owned array
        let rotated_plane = flipud(&plane.into_shape([n, p]).unwrap()); // Pass a 2-dimensional array slice
                                                                        /*
                                                                        // swap 1 and 2, 3 and 4
                                                                        if i == 1 {
                                                                            result.slice_mut(s![2, .., ..]).assign(&rotated_plane);
                                                                        } else if i == 2 {
                                                                            result.slice_mut(s![1, .., ..]).assign(&rotated_plane);
                                                                        } else if i == 3 {
                                                                            result.slice_mut(s![4, .., ..]).assign(&rotated_plane);
                                                                        } else if i == 4 {
                                                                            result.slice_mut(s![3, .., ..]).assign(&rotated_plane);
                                                                        } else {
                                                                            result.slice_mut(s![i, .., ..]).assign(&rotated_plane);
                                                                        }
                                                                         */
        result.slice_mut(s![i, .., ..]).assign(&rotated_plane);
    }

    /*
       // for the last plane(should be filled with 0 or 1), do not flip, but swap the 0 and 1 then copy
       let plane = array.slice(s![m-1, .., ..]).to_owned(); // Convert borrowed view to owned array
       let flipped_plane = plane.mapv(|x| 1.0 - x); // Pass a 2-dimensional array slice
       result.slice_mut(s![m-1, .., ..]).assign(&flipped_plane);
    */
    result
}

fn fliplr_planes(array: &Array3<f32>) -> Array3<f32> {
    let (m, n, p) = array.dim();
    let mut result = Array::zeros((m, n, p));

    for i in 0..5 {
        let plane = array.slice(s![i, .., ..]).to_owned(); // Convert borrowed view to owned array
        let rotated_plane = fliplr(&plane.into_shape([n, p]).unwrap()); // Pass a 2-dimensional array slice
        result.slice_mut(s![i, .., ..]).assign(&rotated_plane);
    }
    for i in 5..m {
        let plane = array.slice(s![i, .., ..]).to_owned(); // Convert borrowed view to owned array
        result.slice_mut(s![i, .., ..]).assign(&plane);
    }

    result
}

use ndarray::prelude::*;
use ndarray::Array3;
use rand::seq::SliceRandom;
use rand::Rng;
use reqwest::Client;
use serde::Deserialize;
use tokio::sync::Mutex;
use uuid::timestamp;
use std::collections::HashSet;
use std::io;
use std::io::copy;
use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::RwLock;
use std::time::Duration;
use std::{sync::Arc, thread};
use tch::{Device, Tensor};

use super::game::Board;
use super::mcts_a0::MCTSPlayer;
use super::net::{self};
use serde::Serialize;

pub async fn train(http_address: String, num_threads: usize, num_processes: usize) {
    //humanplay(net::Net::new(Some("skating_best.model")), 1e-4, 2., 4000, true, 1, 1);
    //ai_suggestion(net::Net::new(Some("skating_best.model")), 1e-4, 2.0, 4000, true, 0);
    let lock = Arc::new(Mutex::new(0));
    let mut threads = vec![];
    for _i in 0..num_processes {
        let http_address = http_address.clone();
        let num_threads = num_threads.clone();
        let lock = lock.clone();
        let thread = tokio::spawn(async move {
            let mut pipeline = TrainPipeline::new(http_address, num_threads, lock);
            pipeline.train().await;
        });
        threads.push(thread);
    }
    for thread in threads {
        thread.await.unwrap();
    }
    //weight_comparation(Arc::new(net::Net::new(Some("santorini_best copy.model")).into()), 1e-4, 5., 400);
    //evaluate_with_pure_mcts(Arc::new(net::Net::new(Some("skating_best.model")).into()), 1e-4, 5.0, 400, 50000, false);
    //test()
}

pub const BATCH_SIZE: usize = 512;

const SELFPLAY_PLAYOUT: usize = 800;
const SELFPLAY_TEMP: f32 = 1.0;
const SELFPLAY_CPUCT: f32 = 2.0;

struct TrainPipeline {
    num_threads: usize,
    http_address: String,
    timestamp: Arc<Mutex<usize>>,
    net: net::NetTrain,
    data_buffer: Vec<SingleData>,
    kl_targ: f32,
    lr: f64,
    lr_multiplier: f64,
    evaluate_playout: usize,
    win_rate: f32,
}

impl TrainPipeline {
    fn new(http_address: String, num_threads: usize, lock: Arc<Mutex<usize>>) -> Self {
        Self {
            num_threads,
            http_address,
            timestamp: lock,
            net: net::NetTrain::new(if std::path::Path::new("latest.model").exists() {
                Some("latest.model")
            } else {
                None
            }),
            data_buffer: Vec::new(),
            kl_targ: 0.0005,
            lr: 0.002,
            lr_multiplier: 1.0 ,
            evaluate_playout: 3000,
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

    fn collect_data(&mut self, games: usize, max_length: usize, batch: usize) -> usize {
        let mut threads = vec![];
        let len = Arc::new(AtomicUsize::new(0));
        let datas = Arc::new(RwLock::new(vec![]));
        let played = Arc::new(AtomicUsize::new(0));
        let progess: Vec<Arc<AtomicUsize>> = (0..games)
            .map(|_| Arc::new(AtomicUsize::new(0)))
            .collect();
        for i in 0..self.num_threads {
            let datas = datas.clone();
            let net = if i == 0  {self.net.net.clone()} else {net::NetTrain::new(if std::path::Path::new("latest.model").exists() {
                Some("latest.model")
            } else {
                None
            }).net.clone()};
            let games = games;
            let played = played.clone();
            let len = len.clone();
            let progress = progess.clone();
            let t = std::thread::Builder::new()
                .name(format!("selfplay {}", i))
                .spawn(move || {
                    loop {
                        let played_games = played.load(Ordering::SeqCst);
                        if played_games >= games {
                            break;
                        }
                        played.fetch_add(1, Ordering::SeqCst);
                        let (_, data) = start_self_play(
                            progress[played_games].clone(),
                            net.clone(),
                            SELFPLAY_TEMP,
                            SELFPLAY_CPUCT,
                            SELFPLAY_PLAYOUT,
                            4
                        );
                        let mut dva = data;
                        if dva.len() > max_length {
                            dva.reverse();
                            dva.truncate(max_length);
                        }
                        //send_data(dva.clone(), tx.clone().unwrap());
                        len.fetch_add(dva.len(), Ordering::SeqCst);
                        /*dva.iter_mut().for_each(|data| {
                            data.weight = data.weight.powf((50 - batch as i32).max(0) as f32 * 4. / 50.)
                        });*/
                        let equi_data = get_equi_data(dva);
                        //send_data(equi_data.clone(), tx.clone().unwrap());
                        datas.write().unwrap().extend(equi_data);
                    }
                })
                .unwrap();
            threads.push(t);
        }
        loop {
            let progress: usize = progess.iter().map(|p| p.load(Ordering::SeqCst)).sum();
            print!("Batch running {}% ", progress * 100 / 150 / games);
            let bar_length = progress * 50 / 150 / games;
            print!("[{}{}]", "=".repeat(bar_length), " ".repeat(50 - bar_length));
            io::stdout().flush().unwrap();
            if progress >= 150 * games {
                break;
            }
            thread::sleep(Duration::from_millis(1000));
            print!("\r");
            print!("\x1B[K");
        }
        println!();
        self.extend(datas.read().unwrap().clone());
        len.load(Ordering::SeqCst)
    }

    fn train_step(&mut self) {
        for _ in 0..1 {
            let len = self.data_buffer.len();
            let (mut kl_t, mut p_loss_total, mut v_loss_total, mut var_old_t, mut var_new_t) =
                (0., 0., 0., 0., 0.);
            let batches = len / BATCH_SIZE;
            //let mut cloned_buffer: Vec<SingleData> = self.data_buffer.clone().into();
            //cloned_buffer.shuffle(&mut rand::thread_rng());
            self.data_buffer.shuffle(&mut rand::thread_rng());
            for i in 0..batches {
                /*let probs = self.data_buffer.iter().map(|d|1./self.data_buffer.len() as f32);
                let w_i = WeightedIndex::new(probs).unwrap();
                let mut rng = rand::thread_rng();
                let mut mini_batch = HashSet::new();
                while mini_batch.len() < BATCH_SIZE {
                    mini_batch.insert(w_i.sample(&mut rng));
                }*/
                //println!("{:?}", has_duplicate_values(mini_batch.clone()));
                /*let mini_batch =
                sample(&mut rand::thread_rng(), self.data_buffer.len(), BATCH_SIZE).into_vec();*/
                let (mut state, mut prob, mut win) = (vec![], vec![], vec![]);
                for index in i * BATCH_SIZE..(i + 1) * BATCH_SIZE {
                    let (a, b, c) = &self.data_buffer.remove(0).get_state();
                    state.extend(a.clone().into_raw_vec());
                    prob.extend(b);
                    win.push(*c);
                }
                let mut kl = 0.;
                let (oldp, oldv) =
                    self.net
                        .evaluate_batch(state.clone(), prob.clone(), win.clone());
                let value_output: Vec<f64> = oldv
                    .view_(&[512])
                    .iter::<f64>()
                    .unwrap()
                    .map(f64::from)
                    .collect();
                let mut str = value_output
                    .iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<String>>()
                    .join(",");
                str = format!("output{}", str);
                let mut ploss = 0.;
                let mut vloss = 0.;
                let mut newp;
                let mut newv = Tensor::from(0.0);

                (ploss, vloss) = self.net.train_step(
                    state.clone(),
                    prob.clone(),
                    win.clone(),
                    self.lr * self.lr_multiplier,
                );
                p_loss_total += ploss;
                v_loss_total += vloss;
                (newp, newv) = self
                    .net
                    .evaluate_batch(state.clone(), prob.clone(), win.clone());
                let logdiff = oldp
                    .g_add(&Tensor::from(1e-10).to_device(Device::Cuda(0)))
                    .log()
                    .subtract(
                        &newp
                            .g_add(&Tensor::from(1e-10).to_device(Device::Cuda(0)))
                            .log(),
                    );
                let x: Option<&[i64]> = Some(&[1]);
                kl = (&oldp)
                    .multiply(&logdiff)
                    .sum_dim_intlist(x, false, tch::Kind::Float)
                    .mean(tch::Kind::Float)
                    .try_into().unwrap();
                if kl > self.kl_targ * 4. {
                    //early stopping if D_KL diverges badly
                    self.lr_multiplier /= 1.5;
                }
                if kl > self.kl_targ * 2. && self.lr_multiplier > 0.03 {
                    self.lr_multiplier /= 1.5;
                }
                if kl < self.kl_targ / 2. && self.lr_multiplier < 10. {
                    self.lr_multiplier *= 1.5;
                }

                /*for index in mini_batch {
                    self.data_buffer[index].loss = ploss + vloss;
                    self.data_buffer[index].weight *= 0.65
                }*/
                //self.lr_multiplier *= 0.995;
                //self.lr_multiplier = self.lr_multiplier.max(0.05);
                //println!("kl {}", kl)\
                let batchtensor =
                    Tensor::from_slice(win.clone().as_slice()).to_device(Device::Cuda(0));
                let batchvar: f32 = batchtensor.var(true).try_into().unwrap();
                let oldvar: f32 = batchtensor
                    .subtract(&oldv.reshape(&[BATCH_SIZE.try_into().unwrap()]))
                    .var(true)
                    .try_into().unwrap();
                let newvar: f32 = batchtensor
                    .subtract(&newv.reshape(&[BATCH_SIZE.try_into().unwrap()]))
                    .var(true)
                    .try_into().unwrap();
                let value_variance_old: f32 = 1.0 - oldvar / batchvar;
                let value_variance_new: f32 = 1.0 - newvar / batchvar;
                kl_t += kl;
                var_new_t += value_variance_new;
                var_old_t += value_variance_old;
                println!("kl: {}, lr_ratio: {:.3}, policy_loss:{:.6}, value_loss:{:.6}, var_old: {:.3}, var_new: {:.3}", kl, self.lr_multiplier, ploss, vloss, value_variance_old, value_variance_new)
            }
            //println!("batch trained, kl: {}, policy_loss:{:.6}, value_loss:{:.6}, var_old: {:.3}, var_new: {:.3}", kl_t / batches as f32, p_loss_total / batches as f32, v_loss_total / batches as f32, var_old_t / batches as f32, var_new_t / batches as f32)
        }
    }

    async fn train(&mut self) {
        let mut batch: usize = 0;
        loop {
            batch += 1;
            //let len = self.collect_data(3, max(10, batch / 10), batch);
            {
                let mut timestamp = self.timestamp.lock().await;
                let client = Client::new();
                let res = client.get(&format!("{}/getmodel", self.http_address))
                    .body(timestamp.to_string())
                    .send()
                    .await
                    .unwrap();

                match res.status().as_u16() {
                    200 => {
                        // remove the old model
                        let _ = std::fs::remove_file("latest.model");
                        let mut file = std::fs::File::create("latest.model").unwrap();
                        let content = res.bytes().await.unwrap();
                        copy(&mut content.as_ref(), &mut file).unwrap();
                        println!("model updated");
                        self.net = net::NetTrain::new(Some("latest.model"));
                    }
                    304 => {
                        println!("model already up to date");
                    }
                    _ => {
                        println!("error: {}", res.status().as_u16());
                    }
                }

                let res = client.get(&format!("{}/gettimestamp", self.http_address))
                    .send()
                    .await
                    .unwrap();

                let content = res.text().await.unwrap();
                println!("timestamp: {}", content);
                *timestamp = content.parse().unwrap();
            }
            let len = self.collect_data(8, 99999, batch);
            println!(
                "batch {}, episode_len:{}, buffer_len:{}",
                batch,
                len,
                self.data_buffer.len()
            );
            if self.data_buffer.len() >= 0 {
                //self.train_step();
                self.send_data_to_server().await;
                self.data_buffer.clear();
            }
        }
    }

    async fn send_data_to_server(&mut self) {
        loop {
            let client = Client::new();
            let data = serde_json::to_string(&self.data_buffer).unwrap();
            let resp = client.post(&format!("{}/train", self.http_address))
                .body(data)
                .send()
                .await;
            match resp {
                Ok(resp) => {
                    if resp.status().as_u16() == 200 {
                        println!("data sent to server");
                        break;
                    } else {
                        println!("error: {}", resp.status().as_u16());
                    }
                }
                Err(e) => {
                    println!("error: {}", e);
                }
            }
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
        let state = Array3::from_shape_vec((9, 17, 17), self.state.0.clone()).unwrap();
        (state, self.state.1.clone(), self.state.2)
    }
    
}

fn start_self_play(
    progress: Arc<AtomicUsize>,
    net: Arc<RwLock<net::Net>>,
    temp: f32,
    c_puct: f32,
    n_playout: usize,
    thread: usize,
) -> (i8, Vec<SingleData>) {
    let mut board = Board::new();
    let mut i: f32 = 0.0;
    board.init(rand::thread_rng().gen_range::<u16, u16, u16>(1, 3));
    let mut player = MCTSPlayer::new(net.clone(), c_puct, n_playout, true, 1);
    let (mut states, mut mcts_probs, mut current_players): (
        Vec<Array3<f32>>,
        Vec<Vec<f32>>,
        Vec<i8>,
    ) = (vec![], vec![], vec![]);
    loop {
        let r_temp = if i < 8.0 {
            temp - i / 16.0 * temp
        } else {
            0.5 * temp
        };
        let (move_, move_probs) = player.get_action(&mut board, temp, true, thread);
        i += 1.0;
        states.push(board.current_state());
        mcts_probs.push(move_probs);
        current_players.push(board.current_player().try_into().unwrap());
        //println!("move:{} status:{}", move_, board.status);
        board.do_move(move_.try_into().unwrap(), true, true);
        progress.fetch_add(1, Ordering::SeqCst);
        let (end, winner) = board.game_end();
        if end {
            progress.store(150, Ordering::SeqCst);
            let mut winner_z = vec![0.0; current_players.len()];
            let mut i = 0;
            while i < current_players.len() {
                winner_z[i] = if winner == current_players[i] {
                    1.0
                } else {
                    if winner == -1 {
                        0.0
                    } else {
                        -1.0
                    }
                };
                i += 1;
            }
            //let net = net.clone();
            let owo = states
                .into_iter()
                .zip(mcts_probs.into_iter())
                .zip(winner_z.into_iter())
                .enumerate()
                .map(|(i, ((a, b), c))| {
                    //let net = net.read().unwrap();
                    //let (p_loss, v_loss) = net.policy_value_loss(a.clone().into_raw_vec(), b.clone(), c);
                    SingleData {
                        state: (a.into_raw_vec(), b, c),
                        loss: 0.,
                        weight: (i as f32 + 2.).log10(),
                    }
                })
                .collect();
            return (winner, owo);
        }
    }
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

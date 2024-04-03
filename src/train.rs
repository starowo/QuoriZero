use ndarray::{prelude::*, IxDynImpl};
use ndarray::{Array3, ArrayView3, ArrayViewMut3, Axis};
use pyo3::types::PyModule;
use pyo3::Python;
use rand::distributions::WeightedIndex;
use rand::prelude::Distribution;
use rand::seq::index::sample;
use rand::seq::SliceRandom;
use rand::Rng;
use std::cmp::max;
use std::collections::{HashSet, VecDeque};
use std::hash::Hash;
use std::io;
use std::io::Write;
use std::ops::Add;
use std::sync::atomic::{AtomicI16, AtomicI32, AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{mpsc, RwLock};
use std::time::Duration;
use std::{sync::Arc, thread};
use tch::{Device, Tensor};
use websocket::OwnedMessage;

use super::mcts_pure_parallel;

use super::game::Board;
use super::mcts_a0::MCTSPlayer;
use super::net::{self, Net};

pub fn train(tx: Option<mpsc::Sender<OwnedMessage>>) {
    //humanplay(net::Net::new(Some("skating_best.model")), 1e-4, 2., 4000, true, 1, 1);
    //ai_suggestion(net::Net::new(Some("skating_best.model")), 1e-4, 2.0, 4000, true, 0);
    TrainPipeline::new(tx).train();
    //weight_comparation(Arc::new(net::Net::new(Some("santorini_best copy.model")).into()), 1e-4, 5., 400);
    //evaluate_with_pure_mcts(Arc::new(net::Net::new(Some("skating_best.model")).into()), 1e-4, 5.0, 400, 50000, false);
    test()
}

pub fn play(tx: Option<Sender<OwnedMessage>>, rx: Receiver<usize>) {
    //humanplay(net::Net::new(Some("latest.model")), 1e-4, 1.5, 800, true, 2, 2, tx.clone(), rx);
    thread::spawn(move || {
        humanplay(
            net::Net::new(Some("latest.model")),
            1e-4,
            1.5,
            800,
            true,
            2,
            2,
            tx.clone(),
            rx,
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
    tx: Option<mpsc::Sender<OwnedMessage>>,
    net: net::NetTrain,
    data_buffer: Vec<SingleData>,
    kl_targ: f32,
    lr: f64,
    lr_multiplier: f64,
    evaluate_playout: usize,
    win_rate: f32,
}

impl TrainPipeline {
    fn new(tx: Option<mpsc::Sender<OwnedMessage>>) -> Self {
        Self {
            tx,
            net: net::NetTrain::new(if std::path::Path::new("latest.model").exists() {
                Some("latest.model")
            } else {
                None
            }),
            data_buffer: Vec::new(),
            kl_targ: 0.0005,
            lr: 0.002,
            lr_multiplier: 1.0 * 1.5 ,
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
        println!("generating data");
        let datas = Arc::new(RwLock::new(vec![]));
        let played = Arc::new(AtomicUsize::new(0));
        for i in 0..2 {
            let datas = datas.clone();
            let net = if i == 0  {self.net.net.clone()} else {net::NetTrain::new(Some("latest.model")).net.clone()};
            let games = games;
            let played = played.clone();
            let len = len.clone();
            let tx = self.tx.clone();
            let t = std::thread::Builder::new()
                .name(format!("selfplay {}", i))
                .spawn(move || {
                    while played.load(Ordering::SeqCst) < games {
                        played.fetch_add(1, Ordering::SeqCst);
                        let (_, data) = start_self_play(
                            net.clone(),
                            SELFPLAY_TEMP,
                            SELFPLAY_CPUCT,
                            SELFPLAY_PLAYOUT,
                            4,
                            if i == 0 { tx.clone() } else { None },
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
                        print!("-");
                        io::stdout().flush().unwrap()
                    }
                })
                .unwrap();
            threads.push(t);
        }
        for t in threads {
            t.join().unwrap();
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
                    let (a, b, c) = &self.data_buffer.remove(0).state;
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
                let _ = self.tx.clone().unwrap().send(OwnedMessage::Text(str));
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
                    .into();
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
                    Tensor::of_slice(win.clone().as_slice()).to_device(Device::Cuda(0));
                let batchvar: f32 = batchtensor.var(true).into();
                let oldvar: f32 = batchtensor
                    .subtract(&oldv.reshape(&[BATCH_SIZE.try_into().unwrap()]))
                    .var(true)
                    .into();
                let newvar: f32 = batchtensor
                    .subtract(&newv.reshape(&[BATCH_SIZE.try_into().unwrap()]))
                    .var(true)
                    .into();
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

    fn train(&mut self) {
        let mut batch: usize = 571;
        loop {
            batch += 1;
            //let len = self.collect_data(3, max(10, batch / 10), batch);
            let len = self.collect_data(4, 99999, batch);
            println!(
                "batch {}, episode_len:{}, buffer_len:{}",
                batch,
                len,
                self.data_buffer.len()
            );
            if self.data_buffer.len() >= BATCH_SIZE {
                self.train_step();
            }
            if batch % 1 == 0 {
                self.net.save("latest.model");
            }
            if batch % 50 == 0 {
                {
                    let wr =
                        weight_comparation(self.net.net.clone(), 1e-4, 4.0, 800, self.tx.clone());
                    println!("winrate: {:.3}", wr);
                    if wr > 0.55 {
                        println!("new best!");
                        self.net.save("best.model");
                    }
                }
            }
        }
    }
}
fn has_duplicate_values(vec: Vec<usize>) -> bool {
    let set: HashSet<_> = vec.iter().collect();
    vec.len() != set.len()
}
#[derive(Clone)]
struct SingleData {
    state: (Array3<f32>, Vec<f32>, f32),
    loss: f32,
    weight: f32,
}

fn ai_suggestion<'a>(
    net: net::Net,
    temp: f32,
    c_puct: f32,
    n_playout: usize,
    a0: bool,
    start: u16,
) {
    /*
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let mut board = Board::new();
            board.init(start);
            let mut input_1 = String::new();
            println!("Please enter board string:");
            io::stdin().read_line(&mut input_1)
                .expect("Failed to read line");

            let states: Vec<&str> = input_1.trim().split(";").collect();
            if states.len() != 4 {
                panic!("invalid board string");
            }
            let board_state = states[0];
            let p1: Vec<&str> = states[1].split(",").collect();
            let p2: Vec<&str> = states[2].split(",").collect();
            let p1_1 = p1[0].parse::<i16>().unwrap();
            let p1_2 = p1[1].parse::<i16>().unwrap();
            let p2_1 = p2[0].parse::<i16>().unwrap();
            let p2_2 = p2[1].parse::<i16>().unwrap();
            let phase = states[3].parse::<u16>().unwrap();
            for i in 0..25 {
                let c = board_state.chars().nth(i).unwrap();
                board.states[i] = c.to_digit(10).unwrap() as u16;
            }
            board.p1_1 = p1_1;
            board.p1_2 = p1_2;
            board.p2_1 = p2_1;
            board.p2_2 = p2_2;
            board.phase = if phase > 2 {0} else {1};
            board.status = (phase - 1) % 2 + 1;
            board.check_available();
            let graphfn = PyModule::from_code(
                py,
                r#"
    def graphic(board, players):
        """Draw the board and show game info"""
        size = 5
        player1 = 1
        player2 = 2
        #print("Player", player1, "with X".rjust(3))
        #print("Player", player2, "with O".rjust(3))
        print()
        print("    ", end='')
        for x in range(size):
            print(str(x).center(6), end='')
        print('\r\n')
        for i in range(size):
            print("{0:4d}".format(i), end='')
            for j in range(size):
                loc = i * size + j
                if players[0] != -1 and players[0] % 25 == loc:
                    print('a'.center(6), end='')
                elif players[1] != -1 and players[1] % 25 == loc:
                    print('b'.center(6), end='')
                elif players[2] != -1 and players[2] % 25 == loc:
                    print('A'.center(6), end='')
                elif players[3] != -1 and players[3] % 25 == loc:
                    print('B'.center(6), end='')
                else:
                    p = board[loc]
                    print('{}'.format(p).center(6), end='')
            print('\r\n')
    "#,
                "graph.py",
                "graph",
            )
            .unwrap()
            .getattr("graphic")
            .unwrap();
            graphfn.call1((board.states, [board.p1_1, board.p1_2, board.p2_1, board.p2_2])).expect("e");
            if a0 {
                let mut history = vec![];
                let mut player = MCTSPlayer::new(Arc::new(net.into()), c_puct, n_playout, false);
                loop {
                    history.push(board.clone());
                    let current = board.current_player();
                    if current == 1 || current == 2 {
                        let (move_, nums) = player.get_action(&mut board, temp, true, 6);
                        /*let mut map = nums
                            .iter()
                            .enumerate()
                            .map(|(k, v)| (k, v))
                            .collect::<Vec<(usize, &f32)>>();
                        map.sort_by(|(_, b), (_, a)| a.partial_cmp(b).unwrap());
                        let mut i = 0;
                        while i < 3 && i < map.len() {
                            let (move_, value) = map[i];
                            let (x, y) = Board::move_to_loc(move_ % 25);
                            println!("move: {},{},{} prob:{}", x, y, move_ / 25 + 1, value);
                            i += 1
                        }*/

                        let mut input_1 = String::new();
                        let mut input_3 = String::new();

                        println!("Please enter coordinate:");
                        io::stdin().read_line(&mut input_1)
                            .expect("Failed to read line");

                        if input_1.contains("undo") {
                            history.pop();
                            board = history.pop().unwrap();
                            player.mcts.update_with_move(-1, false);
                            continue;
                        }

                        // 将输入的字符串转换为数字类型
                        let num1 = input_1.trim().chars().nth(0).unwrap() as i32 - 'a' as i32;
                        let num2 = input_1.trim().chars().nth(1).unwrap() as i32 - '1' as i32;

                        let id = if board.phase != 2 {
                            println!("Please enter id:");
                            io::stdin().read_line(&mut input_3)
                                .expect("Failed to read line");
                            let num: i32 = input_3.trim().parse()
                                .expect("Please enter a valid number");
                            num - 1
                        }else {
                            2
                        };

                        let p = num1 + num2 * 5 + 25 * id;
                        if board.do_move(p.try_into().unwrap(), true) {
                            player.mcts.update_with_move(p, false);
                        }
                        graphfn.call1((board.states, [board.p1_1, board.p1_2, board.p2_1, board.p2_2])).expect("e");
                    }
                    let (end, winner) = board.game_end();
                    if end {
                        println!("winner is {}", winner);
                        return;
                    }
                }
            }
        });
         */
}

fn external_suggestion<'a>(
    net: net::Net,
    temp: f32,
    c_puct: f32,
    n_playout: usize,
    a0: bool,
    start: u16,
) {
    /*
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let mut board = Board::new();
            board.init(start);
            let graphfn = PyModule::from_code(
                py,
                r#"
    def graphic(board, players):
        """Draw the board and show game info"""
        size = 5
        player1 = 1
        player2 = 2
        #print("Player", player1, "with X".rjust(3))
        #print("Player", player2, "with O".rjust(3))
        print()
        print("    ", end='')
        for x in range(size):
            print(('a', 'b', 'c', 'd', 'e')[x].center(6), end='')
        print('\r\n')
        for i in range(size):
            print("{0:4d}".format(i), end='')
            for j in range(size):
                loc = i * size + j
                if players[0] != -1 and players[0] % 25 == loc:
                    print('a'.center(6), end='')
                elif players[1] != -1 and players[1] % 25 == loc:
                    print('b'.center(6), end='')
                elif players[2] != -1 and players[2] % 25 == loc:
                    print('A'.center(6), end='')
                elif players[3] != -1 and players[3] % 25 == loc:
                    print('B'.center(6), end='')
                else:
                    p = board[loc]
                    print('{}'.format(p).center(6), end='')
            print('\r\n')
    "#,
                "graph.py",
                "graph",
            )
            .unwrap()
            .getattr("graphic")
            .unwrap();
            graphfn.call1((board.states, [board.p1_1, board.p1_2, board.p2_1, board.p2_2])).expect("e");
            if a0 {
                let mut player = MCTSPlayer::new(Arc::new(net.into()), c_puct, n_playout, false);
                loop {
                    let current = board.current_player();
                    if current == 1 || current == 2 {
                        /*let mut map = nums
                            .iter()
                            .enumerate()
                            .map(|(k, v)| (k, v))
                            .collect::<Vec<(usize, &f32)>>();
                        map.sort_by(|(_, b), (_, a)| a.partial_cmp(b).unwrap());
                        let mut i = 0;
                        while i < 3 && i < map.len() {
                            let (move_, value) = map[i];
                            let (x, y) = Board::move_to_loc(move_ % 25);
                            println!("move: {},{},{} prob:{}", x, y, move_ / 25 + 1, value);
                            i += 1
                        }*/

                        let mut input_1 = String::new();
                        println!("Please enter board string:");
                        io::stdin().read_line(&mut input_1)
                            .expect("Failed to read line");

                        let states: Vec<&str> = input_1.trim().split(";").collect();
                        if states.len() != 4 {
                            println!("invalid board string");
                            continue;
                        }
                        let board_state = states[0];
                        let p1: Vec<&str> = states[1].split(",").collect();
                        let p2: Vec<&str> = states[2].split(",").collect();
                        let p1_1 = p1[0].parse::<i16>().unwrap();
                        let p1_2 = p1[1].parse::<i16>().unwrap();
                        let p2_1 = p2[0].parse::<i16>().unwrap();
                        let p2_2 = p2[1].parse::<i16>().unwrap();
                        let phase = states[3].parse::<u16>().unwrap();
                        for i in 0..25 {
                            let c = board_state.chars().nth(i).unwrap();
                            board.states[i] = c.to_digit(10).unwrap() as u16;
                        }
                        board.p1_1 = p1_1;
                        board.p1_2 = p1_2;
                        board.p2_1 = p2_1;
                        board.p2_2 = p2_2;
                        board.phase = if phase > 2 {0} else {1};
                        board.status = (phase - 1) % 2 + 1;
                        board.check_available();
                        let (move_, nums) = player.get_action(&mut board, temp, true, 6);
                        board.do_move(move_.try_into().unwrap(), true);
                        if board.phase != 0 {
                            player.mcts.update_with_move(move_, false);
                            player.get_action(&mut board, temp, true, 6);
                        }
                        player.mcts.update_with_move(-1, false);
                        graphfn.call1((board.states, [board.p1_1, board.p1_2, board.p2_1, board.p2_2])).expect("e");
                    }
                    let (end, winner) = board.game_end();
                    if end {
                        println!("winner is {}", winner);
                        return;
                    }
                }
            }
        });
        */
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
        let mut player = MCTSPlayer::new(Arc::new(net.into()), c_puct, n_playout, false, 2);
        let mut history = vec![];
        loop {
            history.push(board.clone());
            let current = board.current_player();
            if current == player_ {
                let (move_, nums) = player.get_action(&mut board, temp, true, 6);
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
                board.do_move(p.try_into().unwrap(), true, false);
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
                        let mut player = MCTSPlayer::new(n, c_puct, n_playout_a0, false, 1);
                        let mut best = MCTSPlayer::new(b, c_puct, n_playout_a0, false, 1);
                        let mut turn = 0;
                        loop {
                            let current = board.current_player();
                            if current == 1 {
                                let (move_, nums) = player.get_action(&mut board, temp, true, 1);
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
                                let (move_, nums) = best.get_action(&mut board, temp, true, 1);
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
                            if turn >= 60 && !end {
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

fn evaluate_with_pure_mcts_parallel(
    net: Arc<RwLock<Net>>,
    temp: f32,
    c_puct: f32,
    n_playout_a0: usize,
    n_playout_pure: usize,
    tx: Option<mpsc::Sender<OwnedMessage>>,
) -> f32 {
    pyo3::prepare_freethreaded_python();
    let a = Arc::new(AtomicI32::new(0));
    let mut threads = vec![];
    //let time = std::time::SystemTime::now();
    let played = Arc::new(AtomicUsize::new(0));
    for i in 0..1 {
        let a1 = a.clone();
        let net = net.clone();
        let played = played.clone();
        let tx = tx.clone();
        let t = std::thread::Builder::new()
            .name(format!("thread {}", i))
            .spawn(move || {
                let mut count = played.load(Ordering::SeqCst);
                while count < 5 {
                    played.fetch_add(1, Ordering::SeqCst);
                    let n = net.clone();
                    let mut board = Board::new(tx.clone());
                    board.init((count % 2 + 1).try_into().unwrap());
                    if true {
                        let mut player = MCTSPlayer::new(n, c_puct, n_playout_a0, false, 1);
                        let mut pure = super::mcts_pure_parallel::MCTSPlayer::new(
                            c_puct,
                            n_playout_pure,
                            false,
                        );
                        loop {
                            let current = board.current_player();
                            if current == 1 {
                                let (move_, nums) = player.get_action(&mut board, temp, true, 1);
                                let mut map = nums
                                    .iter()
                                    .enumerate()
                                    .map(|(k, v)| (k, v))
                                    .collect::<Vec<(usize, &f32)>>();
                                map.sort_by(|(_, b), (_, a)| a.partial_cmp(b).unwrap());
                                board.do_move(move_.try_into().unwrap(), true, true);
                                player.mcts.update_with_move(move_, false);
                            } else {
                                let (move_, nums) = pure.get_action(&mut board, temp, true);
                                let mut map = nums
                                    .iter()
                                    .enumerate()
                                    .map(|(k, v)| (k, v))
                                    .collect::<Vec<(usize, &f32)>>();
                                map.sort_by(|(_, b), (_, a)| a.partial_cmp(b).unwrap());
                                board.do_move(move_.try_into().unwrap(), true, true);
                                player.mcts.update_with_move(move_, false);
                            }
                            let (end, winner) = board.game_end();
                            if end {
                                if winner == 1 {
                                    a1.fetch_add(2, std::sync::atomic::Ordering::SeqCst);
                                    println!("winner is alpha0")
                                } else {
                                    if winner == 2 {
                                        println!("winner is puremcts")
                                    } else {
                                        a1.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                                        println!("draw game")
                                    }
                                }
                                break;
                            }
                        }
                    }
                    count = played.load(Ordering::SeqCst);
                }
            })
            .unwrap();
        threads.push(t);
    }
    for t in threads {
        t.join().unwrap();
    }
    a.load(std::sync::atomic::Ordering::SeqCst) as f32 / 30.
}

/*
fn evaluate_with_pure_mcts(net: Arc<RwLock<Net>>, temp: f32, c_puct: f32, n_playout_a0: usize, n_playout_pure: usize, graph: bool) -> f32 {
    pyo3::prepare_freethreaded_python();
    let a = Arc::new(AtomicI32::new(0));
    let a1 = a.clone();
    Python::with_gil(|py| {
        let graphfn = PyModule::from_code(
            py,
            r#"
def graphic(board, players):
    """Draw the board and show game info"""
    size = 5
    player1 = 1
    player2 = 2
    #print("Player", player1, "with X".rjust(3))
    #print("Player", player2, "with O".rjust(3))
    print()
    print("    ", end='')
    for x in range(size):
        print(str(x).center(6), end='')
    print('\r\n')
    for i in range(size):
        print("{0:4d}".format(i), end='')
        for j in range(size):
            loc = i * size + j
            if players[0] != -1 and players[0] % 25 == loc:
                print('a'.center(6), end='')
            elif players[1] != -1 and players[1] % 25 == loc:
                print('b'.center(6), end='')
            elif players[2] != -1 and players[2] % 25 == loc:
                print('A'.center(6), end='')
            elif players[3] != -1 and players[3] % 25 == loc:
                print('B'.center(6), end='')
            else:
                p = board[loc]
                print('{}'.format(p).center(6), end='')
        print('\r\n')
"#,
            "graph.py",
            "graph",
        )
        .unwrap()
        .getattr("graphic")
        .unwrap();
        for i in 0..32 {
            let n = net.clone();
            let mut board = Board::new();
            board.init(i % 2);
            if graph {
                graphfn.call1((board.states, [board.p1_1, board.p1_2, board.p2_1, board.p2_2])).expect("e");
            }
            if true {
                let mut player = MCTSPlayer::new(n, c_puct, n_playout_a0, false);
                let mut pure = super::mcts_pure_parallel::MCTSPlayer::new(c_puct, n_playout_pure, false);
                loop {
                    let current = board.current_player();
                    if current == 1 {
                        let (move_, nums) = player.get_action(&mut board, temp, true, 2);
                        let mut map = nums
                            .iter()
                            .enumerate()
                            .map(|(k, v)| (k, v))
                            .collect::<Vec<(usize, &f32)>>();
                        map.sort_by(|(_, b), (_, a)| a.partial_cmp(b).unwrap());
                        board.do_move(move_.try_into().unwrap(), true);
                        if graph {
                            graphfn.call1((board.states, [board.p1_1, board.p1_2, board.p2_1, board.p2_2])).expect("e");
                        }
                    } else {
                        let (move_, nums) = pure.get_action(&mut board, temp, true);
                        let mut map = nums
                            .iter()
                            .enumerate()
                            .map(|(k, v)| (k, v))
                            .collect::<Vec<(usize, &f32)>>();
                        map.sort_by(|(_, b), (_, a)| a.partial_cmp(b).unwrap());
                        board.do_move(move_.try_into().unwrap(), true);
                        if graph {
                            graphfn.call1((board.states, [board.p1_1, board.p1_2, board.p2_1, board.p2_2])).expect("e");
                        }
                    }
                    let (end, winner) = board.game_end();
                    if end {
                        if winner == 1 {
                            a1.fetch_add(2, std::sync::atomic::Ordering::SeqCst);
                            println!("winner is alpha0")
                        }else {
                            if winner == 2 {
                                println!("winner is puremcts")
                            }else {
                                a1.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                                println!("draw game")
                            }
                        }
                        break;
                    }
                }
            }
        }
    });
    a.load(std::sync::atomic::Ordering::SeqCst) as f32 / 64.
}
 */

fn start_self_play(
    net: Arc<RwLock<net::Net>>,
    temp: f32,
    c_puct: f32,
    n_playout: usize,
    thread: usize,
    tx: Option<Sender<OwnedMessage>>,
) -> (i8, Vec<SingleData>) {
    let mut board = Board::new(tx);
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
        let (end, winner) = board.game_end();
        if end {
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
                        state: (a, b, c),
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
        let arr = v.state.1;
        result.push(into_data(
            (flipud_planes(&v.state.0), flipud_actions(&arr), v.state.2),
            v.weight,
        ));
        result.push(into_data(
            (fliplr_planes(&v.state.0), fliplr_actions(&arr), v.state.2),
            v.weight,
        ));
        result.push(into_data(
            (
                flipud_planes(&fliplr_planes(&v.state.0)),
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
        state,
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

use ndarray::prelude::*;
use rand::distributions::{Dirichlet, WeightedIndex};
use rand::prelude::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock, Weak};
use std::time::Duration;
use uuid::Uuid;

use crate::net::NetTrain;

use super::net::Net;
use super::game::Board;

const C_LOSS: f32 = 3.;

// 定义一个 Rust 结构体，表示要调用的 Python 类

#[derive(Clone)]
struct TreeNode {
    id: Uuid,
    parent: Option<Weak<RwLock<TreeNode>>>,
    children: HashMap<i32, Arc<RwLock<TreeNode>>>,
    n_visits: i32,
    q: f32,
    p: f32,
    vloss: f32,
    applied_noise: bool,
    ally: bool,
}

impl TreeNode {
    fn new(parent: Option<Weak<RwLock<TreeNode>>>, prior_p: f32, ally: bool) -> TreeNode {
        let id = Uuid::new_v4();
        TreeNode {
            id,
            parent,
            children: HashMap::new(),
            n_visits: 0,
            q: 0.0,
            p: prior_p,
            vloss: 0.0,
            applied_noise: false,
            ally,
        }
    }

    fn update_recursive(&mut self, leaf_value: f32) -> Option<Weak<RwLock<TreeNode>>> {
        match self.parent.as_ref() {
            Some(p) => {
                self.vloss -= 1.;
                self.n_visits += 1;
                self.q += 1.0 * (leaf_value - self.q) / self.n_visits as f32;
                return Some(p.clone());
            }
            None => {
                self.n_visits += 1;
                self.q += 1.0 * (leaf_value - self.q) / self.n_visits as f32;
                return None;
            }
        }
    }

    fn get_value(&self, u: f32) -> f32 {
        if self.vloss > 0. && self.n_visits > 0 {
            (self.q * self.n_visits as f32 - self.vloss * C_LOSS) / self.n_visits as f32 + u
        }else {
            self.q - self.vloss * C_LOSS + u
        }
    }

    fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }
}

fn softmax(x: Vec<f32>) -> Vec<f32> {
    let arr = Array::from_iter(&x);
    let mut max = &x[0];
    for num in &x {
        if num > max {
            max = num;
        }
    }
    let exp_x = arr.mapv(|f| -> f32 { (f - max).exp() });
    let sum_exp_x = exp_x.sum();
    let exp_x = exp_x.mapv(|f| -> f32 { f / sum_exp_x });
    exp_x.to_vec()
}

fn apply_noise(node: &mut TreeNode, alpha: f64, epsilon: f32) {
    let len = node.children.len();
    let mut rng = rand::thread_rng();
    let dirichlet = Dirichlet::new(vec![alpha; len]);
    let noise = dirichlet.sample(&mut rng);
    let mut i = 0;
    for (_action, child) in node.children.iter() {
        let mut c = child.write().unwrap();
        c.p = c.p * (1. - epsilon) + noise[i] as f32 * epsilon;
        i += 1;
    }
    node.applied_noise = true;
}

fn expand(node_rc: Arc<RwLock<TreeNode>>, action_priors: &Vec<(i32, f32)>, selfplay: bool, ally: bool) {
    let mut node = node_rc.write().unwrap();
    //println!("{} writing {} at point 1", std::thread::current().name().unwrap(), node.id);
    // alpha of dirichlet noise; related to sensible actions num
    // for 24 actions, 0.8 is appropriate
    let alpha = 0.03 * 19.*19. / action_priors.len() as f64;
    let len = action_priors.len();
        for (action, prob) in action_priors.iter() {
            if !node.children.contains_key(action) {
                let new = TreeNode::new(Some(Arc::downgrade(&node_rc.clone())), *prob, ally);
                node.children.insert(*action, Arc::new(new.into()));
            }
        }
        if selfplay && len > 1 && node.parent.is_none() && !node.applied_noise {
            apply_noise(&mut node, alpha, 0.25)
        }

    //println!("{} released {} at point 1", std::thread::current().name().unwrap(), node.id);
}
fn update_recursive(node: Arc<RwLock<TreeNode>>, leaf_value: f32) {
    let mut leaf_value = leaf_value;
    let (mut n, ally) = {
        let mut write = node.write().unwrap();
        (write.update_recursive(leaf_value), write.ally)
    };
    if !ally {
        leaf_value = -leaf_value;
    }
    while n.is_some() {
        let (nn, ally) = {
            let arc = n.unwrap().upgrade().unwrap();
            let mut write: std::sync::RwLockWriteGuard<TreeNode> = arc.write().unwrap();
            (write.update_recursive(leaf_value), write.ally)
        };
        n = nn;
        if !ally {
            leaf_value = -leaf_value;
        }
    }
    /*
    let mut n = node;
    let mut v = leaf_value;
    while let Some(parent) = {
        let x = n.read().unwrap();
        x.parent.clone()
    } {
        let nc = n.clone();
        let mut nr = nc.write().unwrap();

        //println!("{} writing {} at point 2", std::thread::current().name().unwrap(), nr.id);
        if nr.ally {
            nr.update(v);
            n = parent.clone();
        } else {
            nr.update(v);
            n = parent.clone();
            v = -v;
        }

        //println!("{} released {} at point 2", std::thread::current().name().unwrap(), nr.id);
    } */
}
fn visit(parent: &TreeNode, node: &TreeNode, c_puct: f32) -> f32 {
    //let binding = node.parent.clone().unwrap();
    //let parent = binding.read().unwrap();
    let forced_playout = (2. * node.p * parent.n_visits as f32).sqrt().round() as i32;
    if node.n_visits < forced_playout && parent.parent.is_none() {
        return 10000.0;
    }
    c_puct * node.p * (parent.n_visits as f32).sqrt() / (1.0 + node.n_visits as f32)
}
fn select(node: &TreeNode, c_puct: f32) -> (i32, Arc<RwLock<TreeNode>>) {
    let nd = node;
    let mut u: HashMap<&i32, f32> = HashMap::new();
    for child in nd.children.iter() {
        /*let n = &mut self.nodes[*child.1];
        let parent = &mut self.nodes[n.parent.unwrap()];
        n.u = c_puct * n.p * (parent.n_visits as f32).sqrt()
            / (1.0 + n.n_visits as f32);*/
        let c = child.1.read().unwrap();
        //println!("{} reading {} at point 1", std::thread::current().name().unwrap(), c.id);
        u.insert(child.0, visit(nd, &c, c_puct));
        //println!("{} read released {} at point 1", std::thread::current().name().unwrap(), c.id);
    }
    let r = nd
        .children
        .iter()
        .max_by(|(action, node1), (ac2, node2)| {
            let o = {
                let n1 = node1.read().unwrap();
                let n2 = node2.read().unwrap();
                n1.get_value(*u.get(*action).unwrap())
                    .partial_cmp(&n2.get_value(*u.get(*ac2).unwrap()))
            };
            if o.is_none() {
                let u1 = *u.get(*action).unwrap();
                let n1 = node1.read().unwrap();
                let v1 = n1.get_value(u1);
                let u2 = *u.get(*ac2).unwrap();
                let n2 = node2.read().unwrap();
                let v2 = n2.get_value(u2);
                println!("{}, {}, {}, {}", u1, v1, u2, v2);
                panic!("booom!")
            }
            o.unwrap()
        })
        .unwrap();
    (*r.0, r.1.clone())
}

fn playout(
    node: Arc<RwLock<TreeNode>>,
    c_puct: f32,
    net: Arc<RwLock<Net>>,
    expanding: Arc<RwLock<Vec<Uuid>>>,
    state: &mut Board,
    selfplay: bool
) {
    let mut node = node;
    loop {
        let selection = {
            loop {
                let has = {
                    let nd = node.read().unwrap();
                    let exp = expanding.read().unwrap();
                    exp.contains(&nd.id)
                };
                if has {
                    std::thread::sleep(Duration::from_millis(1));
                } else {
                    break;
                }
            }
            let nd = node.read().unwrap();

            //println!("{} reading {} at point 2", std::thread::current().name().unwrap(), nd.id);
            if nd.is_leaf() {
                let mut exp = expanding.write().unwrap();
                exp.push(nd.id);
                break;
            }
            let r = select(&nd, c_puct);

            //println!("{} read released {} at point 2", std::thread::current().name().unwrap(), nd.id);
            r
        };
        let (action, next_node) = (selection.0, selection.1);
        {
            let mut nd = next_node.write().unwrap();

            //println!("{} writing {} at point 3", std::thread::current().name().unwrap(), nd.id);
            nd.vloss += 1.;

            //println!("{} released {} at point 3", std::thread::current().name().unwrap(), nd.id);
        }
        if !state.available.contains(&(action as u16)) {
            /*
            let nodes = nodes_arc.read().unwrap();
            let mut track = vec![];
            let mut node = nodes.get(&next_node).unwrap();
            track.push(node);
            let mut parent = node.parent;
            while parent.is_some() {
                node = nodes.get(&parent.unwrap()).unwrap();
                track.push(node);
                parent = node.parent;
            }
            panic!("??");
             */
        }
        //println!("simulate move:{} status:{}", action, state.status);
        state.do_move(action as u16, next_node.read().unwrap().is_leaf(), false);
        //println!("{:?}", state.available);
        node = next_node;
    }
    //println!("{:?}", state.available);
    let (action_probs, mut leaf_value) = {
        let net = net.write()
            .unwrap();
        net.policy_value(state.available.clone(), state.current_state(), false)
    };
    let (end, winner) = state.game_end();
    if !end {
        expand(
            node.clone(),
            &action_probs,
            selfplay,
            state.tiles[0] == state.tiles[1],
        );
        //node.expand(&action_probs, state.status <= 4 && state.status % 2 == 0);
    } else {
        if winner == -1 {
            leaf_value = 0.0;
        } else {
            leaf_value = if winner == state.current_player() as i8 {1.0} else {-1.0};
        }
    }
    update_recursive(node.clone(), -leaf_value);
    let id = {
        let nd = node.read().unwrap();
        if nd.parent.is_none() {
            //println!("root nn out: {}", leaf_value);
        }
        nd.id
    };
    {
        let mut exp = expanding.write().unwrap();
        exp.retain(|x| *x != id);
    }
    //node.update_recursive(-leaf_value);
}

pub(crate) struct MCTS {
    nets: Vec<Arc<RwLock<Net>>>,
    root: Arc<RwLock<TreeNode>>,
    c_puct: f32,
    n_playout: usize,
}

impl MCTS {
    fn new(net: Arc<RwLock<Net>>, c_puct: f32, n_playout: usize, num_nets: usize, path: &str) -> MCTS {
        let mut nets = vec![net];
        for _ in 1..num_nets {
            let cloned_net = NetTrain::new(if std::path::Path::new(path).exists() {
                Some(path)
            } else {
                None
            }).net.clone();
            nets.push(cloned_net);
        }
        let n = MCTS {
            nets,
            root: Arc::new(TreeNode::new(None, 1.0, false).into()),
            c_puct,
            n_playout,
        };
        n
    }
    /*
    fn expand(&self, node: &usize, action_priors: &Vec<(i32, f32)>, ally: bool) {
        for (action, prob) in action_priors.iter() {
            let mut next = 0;
            {
                let nodes = self.nodes.read().unwrap();
                while nodes.contains_key(&next) {
                    next += 1;
                }
            }
            let mut nodes = self.nodes.write().unwrap();
            let node = nodes.get_mut(node).unwrap();
            if !node.children.contains_key(action) {
                let new = TreeNode::new(next, Some(node.index), *prob, ally);
                node.children.insert(*action, next);
                nodes.insert(next, new);
            }
        }
    }
    fn update_recursive(&self, node: &usize, leaf_value: f32) {
        let mut nodes = self.nodes.write().unwrap();
        let mut n = nodes.get_mut(node).unwrap();
        let mut v = leaf_value;
        while let Some(parent) = n.parent {
            if n.ally {
                n.update(v);
                n = nodes.get_mut(&parent).unwrap();
            } else {
                n.update(v);
                n = nodes.get_mut(&parent).unwrap();
                v = -v;
            }
        }
    }
    fn visit(&self, node: &usize, c_puct: f32) -> f32 {
        let nodes = self.nodes.read().unwrap();
        let node = nodes.get(node).unwrap();
        let parent = nodes.get(&node.parent.unwrap()).unwrap();
        c_puct * node.p * (parent.n_visits as f32).sqrt() / (1.0 + node.n_visits as f32)
    }
    fn select(&self, node: &usize, c_puct: f32) -> (i32, usize) {
        let nodes = self.nodes.read().unwrap();
        let nd = nodes.get(node).unwrap();
        let mut u: HashMap<&usize, f32> = HashMap::new();
        for child in nd.children.iter() {
            /*let n = &mut self.nodes[*child.1];
            let parent = &mut self.nodes[n.parent.unwrap()];
            n.u = c_puct * n.p * (parent.n_visits as f32).sqrt()
                / (1.0 + n.n_visits as f32);*/
            u.insert(child.1, self.visit(child.1, c_puct));
        }
        let r = nd.children
            .iter()
            .max_by(|(_, node1), (_, node2)| {
                (nodes
                    .get(*node1)
                    .unwrap()
                    .get_value(*u.get(*node1).unwrap()))
                .partial_cmp(
                    &nodes
                        .get(*node2)
                        .unwrap()
                        .get_value(*u.get(*node2).unwrap()),
                )
                .unwrap()
            })
            .unwrap();
        (*r.0, *r.1)
    }
    fn playout(&self, state: &mut Board) {
        let mut node = 0;
        {
            let nodes = self.nodes.read().unwrap();
            node = nodes.get(&self.root.read().unwrap()).unwrap().index;
            //let mut node = &mut self.nodes[self.root];

            loop {
                if nodes.get(&node).unwrap().is_leaf() {
                    break;
                }
                let selection = self.select(&node, self.c_puct);
                let (action, next_node) = (selection.0, selection.1);
                //println!("simulate move:{} status:{}", action, state.status);
                state.do_move(action as u16, true);
                node = next_node;
            }
        }
        /*
        let (action_probs, _) = MCTS::policy(state);
        let (end, _) = state.game_end();
        if !end {
            self.expand(&node, &action_probs, state.status <= 4 && state.status % 2 == 0);
        }
        let leaf_value = MCTS::_evaluate_rollout(state);
        self.update_recursive(&node, -leaf_value);
        let mut node = &mut self.root; */
        //let arr = state.current_state().into_pyarray(self.py);
        let (action_probs, mut leaf_value) =
            self.net.read().unwrap()
                .policy_value(state.available.clone(), state.current_state(), false);
        let (end, winner) = state.game_end();
        if !end {
            self.expand(
                &node,
                &action_probs,
                false, //state.status <= 4 && state.status % 2 == 0,
            );
            //node.expand(&action_probs, state.status <= 4 && state.status % 2 == 0);
        } else {
            if winner == -1 {
                leaf_value = 0.0;
            } else {
                leaf_value = -1.0;
            }
        }
        self.update_recursive(&node, -leaf_value);
        //node.update_recursive(-leaf_value);
    }
    */
    fn get_move_probs(&self, state: &mut Board, temp: f32, selfplay: bool, thread: usize) -> (Vec<i32>, Vec<f32>, Vec<f32>) {
        let mut threads = vec![];
        let n_playout = if state.walls[0] == 0 && state.walls[1] == 0 && state.available.len() == 1 {
            5
        } else {
            self.n_playout
        };
        let expanding: Arc<RwLock<Vec<Uuid>>> = Arc::new(RwLock::new(vec![]));
        //let time = std::time::SystemTime::now();
        let played = Arc::new(AtomicUsize::new(0));
        for i in 0..thread {
            let root = self.root.clone();
            let net = self.nets[i % self.nets.len()].clone();
            let c_puct = self.c_puct;
            let state_copy = state.clone();
            let played = played.clone();
            let expanding = expanding.clone();
            let t = std::thread::Builder::new()
                .name(format!("thread {}", i))
                .spawn(move || {
                    let root = root;
                    while played.load(Ordering::SeqCst) < n_playout {
                        let mut state = state_copy.clone();
                        played.fetch_add(1, Ordering::SeqCst);
                        playout(
                            root.clone(),
                            c_puct,
                            net.clone(),
                            expanding.clone(),
                            &mut state,
                            selfplay
                        );
                    }
                    /*
                        for _ in 0..arc1.n_playout {
                            let mut state_copy = state.clone();
                            arc1.playout(&mut state_copy);
                        }
                    } */
                })
                .unwrap();
            threads.push(t);
        }
        for t in threads {
            t.join().unwrap();
        }
        //let tm = std::time::SystemTime::now().duration_since(time).unwrap().as_millis();
        //println!("{}v/s", n_playout as f32 / tm as f32*1000.0);
        let visits_most = self.root.read().unwrap().children.iter()
            .map(|(_, node)| node.read().unwrap().n_visits)
            .max()
            .unwrap();
        let root = self.root.read().unwrap();
        let act_visits = root
            .children
            .iter()
            .map(|(&action, node)| {
                let mut visits = node.read().unwrap().n_visits;
                if visits < visits_most {
                    let forced_playout = (2. * node.read().unwrap().p * root.n_visits as f32).sqrt();
                    visits -= forced_playout.round() as i32;
                }
                (action, visits)
            })    
            .collect::<Vec<_>>();
        let (actions, visits) = act_visits
            .iter()
            .map(|(action, visits)| (*action, *visits))
            .unzip::<i32, i32, Vec<i32>, Vec<i32>>();
        let v1 = visits
            .iter()
            .map(|s| -> f32 { 1.0 / temp * (*s as f32 + 1e-10).ln() })
            .collect::<Vec<f32>>();
        let v2 = visits
            .iter()
            .map(|s| -> f32 { 1.0 / 1.0 * (*s as f32 + 1e-10).ln() })
            .collect::<Vec<f32>>();
        let act_probs = softmax(v1);
        (actions, act_probs, softmax(v2))
        /*
        for _ in 0..self.n_playout {
            let mut state_copy = state.clone();
            self.playout(&mut state_copy);
        }
        let act_visits = self.root.children.iter()
            .map(|(&action, node)| (action, node.n_visits))
            .collect::<Vec<_>>();
        let (actions, visits) = act_visits.iter()
            .map(|(action, visits)| (*action, *visits))
            .unzip::<i32, i32, Vec<i32>, Vec<i32>>();
        let v1 = visits.iter().map(|s| -> f32 {
            1.0 / temp * (*s as f32 + 1e-10).log10()
        }).collect::<Vec<f32>>();
        let act_probs = softmax(v1);
        (actions, act_probs) */
    }

    pub fn update_with_move(&mut self, last_move: i32, selfplay: bool) {
        let has = { self.root.read().unwrap().children.contains_key(&last_move) };
        if has {
            let node = self
                .root
                .read()
                .unwrap()
                .children
                .get(&last_move)
                .unwrap()
                .clone();
            self.root = node;
            let mut root = self.root.write().unwrap();
            root.parent = None;
            if selfplay && root.children.len() > 1 && !root.applied_noise {
                let alpha = 0.03 * 19.*19. / root.children.len() as f64;
                apply_noise(&mut root, alpha, 0.25)
            }
            /*
            let n = nodes.get_mut(&node).unwrap();
            n.parent = None;
            let mut exist = vec![];
            let mut stack = vec![node];
            while let Some(node_idx) = stack.pop() {
                exist.push(node_idx);
                let node = nodes.get(&node_idx).unwrap();
                for child_idx in &node.children {
                    stack.push(*child_idx.1);
                }
            }
            nodes.retain(|ind, _| exist.contains(ind));
             */
        } else {
            self.root = Arc::new(TreeNode::new(None, 1.0, false).into());
            /*let mut map = HashMap::new();
            map.insert(0, TreeNode::new(0, None, 1.0, false));
            self.nodes = Arc::new(map.into());*/
        }
    }
}

pub(crate) struct MCTSPlayer {
    pub mcts: MCTS,
    is_selfplay: bool,
}

impl MCTSPlayer {
    pub fn new(
        net: Arc<RwLock<Net>>,
        c_puct: f32,
        n_playout: usize,
        is_selfplay: bool,
        num_nets: usize,
        path: &str,
    ) -> MCTSPlayer {
        MCTSPlayer {
            mcts: MCTS::new(net, c_puct, n_playout, num_nets, path),
            is_selfplay,
        }
    }

    pub fn get_action(
        &mut self,
        board: &mut Board,
        temp: f32,
        _return_prob: bool,
        thread: usize
    ) -> (i32, Vec<f32>) {
        let sensible_moves = &mut board.available;
        // the pi vector returned by MCTS as in the alphaGo Zero paper
        let mut move_probs = vec![0.0; 132];
        if !sensible_moves.is_empty() {
            let (acts, probs, origin_probs) = self.mcts.get_move_probs(board, temp, self.is_selfplay, thread);
            /*if !self.is_selfplay && self.mcts.n_playout >= 1000 {
                loop {
                    let mut input_1 = String::new();
                    println!("winrate: {}", -self.mcts.root.read().unwrap().q);
                    let mut move_probs = vec![0.0; 24];
                    let mut i = 0;
                    while i < acts.len() {
                        move_probs[acts[i] as usize] = origin_probs[i];
                        i += 1;
                    }
                    let mut map = move_probs
                    .iter()
                    .enumerate()
                    .map(|(k, v)| (k, v))
                    .collect::<Vec<(usize, &f32)>>();
                    map.sort_by(|(_, b), (_, a)| a.partial_cmp(b).unwrap());
                    let mut i = 0;
                    while i < 3 && i < map.len() {
                        let (move_, value) = map[i];
                        let (x, y) = (0,0);
                        let chars = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'];
                        println!("move: {},{} ({}{}) prob:{}", x, y, chars[x], y+1, value);
                        i += 1
                    }
                    println!("keep searching? y(yes) or other(no)");
                    std::io::stdin().read_line(&mut input_1)
                        .expect("Failed to read line");
                    if input_1.contains("y") {
                        (acts, probs, origin_probs) = self.mcts.get_move_probs(board, temp, self.is_selfplay, thread);
                        continue;
                    }
                    break;
                }
            }*/
            let mut i = 0;
            while i < acts.len() {
                move_probs[acts[i] as usize] = probs[i];
                i += 1;
            }
            if self.is_selfplay {
                // add Dirichlet Noise for exploration (needed for
                // self-play training) ----- Deprecated
                // now the noise is added in the expand function
                let mut rng = rand::thread_rng();
                let move_idx = probs
                    .iter()
                    .enumerate()
                    .map(|(idx, &prob)| (acts[idx], prob))
                    .collect::<Vec<(i32, f32)>>();

                let chosen_idx = match WeightedIndex::new(
                    &move_idx.iter().map(|(_, prob)| *prob).collect::<Vec<f32>>(),
                ) {
                    Ok(w) => w.sample(&mut rng),
                    Err(_) => panic!("Error: invalid probability vector"),
                };
                let move_ = move_idx[chosen_idx].0 as i32;
                // update the root node and reuse the search tree
                self.mcts.update_with_move(move_, true);
                (move_, move_probs)
            } else {
                // with the default temp=1e-3, it is almost equivalent
                // to choosing the move with the highest prob
                let mut rng = rand::thread_rng();
                let chosen_idx = match WeightedIndex::new(&probs) {
                    Ok(w) => w.sample(&mut rng),
                    Err(_) => panic!("Error: invalid probability vector"),
                };
                let move_ = acts[chosen_idx];
                //println!("winrate: {}", self.mcts.root.read().unwrap().q);
                // reset the root node
                //self.mcts.update_with_move(-1, false);
                if _return_prob {
                    let mut move_probs = vec![0.0; 132];
                    let mut i = 0;
                    while i < acts.len() {
                        move_probs[acts[i] as usize] = origin_probs[i];
                        i += 1;
                    }
                    (move_, move_probs)
                } else {
                    (move_, move_probs)
                }
            }
        } else {
            println!("WARNING: the board is full");
            (-1, vec![])
        }
    }
}

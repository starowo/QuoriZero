use ndarray::prelude::*;
use rand::distributions::{Dirichlet, WeightedIndex};
use rand::prelude::*;
use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::atomic::{AtomicI32, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock, Weak};
use std::time::Duration;
use uuid::Uuid;

use super::net::Net;
use super::game::Board;

// 定义一个 Rust 结构体，表示要调用的 Python 类

#[derive(Clone)]
struct TreeNode {
    id: Uuid,
    id_debug: String,
    parent: Option<Weak<RwLock<TreeNode>>>,
    children: HashMap<i32, Arc<RwLock<TreeNode>>>,
    n_visits: i32,
    q: f32,
    p: f32,
    ally: bool,
}

impl TreeNode {
    fn new(parent: Option<Weak<RwLock<TreeNode>>>, prior_p: f32, ally: bool) -> TreeNode {
        let id = Uuid::new_v4();
        let id_debug = id.to_string();
        TreeNode {
            id,
            id_debug,
            parent,
            children: HashMap::new(),
            n_visits: 0,
            q: 0.0,
            p: prior_p,
            ally,
        }
    }

    fn update_recursive(&mut self, leaf_value: f32) -> Option<Weak<RwLock<TreeNode>>> {
        match self.parent.as_ref() {
            Some(p) => {
                self.q = self.q * self.n_visits as f32 + leaf_value + 3.0;
                //println!("{} -2, {}", self.id, std::thread::current().name().unwrap());
                self.n_visits -= 2;
                self.q = self.q / self.n_visits as f32;
                return Some(p.clone());
            }
            None => {
                self.n_visits += 1;
                self.q += 1.0 * (leaf_value - self.q) / self.n_visits as f32;
                return None;
            }
        }
    }

    fn update(&mut self, leaf_value: f32) {
        self.q = self.q * self.n_visits as f32 + leaf_value + 3.0;
        //println!("{} -2, {}", self.id, std::thread::current().name().unwrap());
        self.n_visits -= 2;
        self.q = self.q / self.n_visits as f32;
        //self.n_visits += 1;
        //self.q += 1.0 * (leaf_value - self.q) / self.n_visits as f32;
    }

    fn get_value(&self, u: f32) -> f32 {
        self.q + u
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

fn expand(node_rc: Arc<RwLock<TreeNode>>, action_priors: &Vec<(i32, f32)>, ally: bool) {
    let mut node = node_rc.write().unwrap();
    //println!("{} writing {} at point 1", std::thread::current().name().unwrap(), node.id);
    for (action, prob) in action_priors.iter() {
        if !node.children.contains_key(action) {
            let new = TreeNode::new(Some(Arc::downgrade(&node_rc.clone())), *prob, ally);
            node.children.insert(*action, Arc::new(new.into()));
        }
    }

    //println!("{} released {} at point 1", std::thread::current().name().unwrap(), node.id);
}
fn update_recursive(node: Arc<RwLock<TreeNode>>, leaf_value: f32) {
    let mut leaf_value = leaf_value;
    let mut n = {
        let mut write = node.write().unwrap();
        write.update_recursive(leaf_value)
    };
    leaf_value = -leaf_value;
    while n.is_some() {
        n = {
            let arc = n.unwrap().upgrade().unwrap();
            let mut write: std::sync::RwLockWriteGuard<TreeNode> = arc.write().unwrap();
            write.update_recursive(leaf_value)
        };
        leaf_value = -leaf_value;
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
    expanding: Arc<RwLock<Vec<Uuid>>>,
    state: &mut Board,
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
            nd.q = nd.q * nd.n_visits as f32 - 3.0;
            //println!("{} +3, {}", nd.id, std::thread::current().name().unwrap());
            nd.n_visits += 3;
            nd.q = nd.q / nd.n_visits as f32;

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
        state.do_move(action as u16, true, false);
        //println!("{:?}", state.available);
        node = next_node;
    }
    //println!("{:?}", state.available);
    let (action_probs, _) = policy(state);
    let (end, _) = state.game_end();
    if !end {
        expand(
            node.clone(),
            &action_probs,
            false, //state.status <= 4 && state.status % 2 == 0,
        );
        //node.expand(&action_probs, state.status <= 4 && state.status % 2 == 0);
    }
    
    let leaf_value = _evaluate_rollout(state);
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
fn policy(state: &Board) -> (Vec<(i32, f32)>, f32) {
    let mut probs = vec![];
    let available = &state.available;
    let mut i = 0;
    while i < available.len() {
        probs.push((available[i] as i32, 1.0 / available.len() as f32));
        i += 1;
    }
    (probs, 0.0)
}
fn rollout_policy(state: &Board) -> Vec<(i32, f32)> {
    let mut rng = rand::thread_rng();
    (0..state.available.len())
        .map(|i| (state.available[i] as i32, rng.gen::<f32>()))
        .collect()
}
fn _evaluate_rollout(state: &mut Board) -> f32 {
    let player = state.current_player();
    let mut i = 0;
    let mut winner = -1;
    while i < 1000 {
        let (end, winner1) = state.game_end();
        winner = winner1;
        if end {
            break;
        }
        let action_probs = rollout_policy(state);
        let max = action_probs
            .iter()
            .max_by(|(_, p1), (_, p2)| p1.total_cmp(p2))
            .unwrap()
            .0;
        state.do_move(max.try_into().unwrap(), true, false);
        i += 1;
    }
    if winner > -1 {
        return if winner == player as i8 { 1.0 } else { -1.0 };
    }
    0.0
}
struct MCTS {
    root: Arc<RwLock<TreeNode>>,
    c_puct: f32,
    n_playout: usize,
}

impl MCTS {
    fn new(c_puct: f32, n_playout: usize) -> MCTS {
        let n = MCTS {
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
    fn get_move_probs(&self, state: &mut Board, temp: f32) -> (Vec<i32>, Vec<f32>, Vec<f32>) {
        let mut threads = vec![];
        let n_playout = self.n_playout;
        let expanding: Arc<RwLock<Vec<Uuid>>> = Arc::new(RwLock::new(vec![]));
        //let time = std::time::SystemTime::now();
        let played = Arc::new(AtomicUsize::new(0));
        for i in 0..16 {
            let root = self.root.clone();
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
                            expanding.clone(),
                            &mut state,
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
        let root = self.root.read().unwrap();
        let act_visits = root
            .children
            .iter()
            .map(|(&action, node)| (action, node.read().unwrap().n_visits))
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

    fn update_with_move(&mut self, last_move: i32) {
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
            self.root.write().unwrap().parent = None;
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
    mcts: MCTS,
    is_selfplay: bool,
}

impl MCTSPlayer {
    pub fn new(
        c_puct: f32,
        n_playout: usize,
        is_selfplay: bool,
    ) -> MCTSPlayer {
        MCTSPlayer {
            mcts: MCTS::new(c_puct, n_playout),
            is_selfplay,
        }
    }

    pub fn get_action(
        &mut self,
        board: &mut Board,
        temp: f32,
        _return_prob: bool,
    ) -> (i32, Vec<f32>) {
        let sensible_moves = &mut board.available;
        // the pi vector returned by MCTS as in the alphaGo Zero paper
        let mut move_probs = vec![0.0; 132];
        if !sensible_moves.is_empty() {
            let (acts, probs, origin_probs) = self.mcts.get_move_probs(board, temp);
            let mut i = 0;
            while i < acts.len() {
                move_probs[acts[i] as usize] = probs[i];
                i += 1;
            }
            if false {
                // add Dirichlet Noise for exploration (needed for
                // self-play training)
                let mut rng = rand::thread_rng();
                let noisy_probs = probs.iter().map(|f| *f as f64 * 0.75).collect::<Vec<f64>>();
                let len = probs.len();
                let dirichlet = Dirichlet::new(vec![1.0; if len > 1 { len } else { 2 }]);
                let noise = dirichlet.sample(&mut rng);

                let move_idx = noisy_probs
                    .iter()
                    .enumerate()
                    .map(|(idx, &prob)| (acts[idx], prob + noise[idx] * 0.25))
                    .collect::<Vec<(i32, f64)>>();

                let chosen_idx = match WeightedIndex::new(
                    &move_idx.iter().map(|(_, prob)| *prob).collect::<Vec<f64>>(),
                ) {
                    Ok(w) => w.sample(&mut rng),
                    Err(_) => panic!("Error: invalid probability vector"),
                };
                let move_ = move_idx[chosen_idx].0 as i32;
                // update the root node and reuse the search tree
                self.mcts.update_with_move(move_);
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
                self.mcts.update_with_move(-1);
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

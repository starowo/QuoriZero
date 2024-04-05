use std::collections::HashMap;

use ndarray::{Array, Array3};



#[derive(Clone)]
pub(crate) struct Board {
    pub tx: Option<std::sync::mpsc::Sender<websocket::OwnedMessage>>,
    pub status: u16,
    pub start: u16,
    pub available: Vec<u16>,
    pub tiles: [u16; 2],
    pub walls: [u16; 2],
    pub shortest_paths: [Vec<u16>; 2],
    pub last_pos: u16,
    // board: 17*17
    pub state: [[u8; 17]; 17],
}

impl Board {
    pub fn new(
        tx: Option<std::sync::mpsc::Sender<websocket::OwnedMessage>>,
    ) -> Board {
        let state = [[0; 17]; 17];
        let tiles = [4*9, 4*9+8];
        let walls = [10, 10];
        Board {
            tx,
            status: 0,
            start: 0,
            available: Vec::new(),
            tiles,
            walls,
            shortest_paths: [Vec::new(), Vec::new()],
            last_pos: 0,
            state,
        }
    }

    pub fn current_state(&self) -> Array3<f32> {
        let mut square_state = Array::zeros((9, 17, 17));
        for i in 0..17 {
            for j in 0..17 {
                if self.state[i][j] == self.status as u8 {
                    square_state[[0, i, j]] = 1.0;
                    square_state[[1, i, j]] = 1.0;
                } else if self.state[i][j] == 3 - self.status as u8 {
                    square_state[[0, i, j]] = 1.0;
                    square_state[[2, i, j]] = 1.0;
                }
            }
        }
        for i in 0..2 {
            let x = self.tiles[i] as usize % 9 * 2;
            let y = self.tiles[i] as usize / 9 * 2;
            let z = if self.status as usize == i+1 {3} else {4};
            square_state[[z, x, y]] = 1.0;
        }
        let path1 = self.shortest_paths[0].clone();
        let path2 = self.shortest_paths[1].clone();

        for i in 0..path1.len() {
            let x = path1[i] as usize % 9 * 2;
            let y = path1[i] as usize / 9 * 2;
            square_state[[if self.status == 1 {5} else {6}, x, y]] = 1.0;
        }

        for i in 0..path2.len() {
            let x = path2[i] as usize % 9 * 2;
            let y = path2[i] as usize / 9 * 2;
            square_state[[if self.status == 2 {5} else {6}, x, y]] = 1.0;
        }

        for i in 0..self.walls[0] as usize {
            for j in 0..8 {
                square_state[[7, j, i]] = 1.0;
            }
        }
        for i in 0..self.walls[1] as usize {
            for j in 9..17 {
                square_state[[7, j, i]] = 1.0;
            }
        }
        
        if self.status == 1 {
            for i in 0..17 {
                square_state[[8, 16, i]] = 1.0;
            }
        } else {
            for i in 0..17 {
                square_state[[8, 0, i]] = 1.0;
            }
        }
        square_state
    }

    pub fn get_state_string(&self) -> String {
        let mut s = String::new();
        s.push_str("state");
        for i in 0..17 {
            for j in 0..17 {
                s.push_str(&self.state[j][i].to_string());
            }
        }
        s.push_str(";");
        for i in 0..2 {
            s.push_str(&self.tiles[i].to_string());
            s.push_str(";");
        }
        s.push_str(&self.status.to_string());
        s
    }

    pub fn init(&mut self, start_player: u16) {
        //println!("{}", start_player);
        self.status = start_player;
        self.start = self.status;
        self.check_available();
        if self.tx.is_some() {
            let _ = self.tx.as_ref().unwrap().send(websocket::OwnedMessage::Text(self.get_state_string()));
        }
    }
    fn check_available(&mut self) {
        self.available.clear();
        let tile = self.tiles[self.status as usize - 1];
        let opp_tile = self.tiles[2 - self.status as usize];
        let moves = self.available_positions_a(tile, opp_tile, true, true, true, true, true);
        let shortest_path = self.bfs(tile, opp_tile, if self.status == 1 {8} else {0});
        let opp_shortest_path = self.bfs(opp_tile, tile, if self.status == 1 {0} else {8});
        self.shortest_paths[2 - self.status as usize] = opp_shortest_path.clone();
        self.shortest_paths[self.status as usize - 1] = shortest_path.clone();
        if tile == opp_tile {
            for i in 128..132 {
                let location = self.move_to_location(i);
                if moves.contains(&location) && self.last_pos != location {
                    self.available.push(i);
                }
            }
            return;
        }
        if self.walls[0] == 0 && self.walls[1] == 0 {
            for i in 128..132 {
                let location = self.move_to_location(i);
                if moves.contains(&location) && shortest_path.contains(&location) {
                    self.available.push(i);
                }
            }
            if self.available.len() == 0{
                for i in 128..132 {
                    let location = self.move_to_location(i);
                    if moves.contains(&location) {
                        self.available.push(i);
                    }
                }
            }
            return;
        } else {
            for i in 128..132 {
                let location = self.move_to_location(i);
                if moves.contains(&location) {
                    self.available.push(i);
                }
            } 
        }
        
        /*
        for i in 128..132 {
            let location = self.move_to_location(i);
            if moves.contains(&location) {
                self.available.push(i);
            }
        }  */
        
        if self.walls[self.status as usize - 1] > 0 {
            for i in 0..8 {
                for j in 0..8 {
                    if self.wall_valid(i, j, true) {
                        self.available.push(i + j * 8);
                    }
                    if self.wall_valid(i, j, false) {
                        self.available.push(i + j * 8 + 8*8);
                    }
                }
            }
        }
        
    }

    fn available_positions_a(&self, tile: u16, opp_tile: u16, up: bool, right: bool, down: bool, left: bool, include_opp: bool) -> Vec<u16> {
        let mut positions = Vec::new();
        let x = tile as usize % 9 * 2;
        let y = tile as usize / 9 * 2;
        if up && y > 0 {
            if self.state[x][y - 1] == 0 {
                if tile - 9 == opp_tile {
                    let positions_b = self.available_positions(tile - 9, opp_tile, true, true, false, true);
                    if positions_b.len() > 0 {
                        let jump = tile - 18;
                        if y > 2 && positions_b.contains(&jump) {
                            positions.push(jump);
                        } else {
                            if include_opp {
                                positions.push(opp_tile);
                            }
                        }
                    }
                } else {
                    positions.push(tile - 9);
                }
            }
        }
        if right && x < 16 {
            if self.state[x + 1][y] == 0 {
                if tile + 1 == opp_tile {
                    let positions_b = self.available_positions(tile + 1, opp_tile, true, true, true, false);
                    if positions_b.len() > 0 {
                        let jump = tile + 2;
                        if x < 14 && positions_b.contains(&jump) {
                            positions.push(jump);
                        } else {
                            if include_opp {
                                positions.push(opp_tile);
                            }
                        }
                    }
                } else {
                    positions.push(tile + 1);
                }
            }
        }
        if down && y < 16 {
            if self.state[x][y + 1] == 0 {
                if tile + 9 == opp_tile {
                    let positions_b = self.available_positions(tile + 9, opp_tile, false, true, true, true);
                    if positions_b.len() > 0 {
                        let jump = tile + 18;
                        if y < 14 && positions_b.contains(&jump) {
                            positions.push(jump);
                        } else {
                            if include_opp {
                                positions.push(opp_tile);
                            }
                        }
                        
                    }
                } else {
                    positions.push(tile + 9);
                }
            }
        }
        if left && x > 0 {
            if self.state[x - 1][y] == 0 {
                if tile - 1 == opp_tile {
                    let positions_b = self.available_positions(tile - 1, opp_tile, true, false, true, true);
                    if positions_b.len() > 0 {
                        let jump = tile - 2;
                        if x > 2 && positions_b.contains(&jump) {
                            positions.push(jump);
                        } else {
                            if include_opp {
                                positions.push(opp_tile);
                            }
                        }
                    }
                } else {
                    positions.push(tile - 1);
                }
            }
        }
        positions
    }

    fn available_positions(&self, tile: u16, opp_tile: u16, up: bool, right: bool, down: bool, left: bool) -> Vec<u16> {
        let include_opp = false;
        let mut positions = Vec::new();
        let x = tile as usize % 9 * 2;
        let y = tile as usize / 9 * 2;
        if up && y > 0 {
            if self.state[x][y - 1] == 0 {
                if tile - 9 == opp_tile {
                    let positions_b = self.available_positions(tile - 9, opp_tile, true, true, false, true);
                    if positions_b.len() > 0 {
                        let jump = tile - 18;
                        if y > 2 && positions_b.contains(&jump) {
                            positions.push(jump);
                        } else {
                            positions.extend(positions_b);
                        }
                        if include_opp {
                            positions.push(opp_tile);
                        }
                    }
                } else {
                    positions.push(tile - 9);
                }
            }
        }
        if right && x < 16 {
            if self.state[x + 1][y] == 0 {
                if tile + 1 == opp_tile {
                    let positions_b = self.available_positions(tile + 1, opp_tile, true, true, true, false);
                    if positions_b.len() > 0 {
                        let jump = tile + 2;
                        if x < 14 && positions_b.contains(&jump) {
                            positions.push(jump);
                        } else {
                            positions.extend(positions_b);
                        }
                        if include_opp {
                            positions.push(opp_tile);
                        }
                    }
                } else {
                    positions.push(tile + 1);
                }
            }
        }
        if down && y < 16 {
            if self.state[x][y + 1] == 0 {
                if tile + 9 == opp_tile {
                    let positions_b = self.available_positions(tile + 9, opp_tile, false, true, true, true);
                    if positions_b.len() > 0 {
                        let jump = tile + 18;
                        if y < 14 && positions_b.contains(&jump) {
                            positions.push(jump);
                        } else {
                            positions.extend(positions_b);
                        }
                        if include_opp {
                            positions.push(opp_tile);
                        }
                    }
                } else {
                    positions.push(tile + 9);
                }
            }
        }
        if left && x > 0 {
            if self.state[x - 1][y] == 0 {
                if tile - 1 == opp_tile {
                    let positions_b = self.available_positions(tile - 1, opp_tile, true, false, true, true);
                    if positions_b.len() > 0 {
                        let jump = tile - 2;
                        if x > 2 && positions_b.contains(&jump) {
                            positions.push(jump);
                        } else {
                            positions.extend(positions_b);
                        }
                        if include_opp {
                            positions.push(opp_tile);
                        }
                    }
                } else {
                    positions.push(tile - 1);
                }
            }
        }
        positions
    }

    pub fn move_to_wall(id: usize) -> (usize, usize, bool) {
        let horizontal = id < 8*8;
        let id = id % 64;
        let x = id % 8 * 2 + 1;
        let y = id / 8 * 2 + 1;
        return (x, y, horizontal);
    }

    pub fn move_to_location(&self, id: u16) -> u16 {
        let tile = self.tiles[self.status as usize - 1];
        let opp_tile = self.tiles[2 - self.status as usize];
        let moves = self.available_positions_a(tile, opp_tile, true, true, true, true, true);
        let tile_x = tile % 9;
        let tile_y = tile / 9;
        if id == 8*8*2 && tile_y > 0 {
            let targ = tile_x + (tile_y-1) * 9;
            if targ == opp_tile && tile_y > 1 {
                let try_move = tile_x + (tile_y-2) * 9;
                if moves.contains(&try_move) {
                    return try_move;
                }
            }
            return targ;
        }
        if id == 8*8*2 + 1 && tile_x < 8 {
            let targ = tile_x + 1 + tile_y * 9;
            if targ == opp_tile && tile_x < 7 {
                let try_move = tile_x + 2 + tile_y * 9;
                if moves.contains(&try_move) {
                    return try_move;
                }
            }
            return targ;
        }
        if id == 8*8*2 + 2 && tile_y < 8 {
            let targ = tile_x + (tile_y+1) * 9;
            if targ == opp_tile && tile_y < 7 {
                let try_move = tile_x + (tile_y+2) * 9;
                if moves.contains(&try_move) {
                    return try_move;
                }
            }
            return targ;
        }
        if id == 8*8*2 + 3 && tile_x > 0 {
            let targ = tile_x - 1 + tile_y * 9;
            if targ == opp_tile && tile_x > 1 {
                let try_move = tile_x - 2 + tile_y * 9;
                if moves.contains(&try_move) {
                    return try_move;
                }
            }
            return targ;
        }
        tile
    }

    pub fn current_player(&self) -> u8 {
        self.status.try_into().unwrap()
    }

    pub fn do_move(&mut self, d: u16, calc_available: bool, safe: bool) {
        if safe {
            if !self.available.contains(&d) {
                return;
            }
        }
        if d < 8*8*2 {
            let (x, y, horizontal) = Board::move_to_wall(d as usize);
            self.state[x][y] = self.status as u8;
            if horizontal {
                self.state[x + 1][y] = self.status as u8;
                self.state[x - 1][y] = self.status as u8;
            } else {
                self.state[x][y + 1] = self.status as u8;
                self.state[x][y - 1] = self.status as u8;
            }
            self.walls[self.status as usize - 1] -= 1;
        } else {
            let d = self.move_to_location(d);
            self.last_pos = self.tiles[self.status as usize - 1];
            self.tiles[self.status as usize - 1] = d;
            if self.tiles[0] == self.tiles[1] {
                self.status = 3 - self.status;
            }
        }
        self.status = 3 - self.status;
        if self.tx.is_some() && safe {
            let _ = self.tx.as_ref().unwrap().send(websocket::OwnedMessage::Text(self.get_state_string()));
        }
        if calc_available {
            self.check_available();
        }
    }

    fn wall_valid(&mut self, x: u16, y: u16, horizontal: bool) -> bool {
        if x >= 8 || y >= 8 {
            return false;
        }
        let x = x as usize * 2 + 1;
        let y = y as usize * 2 + 1;
        if horizontal {
            if self.state[x][y] != 0 || self.state[x + 1][y] != 0 || self.state[x - 1][y] != 0 {
                return false;
            }
        } else {
            if self.state[x][y] != 0 || self.state[x][y + 1] != 0 || self.state[x][y - 1] != 0 {
                return false;
            }
        }
        if horizontal {
            let mut count = 0;
            if x <= 1 || self.state[x - 3][y] != 0 || self.state[x - 2][y - 1] != 0 || self.state[x - 2][y + 1] != 0 {
                count += 1;
            }
            if x >= 15 || self.state[x + 3][y] != 0 || self.state[x + 2][y - 1] != 0 || self.state[x + 2][y + 1] != 0 {
                count += 1;
            }
            if self.state[x][y - 1] != 0 || self.state[x][y + 1] != 0 {
                count += 1;
            }
            if count < 2 {
                return true;
            }
            self.state[x][y] = 1;
            self.state[x + 1][y] = 1;
            self.state[x - 1][y] = 1;
            let valid = self.has_path();
            self.state[x][y] = 0;
            self.state[x + 1][y] = 0;
            self.state[x - 1][y] = 0;
            return valid;
        } else {
            let mut count = 0;
            if y <= 1 || self.state[x][y - 3] != 0 || self.state[x - 1][y - 2] != 0 || self.state[x + 1][y - 2] != 0 {
                count += 1;
            }
            if y >= 15 || self.state[x][y + 3] != 0 || self.state[x - 1][y + 2] != 0 || self.state[x + 1][y + 2] != 0 {
                count += 1;
            }
            if self.state[x - 1][y] != 0 || self.state[x + 1][y] != 0 {
                count += 1;
            }
            if count < 2 {
                return true;
            }
            self.state[x][y] = 1;
            self.state[x][y + 1] = 1;
            self.state[x][y - 1] = 1;
            let valid = self.has_path();
            self.state[x][y] = 0;
            self.state[x][y + 1] = 0;
            self.state[x][y - 1] = 0;
            return valid;
        }
    
    }

    fn has_path(&self) -> bool {
        self.dfs(self.tiles[0], 8, &mut vec![]) && self.dfs(self.tiles[1], 0, &mut vec![])
    }

    fn dfs(&self, tile: u16, target: u16, visited: &mut Vec<u16>) -> bool {
        if visited.contains(&tile) {
            return false;
        }
        if tile % 9 == target {
            return true;
        }
        visited.push(tile);
        let next = self.available_positions(tile, tile, true, true, true, true);
        for n in next {
            if self.dfs(n, target, visited) {
                return true;
            }
        }
        false
    }

    fn bfs(&self, tile: u16, opp_tile: u16, target_x: u16) -> Vec<u16> {
        let mut queue = Vec::new();
        let mut visited = HashMap::new();
        queue.push(tile);
        visited.insert(tile, vec![]);
        while queue.len() > 0 {
            let current = queue.remove(0);
            let path = visited.get(&current).unwrap().clone();
            let next = self.available_positions_a(current, opp_tile, true, true, true, true, true);
            for n in next {
                if !visited.contains_key(&n) {
                    let mut path_n = path.clone();
                    path_n.push(n);
                    visited.insert(n, path_n);
                    queue.push(n);
                    if n % 9 == target_x {
                        return visited.get(&n).unwrap().clone();
                    }
                }
            }
        }
        vec![]
    }

    fn check_win(&self) -> (bool, i8) {
        if self.tiles[0] % 9 == 8 {
            return (true, 1);
        }
        if self.tiles[1] % 9 == 0 {
            return (true, 2);
        }
        (false, -1)
    }
    pub fn game_end(&self) -> (bool, i8) {
        let (win, winner) = self.check_win();
        if win {
            return (true, winner);
        }
        (false, -1)
    }

}

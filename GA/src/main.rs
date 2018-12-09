extern crate csv;
extern crate rand;
extern crate primal;

use rand::{Rng, thread_rng};
use std::time::SystemTime;

use std::io::{Write, Read, BufWriter, BufReader, copy};

use std::fs::File;
use std::f64;

const TIME_LIMIT: f64 = 3600.;
#[derive(Debug)]
struct TspData {
    city_ids: Vec<usize>,
    primals: Vec<usize>,
    coods: Vec<(f64, f64)>,
}

#[derive(Debug, Clone)]
struct Sol {
    score: f64,
    route: Vec<usize>,
}

impl TspData {
    fn calc_score(&self, route: &Vec<usize>) -> f64 {
        let mut dist: f64 = 0.;
        for i in 1..route.len() {
            let from_city = self.city_ids[route[i - 1]];
            let to_city = self.city_ids[route[i]];
            let from_coord = self.coods[from_city];
            let to_coord = self.coods[to_city];                        
            
            let d = distance(from_coord, to_coord);
            dist += d;
            if i % 10 == 0 && !self.primals.contains(&from_city) {
                dist += d * 0.1;
            }
        }
        dist
    }

    fn calc_dists(&self, route: &Vec<usize>) -> Vec<f64> {
        let mut dists: Vec<f64>  = Vec::new();        
        for i in 1..route.len() {
            let from_city = self.city_ids[route[i - 1]];
            let to_city = self.city_ids[route[i]];
            let from_coord = self.coods[from_city];
            let to_coord = self.coods[to_city];                        
            
            let d = distance(from_coord, to_coord);
            if i % 10 == 0 && !self.primals.contains(&from_city) {
                dists.push(d * 1.1);
            } else {
                dists.push(d);        
            }

        }
        dists
    }
    fn calc_score_diff(&self, route: &Vec<usize>, dists: &mut Vec<f64>, left: usize, right: usize) -> f64 {
        let mut diff: f64 = 0.;
        let from_city = self.city_ids[route[left - 1]];
        let curr_city = self.city_ids[route[left]];        
        let to_city = self.city_ids[route[left + 1]];
        let from_coord = self.coods[from_city];
        let curr_coord = self.coods[curr_city];        
        let to_coord = self.coods[to_city];
        let old_left1 = dists[left - 1].clone();
        let old_left2 = dists[left].clone();        
        
        dists[left - 1] = distance(from_coord, curr_coord);
        if left % 10 == 0 && !self.primals.contains(&from_city) {
            dists[left - 1] *= 1.1;
        }
        diff -= old_left1 - dists[left - 1];
        
        dists[left] = distance(curr_coord, to_coord);
        if (left + 1) % 10 == 0 && !self.primals.contains(&curr_city) {
            dists[left] *= 1.1;
        }
        diff -= old_left2 - dists[left];
        
        let from_city = self.city_ids[route[right - 1]];
        let curr_city = self.city_ids[route[right]];        
        let to_city = self.city_ids[route[right + 1]];
        let from_coord = self.coods[from_city];
        let curr_coord = self.coods[curr_city];        
        let to_coord = self.coods[to_city];
        let old_right1 = dists[right - 1].clone();
        let old_right2 = dists[right].clone();
        
        dists[right - 1] = distance(from_coord, curr_coord);
        if (right) % 10 == 0 && !self.primals.contains(&from_city) {
            dists[right - 1] *= 1.1;
        }
        diff -= old_right1 - dists[right - 1];
        
        dists[right] = distance(curr_coord, to_coord);
        if (right + 1) % 10 == 0 && !self.primals.contains(&curr_city) {
            dists[right] += dists[right] * 0.1;
        }
        diff -= old_right2 - dists[right];
        
        diff
    }    
    fn get_nearest(&self, city_id: usize, is_visited: &Vec<bool>) -> usize {
        let from_coord = self.coods[city_id];
        let mut min_dist = f64::INFINITY;
        let mut min_city: usize = 0;
        for i in &self.city_ids {
            if is_visited[*i] || self.coods[*i] == from_coord {
                continue;
            }
            let dist = distance(from_coord, self.coods[*i]);
            if dist < min_dist {
                min_dist = dist;
                min_city = *i;
            }
        }
        min_city
    }
    fn get_nearest_rev(&self, city_id: usize, is_visited: &Vec<bool>, step: &usize) -> usize {
        let from_coord = self.coods[city_id];
        let mut min_dist = f64::INFINITY;
        let mut min_city: usize = 0;
        for i in &self.city_ids {
            if is_visited[*i] || self.coods[*i] == from_coord {
                continue;
            }
            let mut dist = distance(from_coord, self.coods[*i]);
            if step % 10 == 0 {
                dist *= 1.1;
            }
            if dist < min_dist {
                min_dist = dist;
                min_city = *i;
            }
        }
        min_city
    }    
    
}

fn distance(from_: (f64, f64), to_: (f64, f64)) -> f64{
    return ((from_.0 - to_.0).powi(2) + (from_.1 - to_.1).powi(2)).sqrt();
}

fn write_route(filename: String, tsp_data: &TspData, route: &Vec<usize>) {
    let mut f = File::create(filename).unwrap();
    for i in route {
        f.write_all(format!("{}\n", *i).as_bytes()).unwrap();
    }
}

fn read_route(filename: String) -> Vec<usize> {
    let mut f = File::open(filename);
    let mut route: Vec<usize>  = Vec::new();
    let mut rdr = csv::ReaderBuilder::new().has_headers(false).from_reader(f.unwrap());
    for result in rdr.records() {
        let record = result.unwrap();
        let city: usize = record[0].parse().unwrap();
        route.push(city);
    }
    route
}


fn parse(filename: String) -> TspData {
    let mut city_ids: Vec<usize> = Vec::new();
    let mut primals: Vec<usize> = Vec::new();    
    let mut coods: Vec<(f64, f64)> = Vec::new();
    
    let file = File::open(filename);
    let mut rdr = csv::Reader::from_reader(file.unwrap());
    for result in rdr.records() {
        let record = result.unwrap();
        let id: usize = record[0].parse().unwrap();
        let x: f64 = record[1].parse().unwrap();
        let y: f64 = record[2].parse().unwrap();
        city_ids.push(id);
        coods.push((x, y));
    }

    for p in primal::Primes::all().take_while(|p| *p <= city_ids.len()) {
        primals.push(p as usize);
    }
    let data = TspData {
        city_ids: city_ids,
        primals: primals,
        coods: coods
    };
    data
}

fn greedy(tsp_data: &TspData) -> Vec<usize> {
    let mut route = vec![0];
    let mut is_visited = vec![false; tsp_data.city_ids.len()];
    is_visited[0] = true;
    let mut current_city = 0;
    while route.len() < tsp_data.city_ids.len() {
        current_city = tsp_data.get_nearest(current_city, &is_visited);
        is_visited[current_city] = true;
        route.push(current_city);
        if route.len() % 10000 == 0 {
            println!("{}", route.len());
        }
    }
    route.push(0);
    //println!("{:?}", route);
    write_route(format!("greedy_{}.csv", tsp_data.city_ids.len()), &tsp_data, &route);
    println!("score: {:?}", tsp_data.calc_score(&route));
    route
}

fn greedy_rev(tsp_data: &TspData) -> Vec<usize> {
    let mut route = vec![0];
    let mut is_visited = vec![false; tsp_data.city_ids.len()];
    is_visited[0] = true;
    let mut current_city = 0;
    let mut step = tsp_data.city_ids.len();
    
    while route.len() < tsp_data.city_ids.len() {
        current_city = tsp_data.get_nearest_rev(current_city, &is_visited, &step);
        step -= 1;
        is_visited[current_city] = true;
        route.push(current_city);
        if route.len() % 10000 == 0 {
            println!("{}", route.len());
        }
    }
    route.push(0);
    route.reverse();
    //println!("{:?}", route);
    write_route(format!("greedy_rev_{}.csv", tsp_data.city_ids.len()), &tsp_data, &route);
    println!("score: {:?}", tsp_data.calc_score(&route));
    route
}    


fn prob_bolzman(diff: &f64, t: &f64) -> f64{
    if *diff < 0. {
        1.
    } else {
        (- diff / t).exp()
    }
}


fn ga(tsp_data: &TspData, _route: &Vec<usize>, start_time: f64, max_iter: usize) {
    let mut current_sol = _route.clone();
    let mut dists = tsp_data.calc_dists(&current_sol);
    let mut current_score: f64 = dists.iter().sum(); //tsp_data.calc_score(&current_sol);

    let N = 100;
    let mut rng = rand::thread_rng();
    let mut populations: Vec<Sol> = vec![];

    let data = Sol {
            score: tsp_data.calc_score(&current_sol),
            route: current_sol.clone()
        };
    
    populations.push(data);
    let city_num: i32 = tsp_data.city_ids.len() as i32;
    for i in 0..N * 2 {
        let mut sol: Vec<usize> = vec![0];
        let mut nums: Vec<i32> = (1..city_num).collect();
        thread_rng().shuffle(&mut nums);            
        for j in nums {
            sol.push(j as usize);
        }
        sol.push(0);
        let mut data = Sol {
            score: tsp_data.calc_score(&sol),
            route: sol
        };

        populations.push(data);
    }
    
    let sa_start_time = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_secs() as f64;
    let mut comp_time = 0.;
    let mut diff = 0.;
    let mut prev_score = -1.;

    for itr in 0..max_iter {
        let now = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_secs() as f64;
        comp_time = (now - sa_start_time) + start_time;
        if comp_time > TIME_LIMIT {
            break;
        }

        populations.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());
        if itr % 1 == 0 {
            println!("iter: {}, time: {}, curr_score: {}, {}, {}", itr, comp_time, populations[0].score, populations[1].score, populations[2].score,);        
        }

        let mut new_populations: Vec<Sol> = vec![];

        for i in 0..N {
            new_populations.push(populations[i].clone());
        }
        for i in 0..N {       

            let parent1: usize = rng.gen_range(0, new_populations.len());
            let mut parent1_data = new_populations[parent1].clone();
            let parent2: usize = rng.gen_range(0, new_populations.len());
            let mut parent2_data = new_populations[parent2].clone();
            
            let split: usize = rng.gen_range(1, tsp_data.city_ids.len() - 1);            
            
            if rng.gen_range(0., 1.) > 0.5 {
                parent1_data.route.reverse();
            }
            if rng.gen_range(0., 1.) > 0.5 {
                parent2_data.route.reverse();                
            }            
            let mut child1 = Sol {
                score: -1.0,
                route: vec![]
            };
            let mut child2 = Sol {
                score: -1.0,
                route: vec![]
            };            

            for i in 0..split {
                child1.route.push(parent1_data.route[i]);
                child2.route.push(parent2_data.route[i]);                
            }
            for i in split..city_num as usize {
                if !child1.route.contains(&parent2_data.route[i]) {
                    child1.route.push(parent2_data.route[i]);
                }
                if !child2.route.contains(&parent1_data.route[i]) {
                    child2.route.push(parent1_data.route[i]);
                }
            }
            for i in 1..split {
                if !child1.route.contains(&parent2_data.route[i]) {
                    child1.route.push(parent2_data.route[i]);
                }
                if !child2.route.contains(&parent1_data.route[i]) {
                    child2.route.push(parent1_data.route[i]);
                }
            }
            child1.route.push(0);
            child2.route.push(0);
            child1.score = tsp_data.calc_score(&child1.route);
            child2.score = tsp_data.calc_score(&child2.route);
            new_populations.push(child1);
            new_populations.push(child2);            
        }
        for data in &new_populations {
            if rng.gen_range(0., 1.) < 0.2 {
                let left: usize = rng.gen_range(1, tsp_data.city_ids.len());
                let right: usize = rng.gen_range(1, tsp_data.city_ids.len());
                let mut new_data = Sol {
                    score: -1.0,
                    route: data.route.clone(),
                };            
                new_data.route.swap(left, right);
                new_data.score = tsp_data.calc_score(&new_data.route);
            }
        }
        populations = new_populations;
    }

}



fn main() {
    let start_time = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_secs() as f64;
    let max_iter = 1000000000;
    //let max_iter = 10000000;
    let tsp_data = parse(String::from("../cities1000.csv"));
    //let tsp_data = parse(String::from("../cities10000.csv"));
    //let tsp_data = parse(String::from("../cities.csv"));
    let route = greedy(&tsp_data);
    /*
    let mut route = Vec::new();
    for i in 1..tsp_data.city_ids.len(){
        route.push(i);
    }

    route.push(0);
     */
    //let route = read_route(String::from("greedy_197769.csv"));
    //let route = read_route(String::from("greedy_1000.csv"));
    //let route = read_route(String::from("../cities.csv.path.csv.csv"));
    let sa_start_time = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_secs() as f64;
    ga(&tsp_data, &route, sa_start_time - start_time, max_iter);
}


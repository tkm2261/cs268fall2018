extern crate csv;
extern crate rand;
extern crate primal;


use rand::Rng;


use std::io::{Write, Read, BufWriter, BufReader, copy};

use std::fs::File;
use std::f64;

#[derive(Debug)]
struct TspData {
    city_ids: Vec<usize>,
    primals: Vec<usize>,    
    coods: Vec<(f64, f64)>,
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
            if i % 10 == 0 && !self.primals.contains(&route[i - 1]) {
                dist += d * 0.1;
            }
        }
       dist
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
    let mut rdr = csv::Reader::from_reader(f.unwrap());
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


fn prob_bolzman(e1: &f64, e2: &f64, t: &f64) -> f64{
    if e2 < e1 {
        1.
    } else {
        (- (e2 - e1) / t).exp()
    }
}


fn sa(tsp_data: &TspData, _route: &Vec<usize>) {
    let mut current_sol = _route.clone();
    let mut current_score = tsp_data.calc_score(&current_sol);
    let mut curr_temp = 10.;
    let ending_temp = 0.0;
    let mut rng = rand::thread_rng();

    let mut global_sol = _route.clone();
    let mut global_score = current_score;
    
    println!("init score: {:?}", tsp_data.calc_score(&_route));    
    for i in 1..100000000 {
        if curr_temp < ending_temp{
            break   
        }
        curr_temp = curr_temp * 0.999;

        let left: usize = rng.gen_range(1, tsp_data.city_ids.len());
        let right: usize = rng.gen_range(1, tsp_data.city_ids.len());
        let p: f64 = rng.gen_range(0., 1.);
        current_sol.swap(left, right);
        
        let next_score = tsp_data.calc_score(&current_sol);
        
        if global_score > next_score {
            global_score = next_score;
            global_sol = current_sol.clone();
        }
        if next_score < current_score || prob_bolzman(&current_score, &next_score, &curr_temp) > p {
            current_score = next_score;
        } else {
            current_sol.swap(left, right); //reverse
        }

        if i % 10000 == 0 {
            println!("{} {} {}", i, current_score, global_score);
        }
    }
}



fn main() {
    let tsp_data = parse(String::from("../cities.csv"));
    let route = greedy(&tsp_data);
    /*
    let mut route = Vec::new();
    for i in 1..tsp_data.city_ids.len(){
        route.push(i);
    }

    route.push(0);
     */
    //let route = read_route(String::from("greedy_1000.csv"));
    //sa(&tsp_data, &route);
}


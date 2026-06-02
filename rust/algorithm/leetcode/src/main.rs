use std::collections::HashMap;

// https://leetcode.cn/problems/two-sum/?envType=study-plan-v2&envId=top-100-liked
fn two_sum(nums: Vec<i32>, target: i32) -> Vec<i32> {
    let mut nums_map = HashMap::new();
    for i in 0..nums.len() {
        nums_map.insert(nums[i], i as i32);
    }
    for i in 0..nums.len() {
        if let Some(value) = nums_map.get(&(target - nums[i])) && *value != i as i32 {
            return vec![i as i32,*value];
        }
    }
    return vec![0,0];
}

fn main() {
    println!("Hello, world!");
    println!("result: {:?}",two_sum(vec![2,7,11,15], 9));
}

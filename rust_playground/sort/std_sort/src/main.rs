fn main() {
    let mut arr = [12,24,534,2,123,5,3,7,2,-123,0,23];
    println!("before\t:{:?}",arr);
    //arr.sort();
    arr.sort_unstable();
    println!("after\t:{:?}",arr);
}

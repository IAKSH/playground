fn main() {
    let mut arr = [12,24,534,2,123,5,3,7,2,-123,0,23];
    println!("before\t:{:?}",arr);
    arr.sort_unstable();
    println!("after\t:{:?}",arr);
    assert_eq!([-123, 0, 2, 2, 3, 5, 7, 12, 23, 24, 123, 534],arr);
}

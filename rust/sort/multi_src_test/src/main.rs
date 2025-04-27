// 告诉rustc有这个模块，让他去找同名文件，然后尝试编译
// 我的理解是，一个.rs就是一个模块，模块中可以有（也可以没有）子模块
// 类似于java的源文件即类（？），只不过rust的这个似乎是隐式声明的
// 好吧，应该更像是python的模块机制
mod my_func;
// 比如这几个mod里就没有子mod
mod add;
mod duck;
// 尝试使用子目录中的mod
mod sub_folder_mod;

// 引用这个模块中的一个pub mod
// 如果不引用，当然也就没得用
// 老实说，有点繁琐
use my_func::my_funcs;
// 这样可以少写点重复的namespace
// 其实完全就是C++的using xxx::xxx吧
use duck::Duck;

// 尝试引用另一个（本地）Crate
extern crate multi_src_lib_test;
use multi_src_lib_test::say_nihao::say;

fn feed_duck() {
    let mut ducks: Vec<Duck> = vec![];
    ducks.push(Duck::new("ducka", 114));
    ducks.push(Duck::new("dackb", 0));
    ducks.push(Duck::new("dackc", 1919810));
    ducks.push(Duck::new("dackd", 0));
    ducks.push(Duck::new("dacke", 514));
    ducks.push(Duck::new("dackf", -514));
    ducks.sort_unstable_by_key(|x| x.get_age());

    for i in 0..5 {
        for duck in &mut ducks {
            duck.pass_yaer(i);
            duck.show_info();
        }
    }
}

pub fn reverse(text: &str) -> String {
    text.chars().rev().collect()
}

pub fn println_reverse(text: &str) {
    println!("{}",text.chars().rev().collect::<String>());
}

fn main() {
    println!("Hello, world!");
    my_funcs::say_hello_en();
    my_funcs::say_hello_zh();
    my_func::my_funcs::say_hello_en();
    println!("add = {}",add::add(114,514));
    println!("add_nums = {}",add::add_numbers([114,514,1919,810].to_vec()));
    feed_duck();
    sub_folder_mod::say_hello();
    
    say();
    multi_src_lib_test::say_nihaoma::say();

    println!("{}",reverse("我在想如果用中文会不会炸掉"));
    println_reverse("然而并没有");
    println!("{}","甚至可以直接在这里反向".chars().rev().collect::<String>());
}

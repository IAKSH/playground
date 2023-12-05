pub struct Duck {
    name: String,
    age: i32
}

impl Drop for Duck {
    fn drop(&mut self) {
        println!("Duck {} 寄了,享年{}",self.name,self.age);
    }
}

impl Duck {
    pub fn new(name: &str,age: i32) -> Duck {
        match age {
            0 => {println!("Duck {} 诞生了",name);}
            _ => {println!("Duck {} 诞生了，但是现年{}",name,age);}
        }

        Duck {
            name: (name.to_string()),
            age: (age)
        }
    }

    pub fn show_info(&self) {
        println!("Duck {} 现年{}",self.name,self.age);
    }

    pub fn pass_yaer(&mut self,n: u8) {
        for _i in 0..n {
            self.age_add();
        }
    } 

    pub fn get_age(&self) -> i32 {
        self.age
    }

    fn age_add(&mut self) {
        self.age += 1;
    }
}
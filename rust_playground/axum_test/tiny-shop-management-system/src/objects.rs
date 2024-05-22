pub struct Staff {
    staff_id: i32,
    name: String,
    gender: char,
    age: i32,
    monthly_salary: f32,
}

pub struct Product {
    product_id: i32,
    name: String,
    brand: String,
    unit_price: f32,
    quantity: i32,
}

pub struct Sales {
    sales_id: i32,
    product_id: i32,
    sale_time: String,  // 使用String类型来存储日期时间
    actual_unit_price: f32,
    sold_quantity: i32,
    member_id: Option<i32>,  // 使用Option类型来处理可能为空的会员ID
}

pub struct Member {
    member_id: i32,
    name: String,
    membership_start_date: String,
    membership_end_date: String,
}

pub struct Purchase {
    purchase_id: i32,
    product_id: i32,
    purchase_time: String,
    purchase_unit_price: f32,
    purchase_quantity: i32,
    source_id: i32,
}

pub struct Source {
    source_id: i32,
    name: String,
}

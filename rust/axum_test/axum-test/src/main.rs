use axum::{
    routing::get,
    response::Json,
    Router,
};
use serde_json::{Value, json};

#[tokio::main]
async fn main() {
    // build our application with a single route
    let app = Router::new()
        .route("/", get(|| async { "Hello, World!" }))
        .route("/count", get(count_response))
        .route("/json", get(json))
        .route("/plain_text", get(plain_text));

    // run our app with hyper, listening globally on port 3000
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

static mut COUNT: i32 = 0;

async fn count_response() -> String {
    unsafe {COUNT += 1};
    return unsafe {COUNT}.to_string();
}

async fn plain_text() -> &'static str {
    "foo"
}

async fn json() -> Json<Value> {
    Json(json!({ "data": 42 }))
}

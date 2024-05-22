use axum::extract::{Path, Query};
use axum::http::{Result, StatusCode};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{extract::Extension, Json, Router};
use serde_json::{error, to_string, Value};
use sqlx::mysql::{MySqlPool, MySqlRow};
use anyhow::Context;
use std::collections::HashMap;
use serde::Deserialize;

#[derive(Debug)]

pub enum CustomError {
    BadRequest,
    NotFound,
    InternalServerError,
}

impl IntoResponse for CustomError {
    fn into_response(self) -> axum::response::Response {
        let (status, error_message) = match self {
            Self::InternalServerError => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Internal Server Error",
            ),
            Self::BadRequest=> (StatusCode::BAD_REQUEST, "Bad Request"),
            Self::NotFound => (StatusCode::NOT_FOUND, "DB Recored Not Found"),
        };
        (status, Json(serde_json::json!({"error": error_message}))).into_response()
    }
}

const url: &str = "mariadb://root:mariadb@192.168.229.131:3306/shop";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let pool = MySqlPool::connect(&url).await.context("could not connect to database_url")?;

    // build our application with a single route
    let app = Router::new()
    .route("/", get(|| async { "Hello, World!" }))
    .route("/count", get(count_response))
    .route("/json", get(json))
    .route("/plain_text", get(plain_text))
    .route("/test/get", get(get_test))
    .route("/test/post", post(post_test))
    .layer(Extension(pool));

    // run our app with hyper, listening globally on port 3000
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    axum::serve(listener, app).await.unwrap();

    Ok(())
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
    Json(serde_json::json!({ "data": 42 }))
}

// 获取get参数

#[derive(Deserialize)]
pub struct IdArg {
    pub id: i32
}

async fn get_test(Query(args): Query<IdArg>,Extension(pool): Extension<MySqlPool>) -> anyhow::Result<Json<Value>, CustomError> {
    // 使用连接池进行数据库操作
    let row: (i32, String, String, i32, f32) = sqlx::query_as("SELECT StaffID, Name, Gender, Age, CAST(MonthlySalary AS FLOAT) FROM Staff WHERE StaffID = ?")
        .bind(args.id)
        .fetch_one(&pool)
        .await
        .map_err(|_| {
            CustomError::NotFound
        })?;

    Ok(Json(serde_json::json!(row)))
}

// 获取post参数

#[derive(Deserialize)]
pub struct StaffArgs {
    pub id: i32,
    pub name: String,
    pub gender: u8,
    pub age: i32,
    pub monthly_salary: i32,
}


async fn post_test(
    post_txt: String,
    Extension(pool): Extension<MySqlPool>
) -> anyhow::Result<Json<Value>, CustomError> {
    let json = serde_json::json!(post_txt);
    // 使用连接池进行数据库操作
    //let row: (i32, String, String, i32, f32) = sqlx::query_as("INSERT INTO Staff (StaffID, Name, Gender, Age, MonthlySalary) VALUES (?, ?, ?, ?, ?)")
    //    .bind(json["id"])
    //    .bind(json["name"])
    //    .bind(json["gender"])
    //    .bind(json["age"])
    //    .bind(json["monthly_salary"])
    //    .fetch_one(&pool)
    //    .await
    //    .map_err(|_| {
    //        CustomError::InternalServerError
    //    })?;
//
    Ok(Json(serde_json::json!("CNM")))
}

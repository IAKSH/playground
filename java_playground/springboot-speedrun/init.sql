CREATE DATABASE myDatabase;

\c myDatabase;

CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    age INTEGER,
    email VARCHAR(100)
);

INSERT INTO users (name, age, email) VALUES ('John Doe', 30, 'john.doe@example.com');
INSERT INTO users (name, age, email) VALUES ('Jane Doe', 28, 'jane.doe@example.com');

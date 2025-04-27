CREATE DATABASE myDatabase;

\c myDatabase;

CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    age INTEGER,
    email VARCHAR(100),
    password VARCHAR(255)
);

CREATE TABLE roles (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100)
);

CREATE TABLE user_roles (
    user_id INTEGER,
    role_id INTEGER,
    PRIMARY KEY (user_id, role_id),
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (role_id) REFERENCES roles(id)
);

-- password: johns_hashed_password
INSERT INTO users (name, age, email, password) VALUES ('John Doe', 30, 'john.doe@example.com', '$2a$10$qq7is/52CFw1iLbbj4N8Ae/TFK6CnUNLJ7C13CUWPl1sxo5u3C8Sa');
-- password: janes_hashed_password
INSERT INTO users (name, age, email, password) VALUES ('Jane Doe', 28, 'jane.doe@example.com', '$2a$10$miLuo7krmjo0NKqcqH2YeeZ6Xt8YsqiHQ55ikiGe.AurNXMZ4q6Nq');

INSERT INTO roles (name) VALUES ('ADMIN');
INSERT INTO roles (name) VALUES ('USER');

INSERT INTO user_roles (user_id, role_id) VALUES (1, 1); -- John Doe is an ADMIN
INSERT INTO user_roles (user_id, role_id) VALUES (2, 2); -- Jane Doe is a USER

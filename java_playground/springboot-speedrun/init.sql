CREATE DATABASE myDatabase;

\c myDatabase;

CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    age INTEGER,
    email VARCHAR(100),
    password VARCHAR(255)
);

-- password: johns_hashed_password
INSERT INTO users (name, age, email, password) VALUES ('John Doe', 30, 'john.doe@example.com', '$2a$10$qq7is/52CFw1iLbbj4N8Ae/TFK6CnUNLJ7C13CUWPl1sxo5u3C8Sa');
-- password: janes_hashed_password
INSERT INTO users (name, age, email, password) VALUES ('Jane Doe', 28, 'jane.doe@example.com', '$2a$10$miLuo7krmjo0NKqcqH2YeeZ6Xt8YsqiHQ55ikiGe.AurNXMZ4q6Nq');

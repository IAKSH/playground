#include <iostream>
#include <pqxx/pqxx>

static const std::string conn_string {
    "hostaddr = 192.168.1.110 port = 5432 dbname = mydatabase user = lain password = postgresql"
};

static const std::string init_sql {
R"(
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
)"
};

int main() {
    try {
        pqxx::connection c(conn_string);

        pqxx::work w(c);
        //w.exec0(init_sql);
        //w.commit();

        for(const auto& i : w.exec("SELECT * FROM users;")) {
            std::cout << "------------------------------------\n";
            for(const auto& j : i) {
                std::cout << j << '\n';
            }
        }
    }
    catch (std::exception const &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}

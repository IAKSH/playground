CREATE DATABASE IF NOT EXISTS books DEFAULT CHARSET=gbk COLLATE=gbk_bin;

CREATE DATABASE IF NOT EXISTS test;
DROP DATABASE test;

USE books;

CREATE TABLE IF NOT EXISTS reader (
	reader_no VARCHAR(16) PRIMARY KEY,
	reader_type VARCHAR(16),
	reader_name VARCHAR(16) NOT NULL,
	reader_academy VARCHAR(16)
) ENGINE=INNODB;

CREATE TABLE IF NOT EXISTS book (
	book_no VARCHAR(16) PRIMARY KEY,
	book_name VARCHAR(16) NOT NULL,
	book_author VARCHAR(16),
	year_of_book_publication YEAR,
	book_price DECIMAL(10,1),
	book_status VARCHAR(2) DEFAULT '未借'
) ENGINE=INNODB;

CREATE TABLE IF NOT EXISTS borrow (
	reader_no VARCHAR(16) NOT NULL,
	book_no VARCHAR(16) PRIMARY KEY,
	borrow_date DATE,
	return_date DATE,
	FOREIGN KEY (reader_no) REFERENCES reader(reader_no),
	FOREIGN KEY (book_no) REFERENCES book(book_no)
) ENGINE=INNODB;

INSERT INTO reader (reader_no,reader_type,reader_name,reader_academy) VALUES ("0041001","学生","李笑笑","计算机");
INSERT INTO reader (reader_no,reader_type,reader_name,reader_academy) VALUES ("1110","教师","王四海","计算机");
INSERT INTO reader (reader_no,reader_type,reader_name,reader_academy) VALUES ("0071015","学生","杜拉拉","经济管理");
SELECT * FROM reader;

INSERT INTO book (book_no,book_name,book_author,year_of_book_publication,book_price,book_status) VALUES ("TP273-1","大数据技术","林子明",2017,40.5,"借出");
INSERT INTO book (book_no,book_name,book_author,year_of_book_publication,book_price,book_status) VALUES ("TP273-3","云计算","刘鹏",2018,39,"借出");
INSERT INTO book (book_no,book_name,book_author,year_of_book_publication,book_price,book_status) VALUES ("TP311-317","数据库系统概论","王珊",2014,39.6,"未借");
SELECT * FROM book;

INSERT INTO borrow (reader_no,book_no,borrow_date,return_date) VALUES ("1110","TP273-3","2022-3-1","2022-9-1");
INSERT INTO borrow (reader_no,book_no,borrow_date,return_date) VALUES ("0041001","TP273-1","2022-3-6","2022-9-6");
SELECT * FROM borrow;

#DELETE FROM reader WHERE reader_no = "1110";
DELETE FROM reader WHERE reader_no = "0071015";

SELECT * FROM borrow;
SELECT * FROM reader;

CREATE TABLE IF NOT EXISTS teacherInfo( 
id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY, 
NAME VARCHAR(20),
sex CHAR(1) DEFAULT '男',
birthday DATE,
address VARCHAR(50) 
);

DELETE FROM teacherInfo;
SELECT * FROM teacherInfo;

ALTER TABLE teacherInfo MODIFY NAME VARCHAR(30) NOT NULL;
ALTER TABLE teacherInfo MODIFY birthday DATE AFTER NAME;
ALTER TABLE teacherInfo CHANGE id t_id INT NOT NULL; 
ALTER TABLE teacherInfo DROP address;
ALTER TABLE teacherInfo  ADD wages DECIMAL(5,2);
ALTER TABLE teacherInfo RENAME teacherInfo_Info;

DESCRIBE teacherInfo_Info;

DROP TABLE teacherInfo_Info;
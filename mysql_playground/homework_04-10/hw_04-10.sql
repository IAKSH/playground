# init
/*
SQLyog Community v13.1.6 (64 bit)
MySQL - 8.0.11 : Database - books
*********************************************************************
*/

/*!40101 SET NAMES utf8 */;

/*!40101 SET SQL_MODE=''*/;

/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;
CREATE DATABASE /*!32312 IF NOT EXISTS*/`books` /*!40100 DEFAULT CHARACTER SET gbk COLLATE gbk_bin */;

USE `books`;

/*Table structure for table `book` */

DROP TABLE IF EXISTS `book`;

CREATE TABLE `book` (
  `bookno` VARCHAR(20) COLLATE gbk_bin NOT NULL COMMENT '书号',
  `bookname` VARCHAR(30) COLLATE gbk_bin DEFAULT NULL COMMENT '书名',
  `author` VARCHAR(20) COLLATE gbk_bin DEFAULT NULL COMMENT '作者',
  `publisher` VARCHAR(10) COLLATE gbk_bin DEFAULT NULL COMMENT '出版社',
  `publishyear` YEAR(4) DEFAULT NULL COMMENT '出版年',
  `price` DECIMAL(5,1) DEFAULT NULL COMMENT '单价',
  PRIMARY KEY (`bookno`)
) ENGINE=INNODB DEFAULT CHARSET=gbk COLLATE=gbk_bin;

/*Data for the table `book` */

INSERT  INTO `book`(`bookno`,`bookname`,`author`,`publisher`,`publishyear`,`price`) VALUES 
('H652','三毛流浪记','张乐平','文化出版社',1953,25.0),
('TN123','电路','童诗白','电子出版社',2005,37.0),
('TP273-1','大数据时代','刘鹏','清华出版社',2017,40.5),
('TP273-3','云计算','刘鹏','电子出版社',2018,39.0),
('TP273-342','SQL SERVER 2008','王利','电子出版社',2009,32.0),
('TP30101','数据库','高尚','电子出版社',2009,28.0),
('TP388','计算机网络','张明','清华出版社',2009,33.0),
('TP458','数据结构','王建设','清华出版社',NULL,NULL),
('TP987','数据库系统概论','林美','高教出版社',2008,45.0),
('TP999','软件工程','张海','清华出版社',NULL,NULL);

/*Table structure for table `borrow` */

DROP TABLE IF EXISTS `borrow`;

CREATE TABLE `borrow` (
  `readerno` VARCHAR(10) COLLATE gbk_bin NOT NULL COMMENT '读者编号',
  `bookno` VARCHAR(20) COLLATE gbk_bin NOT NULL COMMENT '书号',
  `borrowdate` DATE DEFAULT NULL COMMENT '借出日期',
  `returndate` DATE DEFAULT NULL COMMENT '还书日期',
  PRIMARY KEY (`bookno`),
  KEY `readerno` (`readerno`),
  CONSTRAINT `borrow_ibfk_1` FOREIGN KEY (`readerno`) REFERENCES `reader` (`readerno`),
  CONSTRAINT `borrow_ibfk_2` FOREIGN KEY (`bookno`) REFERENCES `book` (`bookno`)
) ENGINE=INNODB DEFAULT CHARSET=gbk COLLATE=gbk_bin;

/*Data for the table `borrow` */

INSERT  INTO `borrow`(`readerno`,`bookno`,`borrowdate`,`returndate`) VALUES 
('003002','H652','2022-03-15','2022-09-15'),
('1110','TP273-1','2022-03-01','2022-09-01'),
('1110','TP273-3','2022-03-01','2022-09-01'),
('004001','TP30101','2022-03-06','2022-09-06'),
('004002','TP458','2022-03-20','2022-09-20');

/*Table structure for table `reader` */

DROP TABLE IF EXISTS `reader`;

CREATE TABLE `reader` (
  `readerno` VARCHAR(10) COLLATE gbk_bin NOT NULL COMMENT '读者编号',
  `readertype` CHAR(2) COLLATE gbk_bin NOT NULL COMMENT '读者类型',
  `readername` VARCHAR(15) COLLATE gbk_bin DEFAULT NULL COMMENT '读者姓名',
  `readerdept` VARCHAR(15) COLLATE gbk_bin DEFAULT NULL COMMENT '学院',
  PRIMARY KEY (`readerno`)
) ENGINE=INNODB DEFAULT CHARSET=gbk COLLATE=gbk_bin;

/*Data for the table `reader` */

INSERT  INTO `reader`(`readerno`,`readertype`,`readername`,`readerdept`) VALUES 
('003002','学生','王小明','电气'),
('004001','学生','李笑笑','计算机'),
('004002','学生','张大卫','计算机'),
('1110','教师','王闵','计算机'),
('2225','教师','高鸿','计算机');

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

# modify
-- book表中增加state（图书状态字段，缺省值是“未借”）
ALTER TABLE book
ADD state CHAR(2) DEFAULT '未借';

-- 将未借出的图书的状态改为'未借' 
UPDATE book 
SET state='未借' 
WHERE bookno NOT IN (SELECT bookno FROM borrow);

-- 将已借出的图书的状态改为'借出' 
UPDATE book 
SET state='借出' 
WHERE bookno IN (SELECT bookno FROM borrow);

-- reader表中增加borrowed_num（已借书数字段，缺省值是0）
ALTER TABLE reader
ADD borrowed_num TINYINT DEFAULT 0;

-- 根据borrow表中记录的借书情况更改reader表的各个读者的已借书数
UPDATE reader 
SET borrowed_num=(SELECT COUNT(*) FROM borrow WHERE readerno=reader.readerno);

-- 查看数据情况：
SELECT * FROM book ORDER BY state;
SELECT * FROM reader ORDER BY borrowed_num DESC;

# 1.1
DROP PROCEDURE IF EXISTS reader_namelist;

DELIMITER $$
CREATE PROCEDURE reader_namelist(IN ser_name VARCHAR(15))
BEGIN
	SELECT readerno,readername,readertype,readerdept
	FROM reader
	WHERE readername LIKE CONCAT('%',ser_name,'%')
	ORDER BY readerdept;
END$$

CALL reader_namelist('王');

# 1.2
DROP PROCEDURE IF EXISTS returnbook;

DELIMITER $$
CREATE PROCEDURE returnbook(IN ser_readerno VARCHAR(10), IN ser_bookno VARCHAR(20))
BEGIN
	DELETE FROM borrow WHERE readerno = ser_readerno AND bookno = ser_bookno;
	UPDATE reader SET borrowed_num = borrowed_num - 1 WHERE readerno = ser_readerno;
	UPDATE book SET state = '未借' WHERE bookno = ser_bookno;
END$$

CALL returnbook('1110','TP273-1');

# for test
#select * from borrow where bookno = 'TP273-1';
#SELECT * FROM book WHERE bookno = 'TP273-1';
#select * from reader where readerno = '1110';

# 1.3
DROP PROCEDURE IF EXISTS if_borrow;

DELIMITER $$
CREATE PROCEDURE if_borrow(IN rno VARCHAR(10),OUT can_borrow TINYINT)
BEGIN
	IF (SELECT borrowed_num FROM reader WHERE readerno = rno) < 10 THEN
		SELECT 1 INTO can_borrow;
	ELSE
		SELECT 0 INTO can_borrow;
	END IF;
END$$

SET @can_borrow = 0;
CALL if_borrow('1110',@can_borrow);
SELECT @can_borrow;

# 2.4
SELECT YEAR(CURDATE());
# 2.5
SELECT DATE_ADD(CURDATE(), INTERVAL 6 MONTH);
# 2.6
SELECT SUBSTRING(TRIM('    爱我中华！  '),3,2);
# 2.7
SELECT FLOOR(RAND() * 1000) + FLOOR(RAND() * 1000);
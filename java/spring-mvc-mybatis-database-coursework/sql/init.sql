CREATE DATABASE shop;
USE shop;

-- 工作人员表
CREATE TABLE Staff (
    StaffID INT PRIMARY KEY AUTO_INCREMENT,
    Name VARCHAR(100),
    Gender CHAR(1),
    Age INT,
    MonthlySalary DECIMAL(10, 2) DEFAULT 0
);

-- 商品表
CREATE TABLE Product (
    ProductID INT PRIMARY KEY AUTO_INCREMENT,
    Name VARCHAR(100),
    Brand VARCHAR(100),
    UnitPrice DECIMAL(10, 2),
    Quantity INT DEFAULT 0
);

-- 会员表
CREATE TABLE Member (
    MemberID INT PRIMARY KEY AUTO_INCREMENT,
    Name VARCHAR(100),
    MembershipStartDate DATE DEFAULT NOW(),
    MembershipEndDate DATE
);

-- 销售表
CREATE TABLE Sales (
    SalesID INT PRIMARY KEY AUTO_INCREMENT,
    ProductID INT NOT NULL,
    SaleTime DATETIME,
    ActualUnitPrice DECIMAL(10, 2),
    SoldQuantity INT,
    MemberID INT,
    FOREIGN KEY (ProductID) REFERENCES Product(ProductID),
    FOREIGN KEY (MemberID) REFERENCES Member(MemberID)
);

-- 采购源表
CREATE TABLE Source (
    SourceID INT PRIMARY KEY AUTO_INCREMENT,
    Name VARCHAR(100)
);

-- 采购表
CREATE TABLE Purchase (
    PurchaseID INT PRIMARY KEY AUTO_INCREMENT,
    ProductID INT NOT NULL,
    PurchaseTime DATETIME DEFAULT NOW(),
    PurchaseUnitPrice DECIMAL(10, 2),
    PurchaseQuantity INT,
    SourceID INT NOT NULL,
    FOREIGN KEY (ProductID) REFERENCES Product(ProductID),
    FOREIGN KEY (SourceID) REFERENCES Source(SourceID)
);

-- 创建视图，获取每日、每月、每年的销售额统计
CREATE VIEW DailySales AS
SELECT DATE(SaleTime) AS Day, SUM(ActualUnitPrice * SoldQuantity) AS TotalSales
FROM Sales
GROUP BY Day;

CREATE VIEW MonthlySales AS
SELECT DATE_FORMAT(SaleTime, '%Y-%m') AS Month, SUM(ActualUnitPrice * SoldQuantity) AS TotalSales
FROM Sales
GROUP BY Month;

CREATE VIEW YearlySales AS
SELECT YEAR(SaleTime) AS Year, SUM(ActualUnitPrice * SoldQuantity) AS TotalSales
FROM Sales
GROUP BY Year;

-- 创建视图，获取总收支统计
CREATE VIEW TotalIncome AS
SELECT SUM(ActualUnitPrice * SoldQuantity) AS Income
FROM Sales;

CREATE VIEW TotalExpense AS
SELECT SUM(MonthlySalary) AS StaffExpense, SUM(PurchaseUnitPrice * PurchaseQuantity) AS ProductExpense
FROM Staff, Purchase;

-- 创建事件，定期自动删除到达有效期限的会员
DELIMITER $$
CREATE EVENT ExpiredMembership
ON SCHEDULE EVERY 1 DAY
DO
BEGIN
    DELETE FROM Member WHERE MembershipEndDate < CURDATE();
END$$
DELIMITER ;

-- 从货源购入商品，若不存在该货源则先添加对该货源的记录
DELIMITER $$
CREATE PROCEDURE PurchaseProduct(IN p_ProductID INT, IN p_SourceName VARCHAR(100), IN p_PurchaseTime DATETIME, IN p_PurchaseUnitPrice DECIMAL(10, 2), IN p_PurchaseQuantity INT)
BEGIN
    DECLARE v_SourceID INT;

    -- 开启事务
    START TRANSACTION;

    -- 检查是否存在该货源
    SELECT SourceID INTO v_SourceID FROM Source WHERE Name = p_SourceName;
    IF v_SourceID IS NULL THEN
        -- 如果不存在该货源，则先添加对该货源的记录
        INSERT INTO Source (Name) VALUES (p_SourceName);
        SET v_SourceID = LAST_INSERT_ID();
    END IF;

    -- 从货源购入商品
    INSERT INTO Purchase (ProductID, PurchaseTime, PurchaseUnitPrice, PurchaseQuantity, SourceID) VALUES (p_ProductID, p_PurchaseTime, p_PurchaseUnitPrice, p_PurchaseQuantity, v_SourceID);

    -- 提交事务
    COMMIT;
END$$
DELIMITER ;

DELIMITER $$
CREATE PROCEDURE SellProduct(IN p_ProductID INT, IN p_SaleTime DATETIME, IN p_SoldQuantity INT, IN p_MemberID INT)
proc_label: BEGIN  -- 定义一个标签
    DECLARE v_ActualUnitPrice DECIMAL(10, 2);
    DECLARE v_UnitPrice DECIMAL(10, 2);
    DECLARE v_Quantity INT;

    -- 开启事务
    START TRANSACTION;

    -- 获取商品的单价和数量
    SELECT UnitPrice, Quantity INTO v_UnitPrice, v_Quantity FROM Product WHERE ProductID = p_ProductID;

    -- 检查商品的数量是否足够
    IF v_Quantity < p_SoldQuantity THEN
        -- 如果商品数量不足，则回滚事务并返回错误信息
        ROLLBACK;
        SELECT 'Insufficient product quantity' AS ErrorMessage;
        LEAVE proc_label;  -- 使用LEAVE语句退出代码块
    END IF;

    -- 判断是否是会员
    IF p_MemberID IS NOT NULL THEN
        -- 如果是会员，则9折出售
        SET v_ActualUnitPrice = v_UnitPrice * 0.9;
    ELSE
        -- 如果不是会员，则按原价出售
        SET v_ActualUnitPrice = v_UnitPrice;
    END IF;

    -- 卖出商品
    INSERT INTO Sales (ProductID, SaleTime, ActualUnitPrice, SoldQuantity, MemberID) VALUES (p_ProductID, p_SaleTime, v_ActualUnitPrice, p_SoldQuantity, p_MemberID);

    -- 更新商品的数量
    UPDATE Product SET Quantity = Quantity - p_SoldQuantity WHERE ProductID = p_ProductID;

    -- 提交事务
    COMMIT;
END$$
DELIMITER ;

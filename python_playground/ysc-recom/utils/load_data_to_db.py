import pandas as pd
import mysql.connector
import argparse
from tqdm import tqdm

# sql里表的定义有一些歧义，暂时先不管了，反正不是我写的

def main(database_info, excel_path):
    data = pd.read_excel(excel_path)
    cnx = mysql.connector.connect(user=database_info['user'], password=database_info['password'],
                                  host=database_info['host'], database=database_info['database'])
    cursor = cnx.cursor()

    for index, row in tqdm(data.iterrows(), total=data.shape[0], desc="Processing rows"):
        col_img_url = row["图片"]
        if len(col_img_url) > 240:
            col_img_url = "none"

        col_price = row["价格"]
        col_name = row["商品介绍"]
        col_shop = row["店铺"]

        # 检查外键
        cursor.execute("SELECT shop_id FROM shop WHERE name = %s", (col_shop,))
        shop_id = cursor.fetchone()

        # 如果shop表中不存在该店铺，则插入店铺数据
        if shop_id is None:
            cursor.execute("INSERT INTO shop (name) VALUES (%s)", (col_shop,))
            cnx.commit()
            shop_id = cursor.lastrowid
        else:
            shop_id = shop_id[0]

        # 插入hanfu数据
        cursor.execute("""
            INSERT INTO hanfu (shop_id, shop_name, name, price, image, label, upload_time, likes, views)
            VALUES (%s, %s, %s, %s, %s, %s, CURDATE(), 0, 0)
        """, (shop_id, col_shop, col_name, col_price, col_img_url, "未分类"))
        cnx.commit()
        hanfu_id = cursor.lastrowid

        # 插入图片URL数据
        cursor.execute("""
            INSERT INTO hanfu_image (hanfu_id, image)
            VALUES (%s, %s)
        """, (hanfu_id, col_img_url))
        cnx.commit()

    # 关闭数据库连接
    cursor.close()
    cnx.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process Excel data and insert into MySQL database.')
    parser.add_argument('--user', required=True, help='Database user')
    parser.add_argument('--password', required=True, help='Database password')
    parser.add_argument('--host', required=True, help='Database host')
    parser.add_argument('--database', required=True, help='Database name')
    parser.add_argument('--excel', required=True, help='Path to Excel file')

    args = parser.parse_args()

    database_info = {
        'user': args.user,
        'password': args.password,
        'host': args.host,
        'database': args.database
    }

    main(database_info, args.excel)

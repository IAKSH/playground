import os
import mysql.connector


def connect_db():
    db_config = {
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD'),
        'host': os.getenv('DB_HOST'),
        'database': os.getenv('DB_DATABASE')
    }
    connection = mysql.connector.connect(**db_config)
    return connection


def get_items_from_db(connection):
    cursor = connection.cursor(dictionary=True)
    cursor.execute("SELECT hanfu_id, CONCAT(shop_name, ' ' , label , ' ', name, ' ', price) AS title FROM hanfu")
    items = cursor.fetchall()
    cursor.close()
    return items


def get_item_titles_from_db_with_ids(connection, item_ids):
    cursor = connection.cursor(dictionary=True)
    format_strings = ','.join(['%s'] * len(item_ids))
    cursor.execute(f"SELECT CONCAT(shop_name, ' ', label, ' ', name, ' ', price) AS title FROM hanfu WHERE "
                   f"hanfu_id IN ({format_strings})", tuple(item_ids))
    items = cursor.fetchall()
    cursor.close()
    return items

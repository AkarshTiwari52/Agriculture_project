import mysql.connector

def get_db(dictionary=False):
    conn = mysql.connector.connect(
        host="127.0.0.1",
        user="root",
        password="Akarsh2.0@",
        database="agri_auth",
        port=3306
    )

    cursor = conn.cursor(dictionary=dictionary)
    return conn, cursor


from werkzeug.security import generate_password_hash

print(generate_password_hash("Akarsh2.0@"))
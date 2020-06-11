import mysql.connector
##追加
import db_info

# データベース接続
def main():
    con = mysql.connector.connect(
        host=db_info.db_user.host,
        db=db_info.db_user.db,
        user=db_info.db_user.username,
        passwd=db_info.db_user.userpass
    )

    # 辞書型カーソル取得
    cur = con.cursor(dictionary=True)

    # 検索
    sql='select * from result'
    cur.execute(sql)
    # すべての行を取得
    rows = cur.fetchall()
    for row in rows:
        # 表示
        print(row)

    # 切断
    cur.close()
    con.close()

if __name__ == '__main__':
    main()
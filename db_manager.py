import mysql.connector
##追加
import db_info

# データベース接続
 #db接続
def connect_db():
    con = mysql.connector.connect(
        host=db_info.db_user.host,
        db=db_info.db_user.db,
        user=db_info.db_user.username,
        passwd=db_info.db_user.userpass
    )
        # 辞書型カーソル取得
    cur = con.cursor(dictionary=True)
    return cur,con

def disconnect_db(cur,con):
    # 切断
    cur.close()
    con.close()


def insert_result(test_label,judgment,answer,model_h5,model_json):
    #挿入
    cur,con = connect_db()
    sql='insert into result(test_label,judgment,answer,model_h5,model_json) VALUES(%s,%s,%s,%s,%s)'
    values = (test_label,judgment,answer,model_h5,model_json)
    cur.execute(sql,values)
    con.commit()
    disconnect_db(cur,con)
    print("good")
    # すべての行を取得







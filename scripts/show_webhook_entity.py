import sqlite3

db='database.sqlite'
con=sqlite3.connect(db)
cur=con.cursor()
try:
    cur.execute("SELECT id, workflowId, path, method, webhookId FROM webhook_entity")
    rows = cur.fetchall()
    if not rows:
        print('No rows in webhook_entity')
    else:
        for r in rows:
            print('id:', r[0])
            print('workflowId:', r[1])
            print('path:', r[2])
            print('method:', r[3])
            print('webhookId:', r[4])
            print('---')
except Exception as e:
    print('Error querying webhook_entity:', e)
finally:
    con.close()

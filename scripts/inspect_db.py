import sqlite3
db='database.sqlite'
con=sqlite3.connect(db)
cur=con.cursor()
cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
print('tables:', [r[0] for r in cur.fetchall()])
for name in ['webhook','webhook_entity','workflow','workflow_entity','active_workflows','tunnel']:
    try:
        cur.execute(f"SELECT count(*) FROM {name}")
        print(f"{name}:", cur.fetchone()[0])
    except Exception:
        pass
# list workflows
try:
    cur.execute("SELECT id,name,active FROM workflow")
    print('\nworkflows:')
    for r in cur.fetchall():
        print(r)
except Exception:
    pass
# list webhooks
try:
    cur.execute("SELECT id,workflowId,path,method,webhookId FROM webhook")
    print('\nwebhooks:')
    for r in cur.fetchall():
        print(r)
except Exception:
    pass
con.close()

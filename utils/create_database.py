import os
import subprocess
import re

database_name_list = [
    "accidents", "ccs", "ergastf1", "hepatitis", "sakila", "talkingdata",
    "airline", "chembl", "financial", "hockey", "sap", "telstra",
    "baseball", "consumer", "fnhk", "imdb", "seznam", "tournament",
    "basketball", "credit", "genome", "legalacts", "ssb", "tpc_h",
    "carcinogenesis", "employee", "grants", "movielens", "stats", "tubepricing",
]

data_dir = "/home/weixun/data/db_workload/datasets/"

pg_user = "weixun"
pg_password = ""
pg_host = "127.0.0.1"
pg_port = "12333"

def execute_psql(sql_file, database="postgres"):
    """使用psql执行SQL文件"""
    env = os.environ.copy()
    env['PGPASSWORD'] = pg_password
    
    cmd = [
        '/data1/weixun/postgresql/pg_13/bin/psql',
        '-h', pg_host,
        '-p', pg_port,
        '-U', pg_user,
        '-d', database,
        '-f', sql_file
    ]
    
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"执行失败: {result.stderr}")
        return False
    return True

def fix_sql_file_paths(input_file, output_file, database_name):
    """修改SQL文件中的路径，并删除以 create index 开头的索引创建语句（含多行）"""
    with open(input_file, 'r') as f:
        content = f.read()

    # 先删除 create index / create unique index 开头的整条语句（直到分号）
    lines = content.splitlines()
    new_lines = []
    skipping = False
    for line in lines:
        if not skipping and re.match(r'^\s*create\s+(unique\s+)?index\b', line, flags=re.IGNORECASE):
            # 开始跳过，直到遇到分号为止
            if ';' in line:
                skipping = False
            else:
                skipping = True
            continue
        if skipping:
            if ';' in line:
                skipping = False
            continue
        new_lines.append(line)
    content = '\n'.join(new_lines)

    # 替换路径
    content = content.replace("./", data_dir + database_name + "/")

    # 确保临时目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w') as f:
        f.write(content)

for database_name in database_name_list:
    sql_file = data_dir + database_name + "/" + f"postgres_create_{database_name}.sql"
    
    if not os.path.exists(sql_file):
        print(f"警告: 文件 {sql_file} 不存在，跳过数据库 {database_name}")
        continue
    
    print(f"正在处理数据库: {database_name}")
    
    # 创建修改路径后的临时SQL文件
    temp_sql_file = f"./tmp/temp_{database_name}.sql"
    fix_sql_file_paths(sql_file, temp_sql_file, database_name)
    
    # 执行SQL文件
    if execute_psql(temp_sql_file):
        print(f"数据库 {database_name} 创建成功")
    else:
        print(f"数据库 {database_name} 创建失败")
    
    # 清理临时文件
    # os.remove(temp_sql_file)

print("所有数据库处理完成！")


from idlelib.iomenu import encoding

import pandas as pd
from sqlalchemy import create_engine

# 数据库连接参数（以MySQL为例）
username = 'root'
password = '123456'
host = '192.168.0.189'
port = '3306'
database = 'shale-gas'

# 创建数据库连接引擎
engine = create_engine(f'mysql+pymysql://{username}:{password}@{host}:{port}/{database}')

# 读取SQL表
para_name = 'gas_base_well'  # 替换为你的表名
fit_name = 'gas_well_fit'  # 替换为你的表名
df_para = pd.read_sql_table(para_name, con=engine)
df_fit = pd.read_sql_table(fit_name, con=engine)
df_type = pd.read_csv(r'data/0423-核心区划分.csv',encoding = 'gbk')
df_depth = pd.read_csv(r'data/0423-钻井深度数据.csv',encoding = 'gbk')
df_combine = pd.merge(df_para,df_fit,how='left',on='well_no')
df_combine = pd.merge(df_combine,df_type,how='left',on='well_no')
df_combine = pd.merge(df_combine,df_depth,how='left',on='well_no')
table_name = 'gas_well_para'  # 替换为你希望的SQL表名
df_combine.to_sql(table_name, con=engine, index=False, if_exists='replace')
print(f'Data has been saved to table {table_name} in the database.')


import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sqlalchemy import create_engine

username = 'root'
password = '123456'
host = '192.168.0.189'
port = '3306'
database = 'shale-gas'
engine = create_engine(f'mysql+pymysql://{username}:{password}@{host}:{port}/{database}')

chunk_size = 1000  # 设置每次读取的行数
dataframes = pd.read_csv('data/0504-日产气数据.csv', chunksize=chunk_size)
df_xz = pd.concat(dataframes)
df_xz = df_xz.dropna(axis=1)
df_xz.head()

df = df_xz.copy()
# 按照井号分组处理
grouped = df.groupby('Wells_id')
# 重新编号生产天数
new_df = pd.DataFrame()
for _, group in grouped:
    # 在对 group 进行操作之前，显式创建副本
    group = group.copy()

    # 剔除日产气量为0的数据
    group = group[group['Daily_production'] != 0]

    # 重新编号生产天数，使用 .loc 解决 SettingWithCopyWarning
    group.loc[:, 'New_days'] = (group['Days'] != 0).cumsum()

    # 更新累计产气量列，使用 .loc
    group.loc[:, 'New_cumulative_production'] = group['Cumulative_production']

    # 删除中间有0的数据
    group = group[group['New_days'] != 0]

    # 重新索引
    group.reset_index(drop=True, inplace=True)

    # 将结果合并回到新的DataFrame
    new_df = pd.concat([new_df, group])

df = new_df.copy()
well_id = df['Wells_id'].unique()

def duong_decay(X, a, m):
    t = X[0]
    qi = X[1]
    return qi * t ** (-m) * np.exp(a / (1 - m) * (t ** (1 - m) - 1))

def intial_production(production, a, m, days):
    Q = production
    t = days
    return Q * a / (np.exp((a / (1 - m)) * (t ** (1 - m) - 1)))

def cumulative_production(qi, a, m, t=330 * 20, t1=330):
    Q = (qi / a) * (np.exp((a / (1 - m)) * (t ** (1 - m) - 1)))
    Q1 = (qi / a) * (np.exp((a / (1 - m)) * (t1 ** (1 - m) - 1)))
    return Q - Q1

well_number = []
well_a_values = []
well_m_values = []
well_q_values = []
well_cum = []
well_dur = []
duong_cum = []

for i in well_id:
    well_df = df[df['Wells_id'] == i]
    well_produciton = well_df.Daily_production
    well_days = well_df.New_days
    t_data = np.array(well_days)  # 时间数据
    q_data = np.array(well_produciton.values)  # 对应的产量数据

    qi_fixed = q_data[0]  # 假设qi的值
    q_i = np.repeat(qi_fixed, len(t_data))

    initial_guess = (0.5, 0.5)  # 初始参数值，qi_fixed为已知值
    bounds = ([0, 0], [5, 2])  # 参数的取值范围，a的下界为负无穷，m的取值范围为[0, 1]
    params, covariance = curve_fit(duong_decay, (t_data, q_i), q_data, p0=initial_guess, bounds=bounds, maxfev=10000)
    a_fit, m_fit = params

    well_number.append(i)
    well_a_values.append(a_fit)
    well_m_values.append(m_fit)
    well_q_values.append(qi_fixed)
    well_cum.append(sum(q_data))
    well_dur.append(len(well_days))

    duong_pro = cumulative_production(qi_fixed, a_fit, m_fit, t=len(q_i), t1=1)
    duong_cum.append(duong_pro)

duong_df = pd.DataFrame(
    {'well_no': well_number, 'a_values': well_a_values, 'm_values': well_m_values, 'q': well_q_values, 'duration': well_dur,
     'actual_production': well_cum, 'duong_production': duong_cum})
# 将DataFrame保存为SQL表

table_name = 'gas_well_fit'  # 替换为你希望的SQL表名
duong_df.to_sql(table_name, con=engine, index=False, if_exists='replace')
print(f'Data has been saved to table {table_name} in the database.')
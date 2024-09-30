import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer

# 创建一个连接到数据库的引擎 (以MySQL为例，替换为你的数据库配置)
username = 'root'
password = '123456'
host = '192.168.0.189'
port = '3306'
database = 'shale-gas'
engine = create_engine(f'mysql+pymysql://{username}:{password}@{host}:{port}/{database}')

table_name = 'gas_base_well'  # 替换为你的表名
df = pd.read_sql_table(table_name, con=engine)

# 数据处理（根据需要处理数据）
# 示例：删除空值行


def process_data(df_combine):

    # 移除气井类型为空的数据
    gas_label = 'well_type'
    gas_label = df_combine.well_type
    df_combine = df_combine[pd.notna(gas_label)]
    df_combine.set_index('well_no', inplace=True)

    target_m = 'm_values'
    target_a = 'a_values'
    target_p = 'days330_first_year'

    del_vars = ['m_values','a_values','days330_first_year']
    X_all =  df_combine.drop(del_vars,axis=1)

    # 提取目标变量
    y_all_a = df_combine[target_a]
    y_all_m = df_combine[target_m]
    y_all_p = df_combine[target_p]

    # 编码类别变量
    label_encoder_region = LabelEncoder()
    label_encoder_type = LabelEncoder()

    X_all['well_type'] = label_encoder_type.fit_transform(X_all['well_type'])
    X_all['core_area'] = label_encoder_region.fit_transform(X_all['core_area'])

    # 使用 KNN 填补缺失值
    imputer = KNNImputer(n_neighbors=3)
    imputed_data = imputer.fit_transform(X_all)

    # 转换为 DataFrame 并恢复索引
    X_imputed = pd.DataFrame(imputed_data, columns=X_all.columns)
    X_imputed['well_no'] = X_all.index
    X_imputed.set_index('well_no', inplace=True)

    # 处理后的目标变量
    y_process_a = df_combine[target_a]
    y_process_m = df_combine[target_m]
    y_process_p = df_combine[target_p]

    # 训练集的特征和标签
    Dataset_X = X_imputed[pd.notna(y_process_p.values)]

    Dataset_y_a = y_process_a[pd.notna(y_process_a.values)]
    Dataset_y_m = y_process_m[pd.notna(y_process_m.values)]
    Dataset_y_p = y_process_p[pd.notna(y_process_p.values)]

    return Dataset_X, Dataset_y_a, Dataset_y_m, Dataset_y_p

def importance_calculation(X, y, model):
    scaler = StandardScaler()
    Xm_imputed = scaler.fit_transform(X)
    ym_imputed = y
    model.fit(Xm_imputed, ym_imputed)
    feature_importance = model.feature_importances_
    feature_importance = np.abs(feature_importance) / np.sum(np.abs(feature_importance))

    return feature_importance


def build_ensemble_model(user_selection=None):
    # 模型映射
    model_mapping = {
        "RandomForestRegressor": RandomForestRegressor,
        "XGBRegressor": XGBRegressor,
        "LGBMRegressor": LGBMRegressor,
        "SVR": SVR,
        "MLPRegressor": MLPRegressor,
        "Lasso": Lasso,
        "Ridge": Ridge,
        "GradientBoostingRegressor": GradientBoostingRegressor
    }

    # 将缩写转换为完整的模型名称
    shorthand_mapping = {
        "rf": "RandomForestRegressor",
        "xgb": "XGBRegressor",
        "lgb": "LGBMRegressor",
        "svr": "SVR",
        "mlp": "MLPRegressor",
        "lasso": "Lasso",
        "ridge": "Ridge",
        "gbr": "GradientBoostingRegressor"
    }

    if user_selection is None:
        raise ValueError("user_selection 不能为 None")

    base_learners_config = user_selection.get('base_learners', {})
    final_model_config = user_selection.get('final_model', {})

    # 动态构建基学习器列表
    base_learners = []
    for name, learner_info in base_learners_config.items():
        model_key = learner_info['model']

        # 检查是否传入了完整名称或简写，并映射为简写
        if model_key in model_mapping:  # 如果传入的是完整名称
            model_name = model_key
        else:
            model_name = shorthand_mapping.get(model_key)  # 使用简写映射
            if not model_name:
                raise ValueError(f"模型缩写 '{model_key}' 未在 shorthand_mapping 中找到")

        model_params = learner_info.get('params', {})

        # 使用模型映射获取模型类并实例化
        model_class = model_mapping.get(model_name)
        if model_class:
            model_instance = model_class(**model_params)
            base_learners.append((name, model_instance))
        else:
            raise ValueError(f"模型 {model_name} 在模型映射中找不到。")

    # 动态构建最终模型
    final_model_key = final_model_config.get('model', 'mlp')  # 默认使用 MLP
    final_model_name = shorthand_mapping.get(final_model_key, final_model_key)  # 获取最终模型名称，支持缩写或完整名称
    if not final_model_name:
        raise ValueError(f"最终模型缩写 '{final_model_key}' 未在 shorthand_mapping 中找到")

    final_model_params = final_model_config.get('params', {})
    final_model_class = model_mapping.get(final_model_name)

    if final_model_class:
        final_model_instance = final_model_class(**final_model_params)
    else:
        raise ValueError(f"最终模型 {final_model_name} 在模型映射中找不到。")

    # 构建 StackingRegressor 模型
    model = StackingRegressor(estimators=base_learners, final_estimator=final_model_instance)

    return model


'''
def cross_validate_and_predict(df, p_model, a_model, m_model, n_splits=5):
    """
    使用 K 折交叉验证对所有样本进行预测，并将预测结果保存到 DataFrame 中。

    Parameters:
    df (pd.DataFrame): 输入数据 DataFrame
    target_column (str): 目标列的名称
    model (StackingRegressor): 已通过 build_ensemble_model 构建好的模型
    n_splits (int): KFold 的分割数量，默认为 5

    Returns:
    df_with_predictions (pd.DataFrame): 包含预测结果的新 DataFrame
    """
    target_var = ['days330_first_year','a_values','m_values']
    # 提取特征和目标变量
    X = df.drop(columns=target_var)
    y_p = df['days330_first_year']
    y_a = df['a_values']
    y_m = df['m_values']

    # 初始化标准化器
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 初始化交叉验证
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # 用于保存最终的预测结果
    y_p_pred_total = np.zeros(len(y_p))
    y_a_pred_total = np.zeros(len(y_a))
    y_m_pred_total = np.zeros(len(y_m))

    # 进行交叉验证
    for train_idx, test_idx in kf.split(X_scaled):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_p_train, y_p_test = y_p.iloc[train_idx], y_p.iloc[test_idx]
        y_a_train, y_a_test = y_a.iloc[train_idx], y_a.iloc[test_idx]
        y_m_train, y_m_test = y_m.iloc[train_idx], y_m.iloc[test_idx]

        # 模型训练
        p_model.fit(X_train, y_p_train)
        a_model.fit(X_train, y_a_train)
        m_model.fit(X_train, y_m_train)

        # 对测试集进行预测
        y_p_pred = p_model.predict(X_test)
        y_a_pred = a_model.predict(X_test)
        y_m_pred = m_model.predict(X_test)

        # 将对应的预测值保存到总结果中
        y_p_pred_total[test_idx] = y_p_pred
        y_a_pred_total[test_idx] = y_a_pred
        y_m_pred_total[test_idx] = y_m_pred

    # 将预测结果添加到原 DataFrame 中
    df_with_predictions = df.copy()
    df_with_predictions['Predicted_330'] = y_p_pred_total
    df_with_predictions['Predicted_a'] = y_a_pred_total
    df_with_predictions['Predicted_m'] = y_m_pred_total

    return df_with_predictions
'''

def cross_validate_and_predict(df, p_model, a_model, m_model, n_splits=5):
    """
    使用 K 折交叉验证对 a_values 和 m_values 不为空的样本进行预测，并用完整数据建模来预测缺失的 a_values 和 m_values。

    Parameters:
    df (pd.DataFrame): 输入数据 DataFrame
    p_model, a_model, m_model: 已通过 build_ensemble_model 构建好的模型
    n_splits (int): KFold 的分割数量，默认为 5

    Returns:
    df_with_predictions (pd.DataFrame): 包含预测结果和补全的 DataFrame
    """
    target_var = ['days330_first_year', 'a_values', 'm_values']
    # 提取特征和目标变量
    X = df.drop(columns=target_var)
    y_p = df['days330_first_year']
    y_a = df['a_values']
    y_m = df['m_values']

    # 只对 a_values 和 m_values 不为空的样本进行交叉验证
    non_nan_a_idx = y_a.notna()
    non_nan_m_idx = y_m.notna()

    X_a_non_nan, y_a_non_nan = X[non_nan_a_idx], y_a[non_nan_a_idx]
    X_m_non_nan, y_m_non_nan = X[non_nan_m_idx], y_m[non_nan_m_idx]

    # 初始化标准化器
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_a_scaled = scaler.transform(X_a_non_nan)
    X_m_scaled = scaler.transform(X_m_non_nan)

    # 进行交叉验证并保存结果
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # 用于保存最终的交叉验证预测结果
    y_p_pred_cv = np.zeros(len(y_p))
    y_a_pred_cv = np.zeros(len(y_a_non_nan))
    y_m_pred_cv = np.zeros(len(y_m_non_nan))

    # 交叉验证用于 y_p
    for train_idx, test_idx in kf.split(X_scaled):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_p_train, y_p_test = y_p.iloc[train_idx], y_p.iloc[test_idx]

        # 训练和预测 Predicted_330 (days330_first_year)
        p_model.fit(X_train, y_p_train)
        y_p_pred_cv[test_idx] = p_model.predict(X_test)

    # 交叉验证用于 a_values 非空样本
    for train_idx, test_idx in kf.split(X_a_scaled):
        X_train_a, X_test_a = X_a_scaled[train_idx], X_a_scaled[test_idx]
        y_a_train, y_a_test = y_a_non_nan.iloc[train_idx], y_a_non_nan.iloc[test_idx]

        a_model.fit(X_train_a, y_a_train)
        y_a_pred_cv[test_idx] = a_model.predict(X_test_a)

    # 交叉验证用于 m_values 非空样本
    for train_idx, test_idx in kf.split(X_m_scaled):
        X_train_m, X_test_m = X_m_scaled[train_idx], X_m_scaled[test_idx]
        y_m_train, y_m_test = y_m_non_nan.iloc[train_idx], y_m_non_nan.iloc[test_idx]

        m_model.fit(X_train_m, y_m_train)
        y_m_pred_cv[test_idx] = m_model.predict(X_test_m)

    # 将交叉验证的预测结果添加到原 DataFrame 中
    df_with_predictions = df.copy()
    df_with_predictions['Predicted_330'] = y_p_pred_cv
    df_with_predictions.loc[non_nan_a_idx, 'Predicted_a'] = y_a_pred_cv
    df_with_predictions.loc[non_nan_m_idx, 'Predicted_m'] = y_m_pred_cv

    # 使用所有 a_values 和 m_values 不为空的样本重新训练模型
    a_model.fit(X_a_scaled, y_a_non_nan)
    m_model.fit(X_m_scaled, y_m_non_nan)

    # 使用训练好的模型对 a_values 和 m_values 为空的样本进行预测
    missing_a_idx = y_a.isna()
    missing_m_idx = y_m.isna()

    if missing_a_idx.any():
        X_missing_a = scaler.transform(X[missing_a_idx])
        df_with_predictions.loc[missing_a_idx, 'Predicted_a'] = a_model.predict(X_missing_a)

    if missing_m_idx.any():
        X_missing_m = scaler.transform(X[missing_m_idx])
        df_with_predictions.loc[missing_m_idx, 'Predicted_m'] = m_model.predict(X_missing_m)

    # 补全缺失值：对于缺失的 a_values 和 m_values 用预测值进行填补
    df_with_predictions['a_values'] = df_with_predictions['a_values'].fillna(df_with_predictions['Predicted_a'])
    df_with_predictions['m_values'] = df_with_predictions['m_values'].fillna(df_with_predictions['Predicted_m'])

    return df_with_predictions

def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true))

def medianape(y_true, y_pred):
    return np.median(np.abs((y_pred - y_true) / y_true))

def duong_decay(X, a, m):
    t = X[0]
    qi = X[1]
    return qi * t ** (-m) * np.exp(a / (1 - m) * (t ** (1 - m) - 1))

def intial_production_calculation(production, a, m, days):
    Q = production
    t = days
    return Q * a / (np.exp((a / (1 - m)) * (t ** (1 - m) - 1)))

def cumulative_production(qi, a, m, t=330 * 6, t1=330):
    Q = (qi / a) * (np.exp((a / (1 - m)) * (t ** (1 - m) - 1)))
    Q1 = (qi / a) * (np.exp((a / (1 - m)) * (t1 ** (1 - m) - 1)))
    return Q - Q1

def eur_function(first_production, a_fit, m_fit, dur_eur=20):
    start_value = 331
    end_value = 330 * dur_eur
    dur = np.arange(start_value, end_value + 1)
    inital_q = intial_production_calculation(first_production, a_fit, m_fit, days=330)
    daily_pro = duong_decay((dur, np.repeat(inital_q, len(dur))), a_fit, m_fit)
    year_production = sum_year_production(daily_pro, 330)
    year_production_all = np.insert(year_production, 0, first_production)
    return year_production_all, sum(year_production_all)

def sum_year_production(data, n=330):
    sums = []
    i = 0
    while i < len(data):
        chunk = data[i:i + n]
        sums.append(sum(chunk))
        i += n
    return sums

def revenue_function(year_production, price=0.989, rate=0.06):
    sy = 0
    for j in range(len(year_production)):
        cl = year_production[j]
        sy = sy + ((cl * price) / ((1 + rate) ** j))
    return sy

def cost_function(df, var, para, inter=2120.96):
    cost_df = df[var].values
    cost_all = np.dot(cost_df, para)[0] + inter

    return cost_all

def main():
    # 创建一个连接到数据库的引擎 (以MySQL为例，替换为你的数据库配置)
    username = 'root'
    password = '123456'
    host = '192.168.0.189'
    port = '3306'
    database = 'shale-gas'
    engine = create_engine(f'mysql+pymysql://{username}:{password}@{host}:{port}/{database}')

    # 读取数据
    table_name = 'gas_well_para'  # 替换为你的表名
    noprocess_var = ['id', 'update_by', 'update_time', 'sys_org_code', 'create_by',
       'create_time','actual_production', 'duong_production','lng', 'lat', 'well_state']
    df = pd.read_sql_table(table_name, con=engine)
    df = df.drop(noprocess_var, axis=1)

    # 处理数据
    target = ''  # 替换为你的目标列名
    Dataset_X, Dataset_y_a, Dataset_y_m, Dataset_y_p = process_data(df)


    lr = 0.01
    user_selection = {
        "base_learners": {
            "rf": {"model": "rf", "params": {"n_estimators": 100, "random_state": 42}},
            "xgb": {"model": "xgb", "params": {"n_estimators": 100, "learning_rate": lr}},  # 调整xgb的学习率
            "lgb": {"model": "lgb", "params": {"n_estimators": 100, "learning_rate": lr}},  # 调整lgb的学习率
        },
        "final_model": {
            "model": "mlp",  # 使用MLP作为元学习器
            "params": {
                "hidden_layer_sizes": (100, 50),
                "activation": 'relu',
                "solver": 'adam',
                "max_iter": 200
            }
        }
    }

    # 构建模型
    p_stack_model = build_ensemble_model(user_selection=user_selection)
    a_stack_model = build_ensemble_model(user_selection=user_selection)
    m_stack_model = build_ensemble_model(user_selection=user_selection)

    df_combined = pd.concat([Dataset_X, Dataset_y_a, Dataset_y_m, Dataset_y_p], axis=1)
    original_columns = list(Dataset_X.columns)
    new_columns = ['a_values', 'm_values', 'days330_first_year']
    # 组合原来的列名和新增加的列名
    df_combined.columns = original_columns + new_columns
    df_with_predictions = cross_validate_and_predict(df_combined, p_model = p_stack_model, a_model = a_stack_model,
                                                     m_model = m_stack_model,
                                                     n_splits=5)

    cost_para = [-23.66, 0.55, 0.64, 34.28, 12.89, -21.62, 1.41]

    # 计算每一个气井的EUR
    well_ids =  df_with_predictions.index.unique()
    eur_values = []
    for well_id in well_ids:
        well_data = df_with_predictions[df_with_predictions.index == well_id]
        a_fit = well_data['Predicted_a'].values[0]
        m_fit = well_data['Predicted_m'].values[0]
        first_production = well_data['Predicted_330'].values[0]
        _, eur = eur_function(first_production, a_fit, m_fit)
        eur_values.append(eur)

    df_with_predictions['Predicted_EUR'] = eur_values

if __name__ == "__main__":
    main()

# 将DataFrame保存为SQL表
#table_name = 'gas_evaluate'  # 替换为你希望的SQL表名
#df_evaluate.to_sql(table_name, con=engine, index=False, if_exists='replace')
#print(f'Data has been saved to table {table_name} in the database.')

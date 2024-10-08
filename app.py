import datetime
from math import sqrt

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import uuid
from util import pao_pai, zeng_ya, lx_qi_ju, jd_qi_ju
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import uuid
from util import pao_pai, zeng_ya, lx_qi_ju, jd_qi_ju
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text
from flask import Flask, request, jsonify
import pandas as pd
from EUR_predict import build_ensemble_model, process_data, cross_validate_and_predict, eur_function
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
import math
from math import sqrt

app = Flask(__name__)
# 配置数据库URI
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:123456@192.168.0.189:3306/shale-gas'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
db = SQLAlchemy(app)

# Database connection
username = 'root'
password = '123456'
host = '192.168.0.189'
port = '3306'
database = 'shale-gas'


# Establish DB connection
def get_db_connection():
    engine = create_engine(f'mysql+pymysql://{username}:{password}@{host}:{port}/{database}')
    return engine


# Model mapping (simplified to use shorthand for models)
model_mapping = {
    "rf": {"model": "RandomForestRegressor", "params": {"n_estimators": 100, "random_state": 42}},
    "xgb": {"model": "XGBRegressor", "params": {"n_estimators": 100}},
    "lgb": {"model": "LGBMRegressor", "params": {"n_estimators": 100}},
}

# Final model mapping
final_model_mapping = {
    "mlp": {"model": "MLPRegressor",
            "params": {"hidden_layer_sizes": [100, 50], "activation": "relu", "solver": "adam", "max_iter": 200}},
    "lasso": {"model": "Lasso", "params": {"alpha": 0.01}},
    "ridge": {"model": "Ridge", "params": {"alpha": 0.01}}
}


class GasProductionWell(db.Model):
    id = db.Column(db.String, primary_key=True)  # 这里 id 是主键
    well_no = db.Column(db.String)
    collect_date = db.Column(db.Date)
    production_gas_day = db.Column(db.Float)
    # 可以添加其他方法和属性


class GasProductionIncrease(db.Model):
    id = db.Column(db.String, primary_key=True)  # 这里 id 是主键
    well_no = db.Column(db.String)
    platform_no = db.Column(db.String)
    begin_time = db.Column(db.Date)
    end_time = db.Column(db.Date)
    days = db.Column(db.String)
    before_pro = db.Column(db.Float)
    after_pro = db.Column(db.Float)
    amplify = db.Column(db.Float)
    absolute_inc = db.Column(db.Float)
    production_inc = db.Column(db.Float)
    # 可以添加其他方法和属性


class GasCost(db.Model):
    platform_no = db.Column(db.String, primary_key=True)
    pp_cost = db.Column(db.Float)
    lxqj_cost = db.Column(db.Float)
    jdqj_cost = db.Column(db.Float)
    zy_cost = db.Column(db.Float)


class GasBaseCompressor(db.Model):  # 压缩机
    id = db.Column(db.String, primary_key=True)
    update_by = db.Column(db.String, default='', nullable=False, comment='更新人')
    update_time = db.Column(db.DateTime, comment='更新日期')
    sys_org_code = db.Column(db.String, default='', nullable=False, comment='所属部门')
    create_by = db.Column(db.String, default='', nullable=False, comment='创建人')
    create_time = db.Column(db.DateTime, comment='创建日期')
    group_model = db.Column(db.String, comment='机组型号')
    compressor_model = db.Column(db.String, comment='压缩机型号')
    intake_pressure_min = db.Column(db.Float, comment='最小进气压力Mpa')
    intake_pressure_max = db.Column(db.Float, comment='最大进气压力Mpa')
    intake_pressure_optimal = db.Column(db.String, comment='进气压力最优')
    intake_temperature = db.Column(db.Float, comment='进气温度℃')
    exhaust_gas_min = db.Column(db.Float, comment='最小排气量Nm3/d')
    exhaust_gas_max = db.Column(db.Float, comment='最大排气量Nm3/d')
    exhaust_pressure = db.Column(db.Float, comment='排气压力Mpa')
    exhaust_temperature = db.Column(db.Float, comment='排气温度℃')
    compressor_columns = db.Column(db.Float, comment='压缩机列数')
    compressor_cylinders = db.Column(db.Float, comment='压缩机气缸数')
    rated_power = db.Column(db.Float, comment='电机额定功率kw')
    rated_rotate_speed = db.Column(db.Float, comment='电机额定转速rpm')
    weight = db.Column(db.Float, comment='重量')
    size = db.Column(db.String, comment='尺寸(长×宽×高)')


class GasCompressorWorkcondition(db.Model):  # 压缩机工况表
    id = db.Column(db.String, primary_key=True)
    update_by = db.Column(db.String, default='', nullable=False, comment='更新人')
    update_time = db.Column(db.DateTime, comment='更新日期')
    sys_org_code = db.Column(db.String, default='', nullable=False, comment='所属部门')
    create_by = db.Column(db.String, default='', nullable=False, comment='创建人')
    create_time = db.Column(db.DateTime, comment='创建日期')
    compressor_id = db.Column(db.String, comment='所属压缩机组(关联主键)')
    rotate_speed = db.Column(db.Float, comment='转速rpm')
    grade1_intake_pressure = db.Column(db.Float, comment='一级进气压力mpa')
    intake_temperature = db.Column(db.Float, comment='进气温度℃')
    exhaust_gas = db.Column(db.Float, comment='排气量Nm3/d')
    power = db.Column(db.Float, comment='功率kw')
    power_load_rate = db.Column(db.Float, comment='轴功率负荷率%')
    grade1_exhaust_pressure = db.Column(db.Float, comment='一级排气压力')
    grade2_exhaust_pressure = db.Column(db.Float, comment='二级排气压力')
    grade3_exhaust_pressure = db.Column(db.Float, comment='三级排气压力')
    exhaust_temperature = db.Column(db.Float, comment='排气温度')


@app.route('/')
def index():
    return render_template('form.html')


@app.route('/deal', methods=['POST'])  # 计算增产气量
def process_form():
    # pao_pai('alldata-2024.1.2.csv', '☆泡排台账-2023.12.31（川庆）.csv', 0.030, 30, '泡排增产气量.csv')
    # 确保在应用上下文中执行数据库操作
    with app.app_context():
        # 假设你想查询所有记录
        results = GasProductionWell.query.all()
        # 将结果转换为字典列表
        data = [
            {'id': result.id, '井号': result.well_no, '日期': result.collect_date, '日产气': result.production_gas_day}
            for result in results]
        # 使用 pandas DataFrame 构造函数将字典列表转换为 DataFrame
        df_rcq = pd.DataFrame(data)

    selectOption = request.form.get('selectOption')
    pppath = request.files.get('pppath')
    dr = float(request.form.get('dr'))
    d = int(request.form.get('d'))
    countdays = int(request.form.get('countdays'))
    # price = float(request.form.get('price', '1.37'))

    print(selectOption, pppath, dr, d, countdays)

    if selectOption == '泡排':
        df_pp = pao_pai(df_rcq, pppath, dr, d, countdays)
        df = df_pp[['井号', '平台', '施工日期', '停止施工日期或当前日期',
                    '统计天数', '措施前平均日产', '措施后平均日产', '增幅',
                    '绝对增加量', '增产气量']]

    elif selectOption == '增压':
        df_zy = zeng_ya(df_rcq, pppath, dr, d, countdays)
        df = df_zy[['井号', '平台', '增压日期', '增压停止日期',
                    '增压天数\n（天）', '措施前平均日产', '措施后平均日产', '增幅',
                    '绝对增加量', '增产气量']]

    elif selectOption == '连续气举':
        df_lx_qi = lx_qi_ju(df_rcq, pppath, dr, d, countdays)
        df = df_lx_qi[['井号', '平台', '最初施工日期', '最后施工日期',
                       '施工天数', '措施前平均日产', '措施后平均日产', '增幅',
                       '绝对增加量', '增产气量']]

    elif selectOption == '间断气举':
        df_jd_qi = jd_qi_ju(df_rcq, pppath, dr, d, countdays)
        df = df_jd_qi[['井号', '平台', '最初施工日期', '最后施工日期',
                       '施工天数', '措施前平均日产', '措施后平均日产', '增幅',
                       '绝对增加量', '增产气量']]

    # df['inc_output'] = df['production_inc'] * price
    df['id'] = df.index.map(lambda _: str(uuid.uuid4()))

    # df.columns = ['井号', '平台', '开始施工日期', '结束施工日期',
    #               '施工天数', '措施前平均日产', '措施后平均日产', '增幅',
    #               '绝对增加量', '增产气量']
    df.columns = ['well_no', 'platform_no', 'begin_time', 'end_time',
                  'days', 'before_pro', 'after_pro', 'amplify',
                  'absolute_inc', 'production_inc', 'id']
    df.loc[df['platform_no'] == '威208', 'platform_no'] = '威204H43'
    df.loc[df['platform_no'] == '威209', 'platform_no'] = '威204H62'
    # 清空表中的数据
    with app.app_context():
        db.session.execute(text("DELETE FROM gas_production_increase;"))
        db.session.commit()  # 提交事务
    df.to_sql('gas_production_increase', con=db.engine, index=False, if_exists='append')

    # df.to_csv(savepath, encoding='utf-8-sig', index=False)

    json_response = {
        "success": True,
        "message": "",
        "code": 200,
        "result": {
        }
    }
    return json_response
    # return redirect(url_for('index'))  # 完成后重定向到首页


# 确保分页参数有默认值
def get_pagination_args():  # 分页默认值
    pageNo = request.args.get('page', 1, type=int)
    pageSize = request.args.get('per_page', 10, type=int)
    return pageNo, pageSize


@app.route('/getIncreaseList', methods=['GET'])  # 获取增产气量
def getIncreaseList():
    try:
        pageNo = int(request.args.get('pageNo'))
        pageSize = int(request.args.get('pageSize'))
    except:
        pageNo, pageSize = get_pagination_args()
    with app.app_context():
        # 使用paginate方法进行分页
        pagination = GasProductionIncrease.query.paginate(page=pageNo, per_page=pageSize, error_out=False)
        items = pagination.items

        # 将结果转换为字典列表
        data = [
            {'id': result.id, 'wellNo': result.well_no, 'platformNo': result.platform_no,
             'beginTime': result.begin_time.strftime('%Y-%m-%d')
                , 'endTime': result.end_time.strftime('%Y-%m-%d'), 'days': result.days, 'beforePro': result.before_pro,
             'afterPro': result.after_pro, 'amplify': result.amplify, 'absoluteInc': result.absolute_inc,
             'productionInc': result.production_inc
             }
            for result in items]
    json_response = {
        "success": True,
        "message": "",
        "code": 200,
        "result": {
            "records": data,
            # 可以在响应中包含分页信息
            'total': pagination.total,
            'size': pageSize,
            'current': pageNo,
            "orders": [],
            "optimizeCountsql": True,
            "searchCount": "",
            "countId": None,
            "maxLimit": None,
            "pages": pagination.pages
        }
    }
    return json_response


@app.route('/getIncreasePlatformList', methods=['GET', 'POST'])  # 计算并获取投产比
def getIncreasePlatformList():
    selectOption = request.args.get('selectOption', '泡排')
    with app.app_context():
        # 假设你想查询所有记录
        results = GasProductionIncrease.query.all()
        # 将结果转换为字典列表
        data = [
            {'platform_no': result.platform_no, 'production_inc': result.production_inc}
            for result in results]
        data = pd.DataFrame(data)
        production_sums = data.groupby('platform_no')['production_inc'].sum().reset_index()

        results = GasCost.query.all()
        # 将结果转换为字典列表
        data = [
            {'platform_no': result.platform_no, 'pp_cost': result.pp_cost, 'lxqj_cost': result.lxqj_cost
                , 'jdqj_cost': result.jdqj_cost, 'zy_cost': result.zy_cost}
            for result in results]
        gas_cost = pd.DataFrame(data)

    if selectOption == '泡排':
        gas_cost = gas_cost[['platform_no', 'pp_cost']]

    elif selectOption == '增压':
        gas_cost = gas_cost[['platform_no', 'zy_cost']]

    elif selectOption == '连续气举':
        gas_cost = gas_cost[['platform_no', 'lxqj_cost']]

    elif selectOption == '间断气举':
        gas_cost = gas_cost[['platform_no', 'jdqj_cost']]

    production_sums['output'] = production_sums['production_inc'] * 1.37
    input_output = pd.merge(production_sums, gas_cost, on=['platform_no'], how='left')
    input_output.columns = ['platform_no', 'production_inc', 'output', 'input']
    input_output['production_ratio'] = np.where(input_output['input'] == 0, 0,
                                                input_output['output'] / input_output['input'])

    dataJson = [
        {'platformNo': row.platform_no, 'production_inc': row.production_inc,
         'output': row.output, 'input': row.input, 'production_ratio': row.production_ratio}
        for row in input_output.itertuples(index=False)
    ]
    # 清空表中的数据
    with app.app_context():
        db.session.execute(text("DELETE FROM gas_production_input_output;"))
        db.session.commit()  # 提交事务
    input_output.to_sql('gas_production_input_output', con=db.engine, index=False, if_exists='append')

    json_response = {
        "success": True,
        "message": "",
        "code": 200,
        "result": {
            "records": dataJson
        }
    }
    return json_response


# 定义一个函数来计算距离
def calculate_distance(row, target_intake, target_exhaust, target_gas):
    return sqrt(
        (row.grade1_intake_pressure - target_intake) ** 2 +
        (row.grade3_exhaust_pressure - target_exhaust) ** 2 +
        (row.exhaust_gas - target_gas) ** 2
    )


@app.route('/getCompressor', methods=['GET'])  # 匹配压缩机
def getCompressor():  # 输入进气压力 排气压力 排气量 --> 满足条件的多个压缩机
    intake_pressure = float(request.args.get('intake_pressure', 2))  # 进气压力
    exhaust_pressure = float(request.args.get('exhaust_pressure', 6))  # 排气压力
    exhaust_gas = float(request.args.get('exhaust_gas', 3))  # 排气量
    data = matchCompressor(intake_pressure, exhaust_pressure, exhaust_gas)
    json_response = {
        "success": True,
        "message": "",
        "code": 200,
        "result": {
            "records": data,
        }
    }
    return json_response


def matchCompressor(intake_pressure, exhaust_pressure, exhaust_gas):  # 输入进气压力 排气压力 排气量 --> 满足条件的多个压缩机
    # 构建查询条件
    query = GasBaseCompressor.query.filter(
        GasBaseCompressor.intake_pressure_min <= intake_pressure,
        GasBaseCompressor.intake_pressure_max >= intake_pressure,
        GasBaseCompressor.exhaust_pressure >= exhaust_pressure,
        GasBaseCompressor.exhaust_gas_min <= exhaust_gas,
        GasBaseCompressor.exhaust_gas_max >= exhaust_gas
    )
    with app.app_context():
        # 执行查询并获取结果
        compressors = query.all()
        compressor_ids = [result.id for result in compressors]
        # 使用in_操作符进行查询
        results = GasCompressorWorkcondition.query.filter(
            GasCompressorWorkcondition.compressor_id.in_(compressor_ids)).all()

    # 初始化最小距离和最佳匹配行
    min_distance = float('inf')
    best_match = None
    # 遍历查询结果，找到最接近的匹配
    for row in results:
        distance = calculate_distance(row, intake_pressure, exhaust_pressure, exhaust_gas)
        if distance < min_distance or (
                distance == min_distance and row.power < (best_match.power if best_match else float('inf'))):
            min_distance = distance
            best_match = row

    best_match_compressor_id = best_match.compressor_id
    for result in compressors:
        if result.id == best_match_compressor_id:
            data = [{
                'id': result.id,
                'updateBy': result.update_by,
                'updateTime': result.update_time.isoformat() if result.update_time else None,
                'sysOrgCode': result.sys_org_code,
                'createBy': result.create_by,  # 同样，这个通常也会保持 createBy
                'createTime': result.create_time.isoformat() if result.create_time else None,
                'groupModel': result.group_model,
                'compressorModel': result.compressor_model,
                'intakePressureMin': result.intake_pressure_min,
                'intakePressureMax': result.intake_pressure_max,
                'intakePressureOptimal': result.intake_pressure_optimal,
                'intakeTemperature': result.intake_temperature,
                'exhaustGasMin': result.exhaust_gas_min,
                'exhaustGasMax': result.exhaust_gas_max,
                'exhaustPressure': result.exhaust_pressure,
                'exhaustTemperature': result.exhaust_temperature,
                'compressorColumns': result.compressor_columns,
                'compressorCylinders': result.compressor_cylinders,
                'ratedPower': result.rated_power,
                'ratedRotateSpeed': result.rated_rotate_speed,
                'weight': result.weight,
                'size': result.size,
            }]
    return data

@app.route('/predict', methods=['GET','POST'])
def predict():
    try:
        # Retrieve form data
        selected_models = request.form.getlist('models') or ['lgb']
        final_model_choice = request.form.get('finalModel', 'mlp')  # Default to "mlp" if not provided
        learning_rate = float(request.form.get('learningRate', 0.01))  # Default learning rate is 0.01
        n_splits = int(request.form.get('nSplits', 5))  # Default to 5 splits
        predict_variables = request.form.getlist('predictVariables') or ['brmc1_layer']
        page = int(request.form.get('pageNo', 1))  # Default to page 1
        size = int(request.form.get('pageSize', 10))  # Default page size is 10

        #if not predict_variables:
            #return jsonify({"state": "error", "message": "No target variable provided"}), 400

        # Build base learners
        base_learners_config = {}
        for model_key in selected_models:  # Models are like ["rf", "xgb"]
            if model_key in model_mapping:  # Check if model exists in the mapping
                model_info = model_mapping[model_key]
                # If the model supports a learning rate, adjust it
                if 'learning_rate' in model_info['params']:
                    model_info['params']['learning_rate'] = learning_rate
                base_learners_config[model_key] = model_info
            else:
                return jsonify({"state": "error", "message": f"Model '{model_key}' not found in model mapping."}), 400

        # Select and instantiate the final model
        final_model_info = final_model_mapping.get(final_model_choice, final_model_mapping['mlp'])

        # Build the user_selection dictionary
        user_selection = {
            "base_learners": base_learners_config,
            "final_model": final_model_info
        }

        # Build the ensemble model
        stacking_model = build_ensemble_model(user_selection=user_selection)

        # Connect to the database and retrieve data
        engine = get_db_connection()
        table_name = 'gas_well_para'
        noprocess_var = ['id', 'update_by', 'update_time', 'sys_org_code', 'create_by', 'create_time', 'actual_production', 'duong_production', 'lng', 'lat', 'well_state']
        df = pd.read_sql_table(table_name, con=engine)
        df = df.drop(noprocess_var, axis=1)

        base_variables = ['well_no','well_type','m_values','a_values', 'days330_first_year','core_area']
        df = df[predict_variables+base_variables]

        # Data processing
        Dataset_X, Dataset_y_a, Dataset_y_m, Dataset_y_p = process_data(df)
        df_combined = pd.concat([Dataset_X, Dataset_y_a, Dataset_y_m, Dataset_y_p], axis=1)
        original_columns = list(Dataset_X.columns)
        new_columns = ['a_values', 'm_values', 'days330_first_year']
        df_combined.columns = original_columns + new_columns

        # Perform cross+alidation and prediction
        df_with_predictions = cross_validate_and_predict(
            df_combined, p_model=stacking_model, a_model=stacking_model, m_model=stacking_model, n_splits = n_splits)

        # 计算每一个气井的EUR
        well_ids = df_with_predictions.index.unique()
        eur_values = []
        year_production_values = {f'Year_{i + 1}_Production': [] for i in range(20)}  # 为19年生成列
        for well_id in well_ids:
            well_data = df_with_predictions[df_with_predictions.index == well_id]
            a_fit = well_data['Predicted_a'].values[0]
            m_fit = well_data['Predicted_m'].values[0]
            first_production = well_data['Predicted_330'].values[0]
            year_production, eur = eur_function(first_production, a_fit, m_fit)
            eur_values.append(eur)
            for i in range(20):
                year_production_values[f'Year_{i + 1}_Production'].append(year_production[i])

        for year_col, year_values in year_production_values.items():
            df_with_predictions[year_col] = year_values
        df_with_predictions['Predicted_EUR'] = eur_values
        df_with_predictions = df_with_predictions.reset_index()
        columns_order = ['well_no'] + [col for col in df_with_predictions.columns if col != 'well_no']
        df_with_predictions = df_with_predictions[columns_order]

        def mean_relative_error(y_true, y_pred):
            return np.abs((y_true - y_pred) / y_true)

        responses_variable = ['well_no', 'Predicted_330', 'days330_first_year','Predicted_EUR'] + [f'Year_{i + 2}_Production' for i in range(19)]
        df_response = df_with_predictions[responses_variable]

        df_response['MAE'] = df_response.apply(lambda row: mean_absolute_error([row['days330_first_year']], [row['Predicted_330']]), axis=1)
        df_response['MRE'] = df_response.apply(
            lambda row: (row['MAE']/row['days330_first_year']), axis=1)

        # Pagination logic
        total = len(df_response)  # Total number of wells
        pages = math.ceil(total / size)  # Total number of pages
        start = (page - 1) * size  # Start index
        end = start + size  # End index

        # Paginated result
        df_paginated = df_response.iloc[start:end]
        # Convert to dictionary format with record-based representation

        write_name = 'gas_eur_predict'  # 替换为你希望的SQL表名
        df_response['id'] = [str(uuid.uuid4()) for _ in range(len(df_response))]  # 生成唯一的 ID
        df_response['update_by'] = 'gas-admin'  # 更新人，默认为 system 或通过其他方式动态获取
        df_response['update_time'] = datetime.now()  # 当前时间作为更新时间
        df_response['sys_org_code'] = 'A11'  # 部门编号
        df_response['create_by'] = 'gas-admin'  # 创建人
        df_response['create_time'] = datetime.now()  # 创建时间

        metadata = MetaData()
        table = Table(write_name, metadata, autoload_with=engine)
        sql_columns = [column.name for column in table.columns]
        df_response = df_response[sql_columns]

        df_response['update_time'] = pd.to_datetime(df_response['update_time'])
        df_response['create_time'] = pd.to_datetime(df_response['create_time'])

        df_response['well_no'] = df_response['well_no'].astype(str)
        df_response['id'] = df_response['id'].astype(str)
        df_response['update_by'] = df_response['update_by'].astype(str)
        df_response['sys_org_code'] = df_response['sys_org_code'].astype(str)
        df_response['create_by'] = df_response['create_by'].astype(str)

        columns = [
            {"title": "井号", "dataIndex": "well_no"},
            {"title": "a值", "dataIndex": "a_values"},
            {"title": "m值", "dataIndex": "m_values"},
            {"title": "首年实际累产", "dataIndex": "Days330_first_year"},
            {"title": "首年预测累产", "dataIndex": "Predicted_330"},
            {"title": "EUR预测", "dataIndex": "Predicted_EUR"},
            {"title": "绝对误差值", "dataIndex": "MAE"},
            {"title": "绝对误差率", "dataIndex": "MRE"}
        ]
        for year in range(2, 20):  # 第2年到第19年
            columns.append({
                "title": f"第{year}年产量",
                "dataIndex": f"Year_{year}_Production"
            })


        with engine.connect() as connection:
            connection.execute(text(f"TRUNCATE TABLE {write_name}"))
        # 然后使用 pandas 的 to_sql() 插入新的数据
        df_response.to_sql(write_name, con=engine, index=False, if_exists='append')
        records = df_paginated.to_dict(orient='records')

        # Return the result as JSON
        response = {
            "state": "success",
            "code": 200,
            "result": {
                "records": records,  # Actual data records
                "total": total,  # Total number of wells
                "size": size,  # Number of wells per page
                "pages": pages,  # Total number of pages
                "current": page, # Current page number
                "columns": columns  # 将表头数据返回
            }
        }
        return jsonify(response)

    except ValueError as ve:
        return jsonify({"state": "error", "message": f"ValueError: {str(ve)}"}), 400
    except Exception as e:
        return jsonify({"state": "error", "message": f"An error occurred: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

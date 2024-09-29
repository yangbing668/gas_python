from math import sqrt

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import uuid
from util import pao_pai, zeng_ya, lx_qi_ju, jd_qi_ju
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text

app = Flask(__name__)
# 配置数据库URI
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:123456@192.168.0.189:3306/shale-gas'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
db = SQLAlchemy(app)


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


@app.route('/getIncreasePlatformList', methods=['GET'])  # 计算并获取投产比
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
    json_response = {
        "success": True,
        "message": "",
        "code": 200,
        "result": {
            "records": data,
        }
    }
    return json_response


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

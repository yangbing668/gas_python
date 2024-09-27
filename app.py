import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash,jsonify
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

@app.route('/')
def index():
    return render_template('form.html')

@app.route('/deal', methods=['POST'])
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


    # rcqpath = request.form['rcqpath']

    selectOption = request.form['selectOption']
    # pppath = request.form['pppath']
    pppath = request.files['pppath']

    dr = float(request.form['dr'])
    d = int(request.form['d'])
    countdays = int(request.form['countdays'])
    # savepath = request.form['savepath']
    # print(rcqpath, selectOption, pppath, dr, d, countdays, savepath)
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

    df['id'] = df.index.map(lambda _: str(uuid.uuid4()))
    # print(df)

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

    # flash(message)  # 使用 Flask 的 flash 机制来显示消息
    return "success"
    # return redirect(url_for('index'))  # 完成后重定向到首页

# 确保分页参数有默认值
def get_pagination_args():
    pageNo = request.args.get('page', 1, type=int)
    pageSize = request.args.get('per_page', 10, type=int)
    return pageNo, pageSize

@app.route('/getIncreaseList', methods=['GET'])
def getIncreaseList():
    pageNo, pageSize = get_pagination_args()
    with app.app_context():
        # 使用paginate方法进行分页
        pagination = GasProductionIncrease.query.paginate(page=pageNo, per_page=pageSize, error_out=False)
        items = pagination.items

        # 将结果转换为字典列表
        data = [
            {'id': result.id, 'wellNo': result.well_no, 'platformNo': result.platform_no, 'beginTime': result.begin_time.strftime('%Y-%m-%d')
                , 'endTime': result.end_time.strftime('%Y-%m-%d'), 'days': result.days, 'beforePro': result.before_pro,
             'afterPro': result.after_pro, 'amplify': result.amplify, 'absoluteInc': result.absolute_inc, 'productionInc': result.production_inc
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


@app.route('/getIncreasePlatformList', methods=['GET'])
def getIncreasePlatformList():
    with app.app_context():
        # 假设你想查询所有记录
        results = GasProductionIncrease.query.all()
        # 将结果转换为字典列表
        data = [
            {'platform_no': result.platform_no, 'production_inc': result.production_inc}
            for result in results]
        data = pd.DataFrame(data)
        production_sums = data.groupby('platform_no')['production_inc'].sum().reset_index()
        print(production_sums)

        json_response = {
            "success": True,
            "message": "",
            "code": 200,
            "result": {
                "records": data
            }
        }
        return json_response
    return "Fail"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

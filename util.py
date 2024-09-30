# 导入所有井的日产气数据
import re
from datetime import timedelta, datetime

import numpy as np
import pandas as pd
from tqdm import tqdm


def conv_rcq(df_rcq):
    # df_rcq = pd.read_csv(path, low_memory=False)
    df_rcq['日期'] = pd.to_datetime(df_rcq['日期'])
    df_rcq.loc[df_rcq['日产气'] == '-', '日产气'] = 0
    df_rcq.loc[df_rcq['日产气'] == ' ', '日产气'] = 0
    df_rcq.loc[df_rcq['日产气'] == '无', '日产气'] = 0
    df_rcq['日产气'] = df_rcq['日产气'].astype('float')

    df_rcq = df_rcq[['井号', '日期', '日产气']]
    return df_rcq


def cal_avg_pp(df1, df2, d):
    for i in range(len(df1)):
        dt = df1.loc[i, '施工日期']
        w = df1.loc[i, '井号']
        df_new = df2.loc[df2['井号'] == w].reset_index(drop=True)
        #         print(df_new)
        if len(df_new.loc[df_new['日期'] == dt].index.values) != 0:
            ind1 = df_new.loc[df_new['日期'] == dt].index.values[0]
            if (dt - df_new.loc[0, '日期']).days < d:
                ind0 = 0
            else:
                days_before = dt - timedelta(days=d)
                ind0 = df_new.loc[df_new['日期'] == days_before].index.values[0]
            if (df_new.loc[len(df_new) - 1, '日期'] - dt).days < d:
                ind2 = len(df_new) - 1
            else:
                days_after = dt + timedelta(days=d)
                ind2 = df_new.loc[df_new['日期'] == days_after].index.values[0]
            # r_l1 = df_new.loc[ind0:ind1, '日产气'].tolist()
            # r1 = [i for i in r_l1 if i]
            r_l2 = df_new.loc[ind1 + 1:ind2, '日产气'].tolist()
            r2 = [i for i in r_l2 if i]
            if r2 == []:
                r2.append(0)
            #             print(r1)
            # df1.loc[i, '措施前平均日产'] = np.mean(r1)
            df1.loc[i, '措施前平均日产'] = df1.loc[i, '施工前日产\n（万方/天）']
            df1.loc[i, '措施后平均日产'] = np.mean(r2)
            df1.loc[i, '增幅'] = (df1.loc[i, '措施后平均日产'] - df1.loc[i, '措施前平均日产']) / df1.loc[
                i, '措施前平均日产']
            df1.loc[i, '绝对增加量'] = df1.loc[i, '措施后平均日产'] - df1.loc[i, '措施前平均日产']


def cal_cc_pp(df1, df2, dr, countdays):
    for i in range(len(df1)):
        dt1 = df1.loc[i, '施工日期']
        dt2 = df1.loc[i, '停止施工日期或当前日期']
        period = dt1 + timedelta(days=countdays)
        dl = min(dt2, period)
        if countdays == -1:
            dl = dt2
        w = df1.loc[i, '井号']
        df_new = df2.loc[df2['井号'] == w].reset_index(drop=True)
        #         print(df_new)
        if len(df_new.loc[df_new['日期'] == dt1].index.values) != 0:
            ind1 = df_new.loc[df_new['日期'] == dt1].index.values[0]
            if len(df_new.loc[df_new['日期'] == dl].index.values) != 0:
                ind2 = df_new.loc[df_new['日期'] == dl].index.values[0]
                df1.loc[i, '统计天数'] = (dl - dt1).days
            else:
                ind2 = len(df_new) - 1
                df1.loc[i, '统计天数'] = (df_new.loc[ind2, '日期'] - dt1).days
            r_l = df_new.loc[ind1:ind2, '日产气'].tolist()
            #             print(r_l)
            l = []
            day_value = df1.loc[i, '施工前日产\n（万方/天）']
            for j in r_l:
                # cc = j - (df1.loc[i, '施工前日产\n（万方/天）']) * (1 - dr)
                day_value = day_value * (1 - dr)
                cc = j - day_value
                if cc < 0:
                    l.append(0)
                else:
                    l.append(cc)
            df1.loc[i, '增产气量'] = np.sum(l)


def pao_pai(df_rcq, pppath, dr, d, countdays):
    df_pp = pd.read_csv(pppath, low_memory=False)

    df_pp['井号'] = df_pp['井号'].ffill()
    df_pp['施工日期'] = pd.to_datetime(df_pp['施工日期'])
    df_pp['停止施工日期或当前日期'] = pd.to_datetime(df_pp['停止施工日期或当前日期'])

    # threshold_date = datetime(2023, 12, 31)
    # df_pp = df_pp[df_pp['施工日期'] < threshold_date]
    # df_pp.loc[df_pp['停止施工日期或当前日期'] > threshold_date, '停止施工日期或当前日期'] = threshold_date
    # df_pp.to_csv('input/☆泡排台账（核实）-2023.12.31.csv', encoding='utf-8-sig')

    df_rcq = conv_rcq(df_rcq)

    cal_avg_pp(df_pp, df_rcq, d)
    df_pp.loc[df_pp['增幅'] < 0, '增幅'] = 0
    df_pp.loc[df_pp['措施前平均日产'] < 0.1, '增幅'] = 1
    df_pp.loc[df_pp['绝对增加量'] < 0, '绝对增加量'] = 0
    cal_cc_pp(df_pp, df_rcq, dr, countdays)
    df_pp['平台'] = df_pp['井号'].str.split('-').str[0]
    print('泡排总增产气量为：', np.sum(df_pp['增产气量']))
    # df_pp.to_csv(savepath, encoding='utf-8-sig')
    return df_pp

def add_wei_if_not_first(s):
    if s and s[0] != '威':
        return '威' + s
    else:
        return s


def contains_dash(string):
    if '-' in string:
        return string


def cal_avg_zy(df1, df2, d):
    for i in range(len(df1)):
        dt = df1.loc[i, '增压日期']
        w = df1.loc[i, '井号']
        df_new = df2.loc[df2['井号'] == w].reset_index(drop=True)
        #         print(df_new)
        if len(df_new.loc[df_new['日期'] == dt].index.values) != 0:
            ind1 = df_new.loc[df_new['日期'] == dt].index.values[0]
            if (dt - df_new.loc[0, '日期']).days < d:
                ind0 = 0
            else:
                days_before = dt - timedelta(days=d)
                ind0 = df_new.loc[df_new['日期'] == days_before].index.values[0]
            if (df_new.loc[len(df_new) - 1, '日期'] - dt).days < d:
                ind2 = len(df_new) - 1
            else:
                days_after = dt + timedelta(days=d)
                ind2 = df_new.loc[df_new['日期'] == days_after].index.values[0]
            # r_l1 = df_new.loc[ind0:ind1, '日产气'].tolist()
            # r1 = [i for i in r_l1 if i]
            r_l2 = df_new.loc[ind1 + 1:ind2, '日产气'].tolist()
            r2 = [i for i in r_l2 if i]
            if r2 == []:
                r2.append(0)
            #             print(r1)
            # df1.loc[i, '措施前平均日产'] = np.mean(r1)
            df1.loc[i, '措施前平均日产'] = df1.loc[i, '增压前瞬量（万方/天）']
            df1.loc[i, '措施后平均日产'] = np.mean(r2)
            df1.loc[i, '增幅'] = (df1.loc[i, '措施后平均日产'] - df1.loc[i, '措施前平均日产']) / df1.loc[
                i, '措施前平均日产']
            df1.loc[i, '绝对增加量'] = df1.loc[i, '措施后平均日产'] - df1.loc[i, '措施前平均日产']


def cal_cc_zy(df1, df2, dr, countdays):
    for i in range(len(df1)):
        dt1 = df1.loc[i, '增压日期']
        dt2 = df1.loc[i, '增压停止日期']
        period = dt1 + timedelta(days=countdays)
        dl = min(dt2, period)
        if countdays == -1:
            dl = dt2
        w = df1.loc[i, '井号']
        df_new = df2.loc[df2['井号'] == w].reset_index(drop=True)
        #         print(df_new)
        if len(df_new.loc[df_new['日期'] == dt1].index.values) != 0:
            ind1 = df_new.loc[df_new['日期'] == dt1].index.values[0]
            if len(df_new.loc[df_new['日期'] == dl].index.values) != 0:
                ind2 = df_new.loc[df_new['日期'] == dl].index.values[0]
                df1.loc[i, '统计天数'] = (dl - dt1).days
            else:
                ind2 = len(df_new) - 1
                df1.loc[i, '统计天数'] = (df_new.loc[ind2, '日期'] - dt1).days
            r_l = df_new.loc[ind1:ind2, '日产气'].tolist()
            l = []
            day_value = df1.loc[i, '增压前瞬量（万方/天）']
            for j in r_l:
                # cc = j - df1.loc[i, '增压前瞬量（万方/天）'] * (1 - dr)
                day_value = day_value * (1 - dr)
                cc = j - day_value
                if cc < 0:
                    l.append(0)
                else:
                    l.append(cc)
            df1.loc[i, '增产气量'] = np.sum(l)


def zeng_ya(df_rcq, zypath, dr, d, countdays):
    df_zy = pd.read_csv(zypath, low_memory=False)
    df_zy['井号'] = df_zy['井号'].apply(lambda x: add_wei_if_not_first(x))
    df_zy = df_zy.loc[~(df_zy['增压日期'] == '/'),].reset_index(drop=True)
    df_zy = df_zy.loc[~(df_zy['增压停止日期'] == '/'),].reset_index(drop=True)
    df_zy = df_zy.loc[~(df_zy['增压前瞬量（万方/天）'] == '/'),].reset_index(drop=True)
    df_zy['井号'] = df_zy['井号'].apply(lambda x: contains_dash(x))
    df_zy = df_zy.dropna(subset=['井号']).reset_index(drop=True)
    df_zy['增压日期'] = pd.to_datetime(df_zy['增压日期'])
    df_zy['增压停止日期'] = pd.to_datetime(df_zy['增压停止日期'])
    df_rcq = conv_rcq(df_rcq)

    df_zy.loc[df_zy['增压前瞬量（万方/天）'] == '/', '增压前瞬量（万方/天）'] = 0
    df_zy['增压前瞬量（万方/天）'] = df_zy['增压前瞬量（万方/天）'].astype('float')
    cal_avg_zy(df_zy, df_rcq, d)
    df_zy.loc[df_zy['增幅'] < 0, '增幅'] = 0
    df_zy.loc[df_zy['措施前平均日产'] < 0.1, '增幅'] = 1
    df_zy.loc[df_zy['绝对增加量'] < 0, '绝对增加量'] = 0
    # df_zy = df_zy.dropna(subset=['措施前平均日产'])

    cal_cc_zy(df_zy, df_rcq, dr, countdays)
    df_zy['平台'] = df_zy['井号'].str.split('-').str[0]
    print('增压总增产气量为：', np.sum(df_zy['增产气量']))
    # df_zy.to_csv(savepath, encoding='utf-8-sig')
    return df_zy

def cov_qj(path):
    # 读取Excel文件
    excel_file = pd.ExcelFile(path)
    # 定义一个空的DataFrame来存储合并后的数据
    merged_data = pd.DataFrame()
    # 遍历所有的sheet表
    for sheet_name in excel_file.sheet_names:
        # 读取每个sheet表中的数据
        sheet_data = excel_file.parse(sheet_name)
        # sheet_data = sheet_data[sheet_data['作业井号'] != None ]
        # 将当前sheet表的数据纵向合并到merged_data中
        sheet_data['气举类型'] = re.sub(u'([^\u4e00-\u9fa5])', '', sheet_name)
        merged_data = pd.concat([merged_data, sheet_data], ignore_index=True)
    merged_data = merged_data.dropna(subset=['作业井号'])
    merged_data.loc[merged_data['气举类型'] == '增压机气举已搬迁至', '气举类型'] = '增压机气举'
    merged_data['作业井号'] = merged_data['作业井号'].astype('string')
    return merged_data


def add_wei_if_not_first(s):
    if s and s[0] != '威':
        return '威' + s
    else:
        return s


# def cal_avg_qj(df1, df2, d):
#     for i in range(len(df1)):
#         dt = df1.loc[i, '施工日期']
#         w = df1.loc[i, '作业井号']
#         df_new = df2.loc[df2['井号'] == w].reset_index(drop=True)
#         #         print(df_new)
#         if len(df_new.loc[df_new['日期'] == dt].index.values) != 0:
#             ind1 = df_new.loc[df_new['日期'] == dt].index.values[0]
#             if (dt - df_new.loc[0, '日期']).days < d:
#                 ind0 = 0
#             else:
#                 days_before = dt - timedelta(days=d)
#                 ind0 = df_new.loc[df_new['日期'] == days_before].index.values[0]
#             if (df_new.loc[len(df_new) - 1, '日期'] - dt).days < d:
#                 ind2 = len(df_new) - 1
#             else:
#                 days_after = dt + timedelta(days=d)
#                 ind2 = df_new.loc[df_new['日期'] == days_after].index.values[0]
#             # r_l1 = df_new.loc[ind0:ind1, '日产气'].tolist()
#             # r1 = [i for i in r_l1 if i]
#             r_l2 = df_new.loc[ind1 + 1:ind2, '日产气'].tolist()
#             r2 = [i for i in r_l2 if i]
#             # if r1 == []:
#             #     r1.append(0)
#             if r2 == []:
#                 r2.append(0)
#             #             print(r1)
#             # df1.loc[i, '措施前平均日产'] = np.mean(r1)
#             df1.loc[i, '措施前平均日产'] = df1.loc[i, '增压前瞬量（万方/天）']
#             df1.loc[i, '措施后平均日产'] = np.mean(r2)
#             df1.loc[i, '增幅'] = (df1.loc[i, '措施后平均日产'] - df1.loc[i, '措施前平均日产']) / df1.loc[
#                 i, '措施前平均日产']
#             df1.loc[i, '绝对增加量'] = df1.loc[i, '措施后平均日产'] - df1.loc[i, '措施前平均日产']
#
#
#         # else:
#         #     print(w)


def convert_to_number_or_zero(s):
    try:
        # 尝试将字符串转换为整数
        return float(s)
    except ValueError:
        # 如果转换失败（即字符串不是数字），则返回0
        return 0


def cal_cc_lxqj(df1, df2, l_lxqj, countdays, dr):
    # 连续：电驱、增压机、固定制氮
    # 电驱、增压：日产-注气量
    # 固定制氮直接用日产
    df = pd.DataFrame()
    well = []
    #     df['井号']=l_lxqj
    l = []
    sd = []
    ed = []
    d = []
    srcq = []
    ercq = []
    sf = []
    sjl = []
    type = []
    for i in l_lxqj:
        df_new1 = df1.loc[df1['井号'] == i].reset_index(drop=True)
        df_new2 = df2.loc[df2['井号'] == i].reset_index(drop=True)
        #         print(df_new)
        df_m = pd.merge(df_new1, df_new2, on=['井号', '日期'], how='left')

        dt1 = df_m.loc[0, '日期']
        if countdays == -1:
            pass
        else:
            period = dt1 + timedelta(days=countdays)
            df_m = df_m.loc[df_m['日期'] <= period,].reset_index(drop=True)
        if len(df_m) != 0:
            for j in range(len(df_m)):
                df_m.loc[j, '增产气量'] = df_m.loc[j, '日产气']

                # if df_m.loc[j, '气举类型'] == '固定制氮':
                #     df_m.loc[j, '增产气量'] = 0
                # else:
                #     df_m.loc[j, '增产气量'] = df_m.loc[j, '日产气']

                # if df_m.loc[j, '气举类型'] == '固定制氮' or df_m.loc[j, '气举类型'] == '电驱气举' or df_m.loc[j, '气举类型'] == '增压机气举': #
                # print(df_m.loc[j, '气举类型'])
                # df_m.loc[j, '增产气量'] = df_m.loc[j, '日产气']

                # else:
                #     df_m.loc[j, '增产气量'] = max(df_m.loc[j, '日产气'] - df_m.loc[j, '注气量（万方）'], 0)
            # print("*"*20)
            well.append(df_m.loc[0, '井号'])
            try:
                s = np.sum(df_m['增产气量'])
                arr = df_m['增产气量'].tolist()
                for _ in range(len(arr)):
                    s = s * (1 - dr)
                l.append(s)
            except:
                l.append(0)
            sd.append(df_m.loc[0, '日期'])
            ed.append(df_m.loc[len(df_m) - 1, '日期'])
            d.append(len(df_m))
            srcq.append(df_m.loc[0, '措施前平均日产'])
            ercq.append(df_m.loc[0, '措施后平均日产'])
            sf.append(df_m.loc[0, '增幅'])
            sjl.append(df_m.loc[0, '绝对增加量'])
    df['井号'] = well
    df['最初施工日期'] = sd
    df['最后施工日期'] = ed
    df['施工天数'] = d
    df['措施前平均日产'] = srcq
    df['措施后平均日产'] = ercq
    df['增幅'] = sf
    df['绝对增加量'] = sjl
    df['增产气量'] = l
    return df


def lx_qi_ju(df_rcq, qjpath, dr, d, countdays):
    df_qj = cov_qj(qjpath)
    df_qj['作业井号'] = df_qj['作业井号'].str.replace('204H62（209）', '209')
    df_qj['作业井号'] = df_qj['作业井号'].str.replace('威208', '208')
    df_qj['作业井号'] = df_qj['作业井号'].str.replace('209.0', '209')

    # df_qj.to_csv('df_qj.csv', encoding='utf-8-sig')
    df_qj['作业井号'] = df_qj['作业井号'].apply(lambda x: add_wei_if_not_first(x))  # 增加井号
    df_qj = df_qj.loc[~(df_qj['施工日期'] == '/'),].reset_index(drop=True)
    df_qj['施工日期'] = pd.to_datetime(df_qj['施工日期'])
    df_rcq = conv_rcq(df_rcq)

    df_qj = df_qj.loc[
        (df_qj['气举类型'] == '电驱气举') | (df_qj['气举类型'] == '增压机气举')].reset_index(drop=True)
    # df_qj.to_csv('df_qj.csv', encoding='utf-8-sig')
    # print(df_qj)

    cal_avg_qj(df_qj, df_rcq, d)
    df_qj.loc[df_qj['增幅'] < 0, '增幅'] = 0
    df_qj.loc[df_qj['措施前平均日产'] < 0.1, '增幅'] = 1
    df_qj.loc[df_qj['绝对增加量'] < 0, '绝对增加量'] = 0

    # df_qj.to_csv('df_qj.csv', encoding='utf-8-sig')

    # 连续：电驱、增压机、固定制氮
    # 间断：制氮车、天然气气举

    # 电驱、增压：日产-注气量
    # 固定制氮直接用日产

    # df_jdqj = df_qj.loc[
    #     (df_qj['气举类型'] == '电驱气举') | (df_qj['气举类型'] == '增压机气举') | (
    #                 df_qj['气举类型'] == '固定制氮')].reset_index(drop=True)
    # df_lxqj = pd.concat([df_qj, df_jdqj, df_jdqj]).drop_duplicates(keep=False).reset_index(drop=True)

    # df_lxqj = df_qj.loc[
    #     (df_qj['气举类型'] == '电驱气举') | (df_qj['气举类型'] == '增压机气举')].reset_index(drop=True)
    df_lxqj = df_qj.reset_index(drop=True)

    # df_lxqj = pd.concat([df_qj, df_lxqj]).reset_index(drop=True)
    df_lxqj['作业井号'] = df_lxqj['作业井号'].str.replace('.0', '')

    mask = df_lxqj['作业井号'].str.contains(r'[、/.一(]', na=False)
    df_lxqj = df_lxqj[~mask]

    df_lxqj.rename(columns={'作业井号': '井号', '施工日期': '日期'}, inplace=True)

    l_lxqj = df_lxqj['井号'].value_counts().index.tolist()
    df_lxqj['日期'] = pd.to_datetime(df_lxqj['日期'])
    df_lxqj['施工前日产（万方）'] = df_lxqj['施工前日产\n（万方）'].apply(lambda x: convert_to_number_or_zero(x))
    df_lxqj['注气量（万方）'] = df_lxqj['注气量\n（万方）'].apply(lambda x: convert_to_number_or_zero(x))
    df_lxqj['施工前日产（万方）'] = df_lxqj['施工前日产（万方）'].fillna(0)
    # df_lxqj.to_csv('df_lxqj.csv', encoding='utf-8-sig')

    df = cal_cc_lxqj(df_lxqj, df_rcq, l_lxqj, countdays, dr)
    df = df.dropna(subset=['措施前平均日产'])
    df = df.dropna(subset=['措施后平均日产'])
    df = df.dropna(subset=['增幅'])
    df = df.dropna(subset=['绝对增加量'])
    df['平台'] = df['井号'].str.split('-').str[0]
    print('连续气举总增产气量为：', np.sum(df['增产气量']))
    # df.to_csv(savepath, encoding='utf-8-sig')
    return df


def cov_qj(path):
    # 读取Excel文件
    excel_file = pd.ExcelFile(path)
    # 定义一个空的DataFrame来存储合并后的数据
    merged_data = pd.DataFrame()
    # 遍历所有的sheet表
    for sheet_name in excel_file.sheet_names:
        # 读取每个sheet表中的数据
        sheet_data = excel_file.parse(sheet_name)
        # sheet_data = sheet_data[sheet_data['作业井号'] != None ]
        # 将当前sheet表的数据纵向合并到merged_data中
        sheet_data['气举类型'] = re.sub(u'([^\u4e00-\u9fa5])', '', sheet_name)
        merged_data = pd.concat([merged_data, sheet_data], ignore_index=True)

    merged_data = merged_data.dropna(subset=['作业井号'])
    merged_data.loc[merged_data['气举类型'] == '增压机气举已搬迁至', '气举类型'] = '增压机气举'
    merged_data['作业井号'] = merged_data['作业井号'].astype('string')
    return merged_data


def add_wei_if_not_first(s):
    if s and s[0] != '威':
        return '威' + s
    else:
        return s


def cal_avg_qj(df1, df2, d):
    for i in range(len(df1)):
        dt = df1.loc[i, '施工日期']
        w = df1.loc[i, '作业井号']
        df_new = df2.loc[df2['井号'] == w].reset_index(drop=True)
        #         print(df_new)
        if len(df_new.loc[df_new['日期'] == dt].index.values) != 0:
            ind1 = df_new.loc[df_new['日期'] == dt].index.values[0]
            if (dt - df_new.loc[0, '日期']).days < d:
                ind0 = 0
            else:
                days_before = dt - timedelta(days=d)
                ind0 = df_new.loc[df_new['日期'] == days_before].index.values[0]
            if (df_new.loc[len(df_new) - 1, '日期'] - dt).days < d:
                ind2 = len(df_new) - 1
            else:
                days_after = dt + timedelta(days=d)
                ind2 = df_new.loc[df_new['日期'] == days_after].index.values[0]
            r_l1 = df_new.loc[ind0:ind1, '日产气'].tolist()
            r1 = [i for i in r_l1 if i]
            r_l2 = df_new.loc[ind1 + 1:ind2, '日产气'].tolist()
            r2 = [i for i in r_l2 if i]
            if r1 == []:
                r1.append(0)
            if r2 == []:
                r2.append(0)
            #             print(r1)
            df1.loc[i, '措施前平均日产'] = np.mean(r1)
            df1.loc[i, '措施后平均日产'] = np.mean(r2)
            df1.loc[i, '增幅'] = (df1.loc[i, '措施后平均日产'] - df1.loc[i, '措施前平均日产']) / df1.loc[
                i, '措施前平均日产']
            df1.loc[i, '绝对增加量'] = df1.loc[i, '措施后平均日产'] - df1.loc[i, '措施前平均日产']


        # else:
        #     print(w)


def deal_temp(result, temp, days, i):
    result = result._append(temp.iloc[i], ignore_index=True)
    row_data = temp.iloc[i].to_frame().T
    row_data = pd.DataFrame(float('NaN'), index=row_data.index, columns=row_data.columns)
    row_data['施工日期'] = temp.iloc[i]['施工日期'] + pd.DateOffset(days=1)
    row_data['作业井号'] = temp.iloc[i]['作业井号']
    row_data['气举类型'] = temp.iloc[i]['气举类型']

    row_data_temp = row_data.copy()
    for t in range(2, days + 1):
        row_data['施工日期'] = temp.iloc[i]['施工日期'] + pd.DateOffset(days=t)
        row_data_temp = pd.concat([row_data_temp, row_data])
    result = pd.concat([result, row_data_temp])
    return result


def add_date(df):
    days = 10
    # 使用str.replace()方法删除列中的空格  文件中有错误，有井的编号加入了无用空格
    df['作业井号'] = df['作业井号'].str.replace(' ', '')

    # 步骤1：获取作业井号列表
    well_list = list(df['作业井号'].unique())

    # 步骤2：筛选并处理数据
    result = pd.DataFrame(columns=df.columns)
    for well in tqdm(well_list):
        temp = df[df['作业井号'] == well].copy()
        temp['施工日期'] = pd.to_datetime(temp['施工日期'])
        temp = temp.sort_values(by=['气举类型', '施工日期'])

        for i in range(0, len(temp)):
            days_diff = (temp.iloc[i]['施工日期'] - temp.iloc[i - 1]['施工日期']).days
            if temp.iloc[i]['气举类型'] != temp.iloc[i - 1]['气举类型']:
                result = deal_temp(result, temp, days, i)

            elif days_diff > days:
                result = deal_temp(result, temp, days, i)

            else:
                result = deal_temp(result, temp, days, i)

                # 删除重复的行
    result.drop_duplicates(subset=['作业井号', '施工日期', '气举类型'], inplace=True)
    result = result.sort_values(by=['作业井号', '施工日期']).reset_index(drop=True)
    return result


def add_date_new(df):
    days = 10
    # 使用str.replace()方法删除列中的空格  文件中有错误，有井的编号加入了无用空格
    df['作业井号'] = df['作业井号'].str.replace(' ', '')

    # 步骤1：获取作业井号列表
    well_list = list(df['作业井号'].unique())

    # 步骤2：筛选并处理数据
    result = pd.DataFrame(columns=df.columns)
    for well in tqdm(well_list):
        temp = df[df['作业井号'] == well].copy()
        temp['施工日期'] = pd.to_datetime(temp['施工日期'])
        temp = temp.sort_values(by=['施工日期'])
        result = deal_temp(result, temp, days, len(temp) - 1)
        result = pd.concat([temp, result])
        # 删除重复的行
    result.drop_duplicates(subset=['作业井号', '施工日期'], inplace=True)
    result = result.sort_values(by=['作业井号', '施工日期']).reset_index(drop=True)
    return result


def convert_to_number_or_zero(s):
    try:
        # 尝试将字符串转换为整数
        return float(s)
    except ValueError:
        # 如果转换失败（即字符串不是数字），则返回0
        return 0


def cal_cc_jdqj_new(df1, df2, countdays, dr):
    df = pd.DataFrame()
    well = []
    sd = []
    l = []
    d = []
    srcq = []
    ercq = []
    sf = []
    sjl = []
    jhs = df1['井号'].value_counts().index.tolist()
    for jh in jhs:
        df_new1 = df1.loc[df1['井号'] == jh].reset_index(drop=True)
        df_new2 = df2.loc[df2['井号'] == jh].reset_index(drop=True)
        #         print(df_new)
        df_m = pd.merge(df_new1, df_new2, on=['井号', '日期'], how='left')
        dt1 = df_m.loc[0, '日期']

        if countdays == -1:
            pass
        else:
            period = dt1 + timedelta(days=countdays)
            df_m = df_m.loc[df_m['日期'] <= period,].reset_index(drop=True)
        if len(df_m) != 0:
            for j in range(len(df_m)):
                # if df_m.loc[j, '气举类型'] == '制氮车' or df_m.loc[j, '气举类型'] == '天然气气举':
                df_m.loc[j, '增产气量'] = df_m.loc[j, '日产气']
                # if df_m.loc[j, '日期'].year < 2022:
                #     df_m.loc[j, '增产气量'] = max(df_m.loc[j, '日产气'] - df_m.loc[j, '注气量（万方）'], 0)
                # else:
                #     df_m.loc[j, '增产气量'] = df_m.loc[j, '日产气']
                # df_m.loc[j, '增产气量'] = max(df_m.loc[j, '日产气'] - df_m.loc[j, '注气量（万方）'], 0)
            try:
                s = np.sum(df_m['增产气量'])
                arr = df_m['增产气量'].tolist()
                for _ in range(len(arr)):
                    s = s * (1 - dr)
                l.append(s)
            except:
                l.append(0)
            well.append(df_m.loc[0, '井号'])
            sd.append(df_m.loc[0, '日期'])
            # l.append(np.sum(df_m['增产气量']))
            # d.append(len(df_m))
            d.append(df_m.loc[len(df_m) - 1, '日期'])
            srcq.append(df_m.loc[0, '措施前平均日产'])
            ercq.append(df_m.loc[0, '措施后平均日产'])
            sf.append(df_m.loc[0, '增幅'])
            sjl.append(df_m.loc[0, '绝对增加量'])
    df['井号'] = well
    df['首次施工日期'] = sd
    df['最后施工日期'] = d
    df['措施前平均日产'] = srcq
    df['措施后平均日产'] = ercq
    df['增幅'] = sf
    df['绝对增加量'] = sjl
    df['增产气量'] = l

    return df


def cal_cc_jdqj(df1, df2, l_jdqjind, countdays):
    df = pd.DataFrame()
    well = []
    sd = []
    l = []
    d = []
    srcq = []
    ercq = []
    sf = []
    sjl = []
    for i in range(len(l_jdqjind) - 1):
        df_new1 = df1.loc[l_jdqjind[i]:l_jdqjind[i + 1] - 1, ]
        jh = df1.loc[l_jdqjind[i], '井号']
        df_new2 = df2.loc[df2['井号'] == jh].reset_index(drop=True)
        #         print(df_new)
        df_m = pd.merge(df_new1, df_new2, on=['井号', '日期'], how='left')
        dt1 = df_m.loc[0, '日期']

        if countdays == -1:
            pass
        else:
            period = dt1 + timedelta(days=countdays)
            df_m = df_m.loc[df_m['日期'] <= period,].reset_index(drop=True)
        if len(df_m) != 0:
            for j in range(len(df_m)):
                if df_m.loc[j, '气举类型'] == '制氮车' or df_m.loc[j, '气举类型'] == '天然气气举':
                    df_m.loc[j, '增产气量'] = df_m.loc[j, '日产气']
                    print(df_m.loc[j, '日产气'])
                # if df_m.loc[j, '日期'].year < 2022:
                #     df_m.loc[j, '增产气量'] = max(df_m.loc[j, '日产气'] - df_m.loc[j, '注气量（万方）'], 0)
                # else:
                #     df_m.loc[j, '增产气量'] = df_m.loc[j, '日产气']
                # df_m.loc[j, '增产气量'] = max(df_m.loc[j, '日产气'] - df_m.loc[j, '注气量（万方）'], 0)
            well.append(df_m.loc[0, '井号'])
            sd.append(df_m.loc[0, '日期'])
            l.append(np.sum(df_m['增产气量']))
            d.append(len(df_m))
            # srcq.append(df_m.loc[0, '措施前平均日产'])
            # ercq.append(df_m.loc[0, '措施后平均日产'])
            # sf.append(df_m.loc[0, '增幅'])
            # sjl.append(df_m.loc[0, '绝对增加量'])
    df['井号'] = well
    df['施工日期'] = sd
    df['措施天数'] = d
    # df['措施前平均日产'] = srcq
    # df['措施后平均日产'] = ercq
    # df['增幅'] = sf
    # df['绝对增加量'] = sjl
    df['增产气量'] = l
    return df


def jd_qi_ju(df_rcq, qjpath, dr, d, countdays):
    # 间断：制氮车、天然气气举
    df_qj = cov_qj(qjpath)
    df_qj['作业井号'] = df_qj['作业井号'].apply(lambda x: add_wei_if_not_first(x))  # 增加井号
    df_qj = df_qj.loc[~(df_qj['施工日期'] == '/'),].reset_index(drop=True)
    df_qj['施工日期'] = pd.to_datetime(df_qj['施工日期'])

    df_rcq = conv_rcq(df_rcq)

    cal_avg_qj(df_qj, df_rcq, d)
    df_qj.loc[df_qj['增幅'] < 0, '增幅'] = 0
    df_qj.loc[df_qj['措施前平均日产'] < 0.1, '增幅'] = 1
    df_qj.loc[df_qj['绝对增加量'] < 0, '绝对增加量'] = 0

    df_jdqj = df_qj.loc[
        (df_qj['气举类型'] == '制氮车') | (df_qj['气举类型'] == '天然气气举')].reset_index(drop=True)

    df_jdqj = df_jdqj.sort_values(by=['作业井号', '施工日期']).reset_index(drop=True)

    # df_jdqj = add_date(df_jdqj)
    df_jdqj = add_date_new(df_jdqj)
    df_jdqj.rename(columns={'作业井号': '井号', '施工日期': '日期'}, inplace=True)
    # df_jdqj.to_csv('df_jdqj.csv', encoding='utf-8-sig')

    df_jdqj['日期'] = pd.to_datetime(df_jdqj['日期'])
    df_jdqj['施工前日产（万方）'] = df_jdqj['施工前日产\n（万方）'].apply(lambda x: convert_to_number_or_zero(x))
    df_jdqj['注气量（万方）'] = df_jdqj['注气量\n（万方）'].apply(lambda x: convert_to_number_or_zero(x))
    df_jdqj['施工前日产（万方）'] = df_jdqj['施工前日产（万方）'].fillna(0)
    df_jdqj['井号'] = df_jdqj['井号'].str.replace('.0', '')
    mask = df_jdqj['井号'].str.contains(r'[、/.一(]', na=False)
    df_jdqj = df_jdqj[~mask]
    # l_jdqjind = df_jdqj.loc[~(df_jdqj['措施前平均日产'].isnull()),].index

    df = cal_cc_jdqj_new(df_jdqj, df_rcq, countdays, dr)
    df['平台'] = df['井号'].str.split('-').str[0]
    print('间断气举总增产气量为：', np.sum(df['增产气量']))
    # df.to_csv(savepath, encoding='utf-8-sig')
    return df



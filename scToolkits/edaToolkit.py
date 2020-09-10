#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/5/6 09:39
# @Author  : cai
# @contact : yuwei.chen@yunzhenxin.com
# @File    : edaToolkit.py
# @Note    :

import sys
sys.path.append("..")

import pandas as pd
import numpy as np
import operator
from prettytable import PrettyTable

from Configs import log_config
from Utils.LogUtil import LogUtil



## 特征分析
class EdaUtil(object):
    def __init__(self, file_name,  df, target_name="y", del_col=[], dist_col=[], serial_col=[], time_col="create_date",
                 time_level="m", max_col_null_pct=0.8, max_row_null_pct=0.8,
                 max_null_pct_delta=0.3, max_various_values=100):
        '''
        :param file_name: 确定日志文件名，以及需要保存的
        :param df: str;label列的名字
        :param target_name: list; 不是离散变量且不是连续变量，e.g. y
        :param del_col: list; 不是离散变量且不是连续变量，e.g. y/i/time
        :param dist_col: list; 连续变量
        :param serial_col: list; 连续变量
        :param time_col: str;时间列名称
        :param time_level: charater;需要观察的时间维度（y-年度；m-月度；d-天）
        :param max_col_null_pct: float; 特征覆盖率阈值
        :param max_row_null_pct: float; 样本浓度阈值
        :param max_null_pct_delta: float; abs(特征在y1样本集中的覆盖率-特征在y0样本集中的覆盖率)
        :param max_various_values: int; 离散遍变量的unique value值个数
        '''

        if file_name:
            self.log = LogUtil(file_name)
        else:
            self.log = LogUtil("EDA_" + datetime.datetime.today().strftime("%Y%m%d"))

        self.del_col = del_col[:]
        if not dist_col:
            self.get_object_cols()
        else:
            self.dist_col = dist_col[:]

        if not serial_col:
            self.serial_col = [i for i in df.columns if i not in del_col and i not in dist_col if i != target_name]
        else:
            self.serial_col = serial_col[:]

        if len(self.serial_col)>0:
            tmp_serical = pd.DataFrame(df[self.serial_col], dtype=float)
            self.df = pd.concat([df[list(set(self.del_col + self.dist_col + [target_name]))], tmp_serical], axis=1)
        else:
            self.df = df[list(set(self.del_col + self.dist_col + [target_name]))]
        self.df.rename(columns={target_name: "y"}, inplace=True)

        self.max_col_null_pct = max_col_null_pct
        self.max_row_null_pct = max_row_null_pct
        self.max_null_pct_delta = max_null_pct_delta
        if max_various_values <= self.df.shape[0] / 10:
            self.max_various_values = max_various_values
        else:
            self.max_various_values = int(self.df.shape[0] / 10)

        if time_col and time_level:
            from Utils.time_tools import get_std_time
            try:
                self.df["format_date"] = self.df[time_col].map(lambda x: get_std_time(x, level=time_level))
                self.time_col = time_col
                self.time_level = time_level
                self.del_col.append('format_date')
            except:
                self.log.info("时间格式错误，无法进行时间维度分析")
                self.time_col = None
                self.time_level = None
        else:
            self.time_col = None
            self.time_level = None
    # 定义打印对象时打印的字符串
    def __str__(self, print_all=True):
        if print_all:
            self.log.info(self.__dict__)
        else:
            self.log.info(' '.join(('%s' % item for item in self.__dict__.keys())))


    def go_init_explore(self):
        self.check_y()
        if self.time_col and self.time_level:
            self.observe_y_by_time()
        self.observe_all_nulls()
        self.observe_null_by_y()
        self.get_single_various_values()
        self.get_null_cols_rows()
        self.output_objects()

    ## 检查Y值
    def check_y(self):
        self.df["y"] = self.df.y.astype(float).astype(int)
        data = self.df.copy()
        if len(data.y.unique()) > 2 :
            self.log.info("哇哦！活见鬼的多分类！\n本函数不适合你，拜拜！")
            sys.exit()
        if any(data.y.isnull()):
            self.log.info("有{}个样本没有匹配到Y值".format(len(data[data.y.isnull()])))
            self.log.info("全局label1_ratio={}".format(round(data.y.value_counts()[1]/len(data),4)))
            self.log.info("没有匹配到Y值的样本label1_ratio={}".format(round(data[data.y.isnull()].y.value_counts()[1]/len(data[data.y.isnull()]),4)))
            self.df = self.df[~self.df.y.isnull()]
        else:
            self.log.info("所有样本均有Y值")
            self.log.info("但请不要得瑟，数据已经有了，模型，嘿嘿嘿～")
        self.log.info("*"*50)
        del data

    ## 检查各个时间单位上的Y分布
    def observe_y_by_time(self):
        data = self.df.copy()
        data = data.y.groupby([data["format_date"]]).agg(["count", "sum"]).sort_values(by="format_date")
        data.rename(columns={"count": "total", "sum":"bad_cnt"}, inplace=True)
        data["bad_pct"] = data.apply(lambda x: round(x.bad_cnt/ x.total, 4), axis=1)
        self.log.info("样本共有{}个时间单位，具体说明如下".format(len(list(data.index))))
        from scToolkits.drawToolkit import get_badpct_by_time
        get_badpct_by_time(data)
        self.log.info("*" * 50)
        del data

    ## 查看样本总体缺失值情况
    def observe_all_nulls(self):
        data = self.df.copy()
        data['null_cnt'] = data.isnull().sum(axis=1)
        data['null_pct'] = data.null_cnt / data.shape[0]
        data = data.sort_values('null_pct')
        data = data[["null_pct"]]
        data.loc[:, 'num'] = [i + 1 for i in range(len(data))]
        self.log.info("样本整体缺失率情况如下:")
        from scToolkits.drawToolkit import get_all_null
        get_all_null(data)
        self.log.info("*" * 50)
        del data

    ## 分y查看样本缺失情况
    def observe_null_by_y(self):
        data = self.df.copy()
        null_pct_by_by_label = data.drop('y', axis=1).groupby(data['y']).apply(
            lambda x: (x.isna().sum()) / (x.shape[0])).T.drop_duplicates(0.0).sort_values(0.0)
        null_pct_by_by_label.columns = ['label_0', 'label_1']

        self.log.info("不同label下样本整体缺失情况如下:")
        from scToolkits.drawToolkit import get_null_pct_by_label
        get_null_pct_by_label(null_pct_by_by_label)

        null_pct_by_by_label['delta'] = null_pct_by_by_label.apply(lambda x: np.abs(x['label_0'] - x['label_1']), axis=1)
        null_pct_by_by_label = null_pct_by_by_label[null_pct_by_by_label.delta > self.max_null_pct_delta]
        null_pct_by_by_label.sort_values(by=['delta'], ascending=False)
        self.log.info("有{}个特征在好坏样本上缺失值(高于{})差异明显".format(len(null_pct_by_by_label), self.max_null_pct_delta))
        if len(null_pct_by_by_label)>0:
            self.log.info("不同label下的缺失值情况如下\n:{}".format(self.format_dataframe(null_pct_by_by_label)))
        self.log.info("*"*50)
        del data
        del null_pct_by_by_label

    ## 获取所有无法转化的连续变量
    def get_object_cols(self):
        data = self.df.copy()
        init_object_cols = list(data.columns[data.dtypes == 'object'])
        self.dist_col = []
        unreal_objects_col = []
        for col in init_object_cols:
            if col in self.del_col:
                continue
            else:
                try:
                    self.df[col].astype(np.float)
                    unreal_objects_col.append(col)
                except:
                    self.dist_col.append(col)
        if len(unreal_objects_col):
            self.log.info("这些离散变量可以处理为连续值:{}".format(unreal_objects_col))
        self.log.info("处理以后剩下的离散变量共{}个，分别是：{}".format(len(self.dist_col), self.dist_col))
        self.log.info("*" * 50)
        del data

    ## 获取唯一值、unique value过多值
    def get_single_various_values(self):
        data = self.df.copy()
        self.single_cols = []
        self.various_cols = []
        self.q5equalq95_col = []
        for col in data.columns:
            tmp_ = data[~data[col].isnull()][col]
            if col in self.del_col:
                continue
            elif col in self.dist_col:
                if len(tmp_.head(100).unique()) > self.max_various_values:
                    self.various_cols.append(col)
                elif len(tmp_.unique()) > self.max_various_values:
                    self.various_cols.append(col)
                elif len(tmp_.unique()) == 1:
                    self.single_cols.append(col)
            elif col in self.serial_col:
                quantile00, quantile99 = tmp_.quantile([.5, .95])
                if quantile00 == quantile99:
                    self.q5equalq95_col.append(col)
        if self.single_cols:
            self.log.info("有{}个特征为唯一值,分别如下{}:".format(len(self.single_cols), self.single_cols))
            self.log.info("*" * 50)
        if self.various_cols:
            self.log.info("有{}个离散特征值过多,分别如下{}".format(len(self.various_cols), self.various_cols))
            self.log.info("*" * 50)
        if self.q5equalq95_col:
            self.log.info("有{}个连续特征无明显差异,分别如下{}".format(len(self.q5equalq95_col), self.q5equalq95_col))
            self.log.info("*" * 50)
        del data

    ## 特征缺失率情况
    def get_null_cols_rows(self):
        data = self.df.copy()
        self.to_drop_cols = []
        self.to_drop_rows = []

        # 对行的探索
        rows_null_percent = data.isnull().sum(axis=1) / data.shape[1]
        to_drop_rows = rows_null_percent[rows_null_percent > self.max_row_null_pct]
        if len(to_drop_rows) > 0:
            to_drop_rows = rows_null_percent[rows_null_percent > self.max_row_null_pct]
            to_drop_rows = to_drop_rows.to_dict()
            self.to_drop_rows = sorted(to_drop_rows.items(), key=operator.itemgetter(1), reverse=True)
            self.log.info("缺失值比例>{}的有{}行".format(self.max_row_null_pct, len(to_drop_rows)))
            data = data[data.index.isin(list(to_drop_rows.keys()))]
            self.log.info("全局label1的占比为:{}".format(round(self.df.y.value_counts()[1]/len(self.df),4)))
            self.log.info("特征饱和度低于{}的样本label1的占比为:{}".format(self.max_row_null_pct, round(data.y.value_counts()[1] / len(data), 4)))
        else:
            self.log.info("哇哦～没有样本有缺失数据")
        self.log.info("*"*50)

        # 对列的探索
        data = self.df.copy()
        cols_null_percent = data.isnull().sum() / data.shape[0]
        to_drop_cols = cols_null_percent[cols_null_percent > self.max_col_null_pct]
        if len(to_drop_cols) > 0:
            to_drop_cols = dict(to_drop_cols)
            self.to_drop_cols = sorted(to_drop_cols.items(), key=operator.itemgetter(1), reverse=True)

            self.log.info("缺失值比例>{}的有{}个变量".format(self.max_col_null_pct, len(to_drop_cols)))
            result = {
                'dtype': data.dtypes,
                'data_cnt': data.shape[0],
                'null_cnt': data.isnull().sum(),
                'null_ratio': data.isnull().sum() / data.shape[0]
            }
            result = pd.DataFrame(result, columns=['dtype', 'data_cnt', 'null_cnt', 'null_ratio'])
            result = result[result.null_ratio > self.max_col_null_pct]
            result = result.sort_values(by=['null_ratio'], ascending=False)
            self.log.info("每列缺失值情况如下:\n{}".format((self.format_dataframe(result))))
        else:
            self.log.info("哇哦～特征全覆盖")
        self.log.info("*"*50)
        del data
        del cols_null_percent
        del rows_null_percent


    ## output show
    def output_objects(self):
        self.log.info("本次EDA，可供参考的数据信息有以下")
        self.log.info("无法转化为连续(float)的特征: {}.dist_col".format(self.__class__.__name__))
        self.log.info("有唯一值的特征: {}.single_cols".format(self.__class__.__name__))
        self.log.info("value值过多的特征: {}.various_cols".format(self.__class__.__name__))
        self.log.info("分布无明显变化的特征: {}.q5equalq95_col".format(self.__class__.__name__))
        self.log.info("缺失率过高的特征: {}.to_drop_cols".format(self.__class__.__name__))
        self.log.info("缺失率过高的行: {}.to_drop_rows".format(self.__class__.__name__))

    @ staticmethod
    def format_dataframe(df):
        data = PrettyTable([''] + list(df.columns))
        for row in df.itertuples():
            data.add_row(row)
        return data
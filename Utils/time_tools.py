#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/4/14 16:23
# @author  : gedn
# @contact : danni.ge@yunzhenxin.com
# @File    : time_tools.py
# @Note    : traditional edition for time tools

import os
import datetime
import re


def check_source(path, day_desc):
    '''
    功能：检查指定表的指定日期中数据源是否正常
    参数：
        1)path:指定表的路径
        2)day_desc:指定日期
    返回：1表示数据异常，0表示数据正常
    '''
    print('check output ' + path + '/' + day_desc)
    if os.system("""hadoop fs -test -e {path}/{day}""".format(path=path, day=day_desc)) != 0:

        print('partition day not exist.')
        return 1
    else:
        for line in os.popen("""hadoop fs -du {path}/ | grep {day} """.format(path=path, day=day_desc)).readlines():
            if int(line.strip().split(' ')[0]) < 50000:

                print('empty file.')
                return 1
            else:
                print('ok.')
                return 0


def get_back_day(day, back_num):
    '''
    功能：获取距离指定日期 往前 指定天数的日期
    参数：
        1)run_day:指定日期
        2)back_num:指定天数
    返回：日期，形式为YYYYMMDD形式
    '''
    day = datetime.datetime.strptime(str(day), "%Y%m%d")
    res_day = datetime.date(day.year, day.month, day.day) + datetime.timedelta(days=-back_num)
    res_day = res_day.strftime("%Y%m%d")

    return res_day


def get_weekday_from_certain_day(day, monday):
    '''
    功能：获取指定日期最近的指定周几
    参数：
        1)run_day:指定日期
        2)back_num:指定周几
    返回：日期，形式为YYYYMMDD形式
    示例：monday = 0 获取周一，monday = 1 获取周二 等等

    '''
    date_time = datetime.datetime.strptime(str(day), "%Y%m%d")
    diff = monday - date_time.weekday()
    marketDate = date_time + datetime.timedelta(days=diff)
    res_day = marketDate.strftime("%Y%m%d")
    return res_day


def get_month_day_ls(month):
    '''
    功能：获取指定月份的所有日期
    参数：
        1)path:指定表的存储路径
        2)day:指定日期
    返回：日期列表，元素形式为YYYYMMDD
    '''
    from dateutil.relativedelta import relativedelta
    month = datetime.datetime.strptime(str(month), "%Y%m")

    future_month = datetime.date(month.year, month.month, month.day) + relativedelta(months=1)

    last_day = datetime.date(future_month.year, future_month.month, future_month.day) + datetime.timedelta(days=-1)
    last_day = last_day.strftime("%Y%m%d")

    day_list = [last_day]
    while int(last_day[4:6]) == int(month.month):

        last_day = datetime.datetime.strptime(str(last_day), "%Y%m%d")
        last_day = datetime.date(last_day.year, last_day.month, last_day.day) + datetime.timedelta(days=-1)

        last_day = last_day.strftime("%Y%m%d")
        if int(last_day[4:6]) == int(month.month):
            day_list.append(last_day)
    return day_list


def get_mondays_between_intervals(monday, left_day, right_day):
    '''
    功能：获取指定时间间隔内所有的指定周几
    参数：
        1)monday:指定周几参数，monday=0为周一，以此类推
        2)left_day:时间间隔的开始日期
        2)right_day:时间间隔的结束日期
    返回：日期列表，元素形式为YYYYMMDD
    '''
    if int(left_day) > int(right_day):
        print('Error: left_day and right_day are illegal, Please check!')
    else:
        date_time = datetime.datetime.strptime(str(right_day), "%Y%m%d")
        diff = monday - date_time.weekday()
        marketDate = date_time + datetime.timedelta(days=diff)
        tmp_day = marketDate.strftime("%Y%m%d")
        day_list = [tmp_day]

        left_day = datetime.datetime.strptime(str(left_day), "%Y%m%d")
        left_day = datetime.date(left_day.year, left_day.month, left_day.day) + datetime.timedelta(days=7)
        left_day = left_day.strftime("%Y%m%d")
        while int(left_day) < int(tmp_day):
            tmp_day = datetime.datetime.strptime(str(tmp_day), "%Y%m%d")
            tmp_day = datetime.date(tmp_day.year, tmp_day.month, tmp_day.day) + datetime.timedelta(days=-7)
            tmp_day = tmp_day.strftime("%Y%m%d")
            day_list.append(tmp_day)

        return day_list


def get_near_day_from_table(path, day):
    '''
    功能：获取指定表中距离指定日期最近的日期
    参数：
        1)path:指定表的存储路径
        2)day:指定日期
    返回：日期，形式为YYYYMMDD
    '''
    flag = 1
    while flag:
        flag = check_source(path, day)
        if flag:
            day = datetime.datetime.strptime(str(day), "%Y%m%d")
            day = datetime.date(day.year, day.month, day.day) + datetime.timedelta(days=-1)
            day = day.strftime("%Y%m%d")
    return day


def get_max_day_from_table(path, month):
    '''
    功能：获取指定表中指定月份的最大日期
    参数：
        1)path:指定表的存储路径
        2)month:指定月份
    返回：日期，形式为YYYYMMDD
    '''
    day_ls = []
    for line in os.popen("""hadoop fs -ls {path}""".format(path=path)).readlines():
        day = line.split('{item}/'.format(item=path.split('/')[-1]))[-1].strip('')[0:8]
        if month in day and len(day) == 8:
            day_ls.append(day)
    res_day = max(day_ls)
    return res_day

def get_std_time(tmStr, level="m"):
    tmStr = str(tmStr)
    if re.match('^\d{8}$', tmStr):
        formatDate = datetime.datetime.strptime(tmStr, "%Y%m%d")
    elif re.match('^\d{4}-\d{2}-\d{2}$', tmStr):
        formatDate = datetime.datetime.strptime(tmStr, "%Y-%m-%d")
    elif re.match('^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$', tmStr):
        formatDate = datetime.datetime.strptime(tmStr[:10], "%Y-%m-%d")
    else:
        formatDate = None
    if formatDate:
        if level == "y":
            return formatDate.strftime("%Y")
        elif level == "m":
            return formatDate.strftime("%Y%m")
        else:
            return formatDate.strftime("%Y%m%d")
    else:
        raise Exception("Format Error!")

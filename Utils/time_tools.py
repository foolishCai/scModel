#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/4/14 16:23
# @author  : gedn
# @contact : danni.ge@yunzhenxin.com
# @File    : time_tools.py
# @Note    : traditional edition for time tools

import datetime
import re


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

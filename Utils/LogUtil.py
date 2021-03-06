#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/5/6 09:48
# @Author  : cai
# @contact : yuwei.chen@yunzhenxin.com
# @File    : LogUtil.py
# @Note    :


import sys
sys.path.append("..")

import os
import logging
from Configs import log_config

class LogUtil(object):
    def __init__(self, log_name=None):
        if log_name is None:
            log_name = log_config['log_name']
        self.logger = logging.getLogger(log_name)
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            # 定义输出格式
            formatter = logging.Formatter('%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s %(message)s')

            # 创建一个文件的handler
            fh = logging.FileHandler(log_config['log_path'] + os.sep + log_name + '.log')
            fh.setFormatter(fmt=formatter)
            self.logger.addHandler(fh)

            # 创建一个控制台输出的handler
            ch = logging.StreamHandler()
            ch.setFormatter(fmt=formatter)
            self.logger.addHandler(ch)

    def info(self, msg):
        self.logger.info(str(msg))

    def debug(self, msg):
        self.logger.debug(str(msg))

    def error(self, msg):
        self.logger.error("!!!ERROR!!!", str(msg))

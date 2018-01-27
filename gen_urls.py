#!/usr/bin/env python
# -*- coding: gbk -*-
########################################################################
# 
# Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: gen_urls.py
Author: daiwenkai(daiwenkai@baidu.com)
Date: 2018/01/28 00:10:09
"""

import os

import sys

path = "./_posts/"
filelist = os.listdir(path)

url_txt = "./urls.txt"

with open(url_txt, 'wb') as fout:
    for file_name in filelist:
        if file_name.endswith(".md"):
            real_name = file_name[11:-3]
            url_name = "https://daiwk.github.io/posts/" + real_name + ".html"
            fout.write(url_name + "\n")

    fout.write("https://daiwk.github.io/tags.html" + "\n")


#!/usr/bin/env python
# -*- coding: utf8 -*-
########################################################################
# 
# Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: gen_dot.py
Author: daiwenkai(daiwenkai@baidu.com)
Date: 2018/05/14 00:22:39
"""

from graphviz import Digraph

##title = 'rl-overview1'
##dot = digraph(comment=title)
##
##dot.node('a', u'king arthur来来来')
##dot.node('b', 'sir bedevere the wise')
##dot.node('l', 'sir lancelot the brave')
##dot.edges(['ab', 'al'])
##dot.edge('b', 'l', constraint='false')
##
##dot.render('dots/demo')

def gen_rl_overview1():
    title = 'rl-overview1'
    dot = Digraph(comment=title, format="png")
    
    dot.node('xgjc', u'序贯决策问题', shape="box")
    dot.node('mdp', '马尔科夫决策过程MDP(s,A,P,R,gamma)', shape="box")
    dot.edge('xgjc', 'mdp', )
    
    dot.node('model_dp', u'基于模型的动态规划方法', shape="box")
    dot.node('no_model_dp', u'无模型的强化学习方法', shape="box")
    dot.edge('mdp', 'model_dp', label="(S,A,P,R,gamma)")
    dot.edge('mdp', 'no_model_dp', label="(S,A,P?,R?,gamma?)")

    dot.render('dots/' + title)


if __name__ == "__main__":
    gen_rl_overview1()

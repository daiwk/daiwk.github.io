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

def gen_rl_overview():
    title = 'rl-overview'
    dot = Digraph(comment=title, format="png")
    
    dot.node('xgjc', u'序贯决策问题', shape="box", style="rounded")
    dot.node('mdp', '马尔科夫决策过程MDP(s,A,P,R,gamma)', shape="box", style="rounded")
    dot.edge('xgjc', 'mdp', )
    
    dot.node('model_dp', u'基于模型的动态规划方法', shape="box", style="rounded")
    dot.node('no_model_dp', u'无模型的强化学习方法', shape="box", style="rounded")
    dot.edge('mdp', 'model_dp', label="(S,A,P,R,gamma)")
    dot.edge('mdp', 'no_model_dp', label="(S,A,P?,R?,gamma?)")

    dot.node('policy_iter_model_dp', u'策略迭代', shape="box", style="rounded")
    dot.node('value_iter_model_dp', u'值迭代', shape="box", style="rounded")
    dot.node('policy_search_model_dp', u'策略搜索', shape="box", style="rounded")
    dot.node('policy_iter_no_model_dp', u'策略迭代', shape="box", style="rounded")
    dot.node('value_iter_no_model_dp', u'值迭代', shape="box", style="rounded")
    dot.node('policy_search_no_model_dp', u'策略搜索', shape="box", style="rounded")

    dot.node('value_function_approximation', u'值函数逼近', shape="box", style="rounded,filled", fillcolor="yellow", fontcolor="red")

    dot.edge('model_dp', 'policy_iter_model_dp', )
    dot.edge('model_dp', 'value_iter_model_dp', )
    dot.edge('model_dp', 'policy_search_model_dp', )
    dot.edge('no_model_dp', 'policy_iter_no_model_dp', )
    dot.edge('no_model_dp', 'value_iter_no_model_dp', )
    dot.edge('no_model_dp', 'policy_search_no_model_dp', )
    dot.edge('value_function_approximation', 'policy_iter_model_dp', style="dashed")
    dot.edge('value_function_approximation', 'value_iter_model_dp', style="dashed")
    dot.edge('value_function_approximation', 'policy_iter_no_model_dp', style="dashed")
    dot.edge('value_function_approximation', 'value_iter_no_model_dp', style="dashed")

    dot.render('dots/' + title)

def gen_rl_overview_value_function():
    title = 'rl-overview-value-function'
    dot = Digraph(comment=title, format="png")
    
    dot.node('xgjc', u'序贯决策问题', shape="box", style="rounded")
    dot.node('mdp', '马尔科夫决策过程MDP(s,A,P,R,gamma)', shape="box", style="rounded")
    dot.edge('xgjc', 'mdp', )
    
    dot.node('model_dp', u'基于模型的动态规划方法', shape="box", style="rounded")
    dot.node('no_model_dp', u'无模型的强化学习方法', shape="box", style="rounded")
    dot.edge('mdp', 'model_dp', label="(S,A,P,R,gamma)")
    dot.edge('mdp', 'no_model_dp', label="(S,A,P?,R?,gamma?)")

    dot.node('policy_iter_model_dp', u'策略迭代', shape="box", style="rounded")
    dot.node('value_iter_model_dp', u'值迭代', shape="box", style="rounded")
    dot.node('policy_search_model_dp', u'策略搜索', shape="box", style="rounded")
    dot.node('monte_carlo_no_model_dp', u'蒙特卡洛方法', shape="box", style="rounded")
    dot.node('temporal_difference_no_model_dp', u'时间差分方法（TD）', shape="box", style="rounded")
    dot.edge('model_dp', 'policy_iter_model_dp', )
    dot.edge('model_dp', 'value_iter_model_dp', )
    dot.edge('model_dp', 'policy_search_model_dp', )
    dot.edge('no_model_dp', 'monte_carlo_no_model_dp', )
    dot.edge('no_model_dp', 'temporal_difference_no_model_dp', )

    dot.node('monte_carlo_on_policy', u'on-policy', shape="box", style="rounded")
    dot.node('temporal_difference_on_policy', u'on-policy', shape="box", style="rounded")
    dot.node('monte_carlo_off_policy', u'off-policy', shape="box", style="rounded")
    dot.node('temporal_difference_off_policy', u'off-policy', shape="box", style="rounded")

    dot.edge('monte_carlo_no_model_dp', 'monte_carlo_on_policy', label=u'行动策略=评估策略')
    dot.edge('monte_carlo_no_model_dp', 'monte_carlo_off_policy', label=u'行动策略!=评估策略')
    dot.edge('temporal_difference_no_model_dp', 'temporal_difference_on_policy', label=u'行动策略=评估策略')
    dot.edge('temporal_difference_no_model_dp', 'temporal_difference_off_policy', label=u'行动策略!=评估策略')

    dot.node('dqn', u'DQN', shape="box", style="rounded,filled", fillcolor="yellow", fontcolor="red")
    dot.edge('dqn', 'temporal_difference_off_policy', style="dashed")

    dot.render('dots/' + title)

def gen_rl_overview_policy_search():
    title = 'rl-overview-policy-search'
    dot = Digraph(comment=title, format="png")
    
    dot.node('policy_search', u'策略搜索方法', shape="box", style="rounded")
    dot.node('has_model', u'基于模型的策略搜索方法', shape="box", style="rounded")
    dot.node('no_model', u'无模型的策略搜索方法', shape="box", style="rounded")
    dot.node('stochastic', u'随机策略', shape="box", style="rounded")
    dot.node('policy_gradient', u'策略梯度方法', shape="box", style="rounded")
    dot.node('statistic_learn', u'统计学习方法', shape="box", style="rounded")
    dot.node('path_calcius', u'路径积分', shape="box", style="rounded")
    dot.node('deterministic', u'确定性策略', shape="box", style="rounded")
    dot.node('gps', u'引导策略搜索GPS', shape="box", style="rounded")
    dot.node('ddpg', u'DDPG', shape="box", style="rounded")
    dot.node('trpo', u'TRPO', shape="box", style="rounded")
    dot.edge('policy_search', 'has_model', )
    dot.edge('policy_search', 'no_model', )
    dot.edge('no_model', 'stochastic', )
    dot.edge('no_model', 'deterministic', )

    dot.edge('stochastic', 'policy_gradient')
    dot.edge('policy_gradient', 'trpo', label=u'单调步长，\n方差约简')
    dot.edge('stochastic', 'statistic_learn')
    dot.edge('stochastic', 'path_calcius')
    dot.edge('has_model', 'gps')
    dot.edge('no_model', 'gps')
    dot.edge('deterministic', 'ddpg')

    dot.render('dots/' + title)

def gen_tf_code_tensorshape():
    title = 'tf_code_tensorshape'
    dot = Digraph(comment=title, format="png")
    
    dot.node('TensorShape', u'TensorShape', shape="box", style="rounded")
    dot.node('TensorShapeBase', u'TensorShapeBase', shape="box", style="rounded")
    dot.node('TensorShapeRep', u'TensorShapeRep', shape="box", style="rounded")
    dot.edge('TensorShape', 'TensorShapeBase', )
    dot.edge('TensorShapeBase', 'TensorShapeRep', )

    dot.node('TensorBuffer', u'TensorBuffer', shape="box", style="rounded")
    dot.node('RefCounted', u'RefCounted', shape="box", style="rounded")
    dot.edge('TensorBuffer', 'RefCounted', )

    dot.render('dots/' + title)

def gen_bert_flow():
    title = "bert_flow_embedding"
    dot = Digraph(comment=title, format="png")
    
    dot.node('input_ids', u'input_ids\n[batch_size, seq_length]', shape="box")
    dot.node('token_type_ids', u'token_type_ids\n[batch_size, seq_length]', shape="box")

    dot.node('embedding_lookup', u'embedding_lookup', shape="box", style="rounded,filled", fillcolor="yellow", fontcolor="red")
    dot.node('embedding_postprocessor', u'embedding_postprocessor', shape="box", style="rounded,filled", fillcolor="yellow", fontcolor="red")
    dot.node('add_op', u'add_op\n+', shape="box", style="rounded,filled", fillcolor="yellow", fontcolor="red")


    dot.node('word_embeddings', u'word_embeddings\n[batch_size, seq_length, embedding_size]', shape="box", style="rounded")
    dot.node('token_type_embeddings', u'token_type_embeddings\n[batch_size, seq_length, embedding_size]', shape="box", style="rounded")
    dot.node('position_embeddings', u'position_embeddings\n[batch_size, seq_length, embedding_size]', shape="box", style="rounded")
    dot.node('embedding_output', u'embedding_output\n[batch_size, seq_length, embedding_size]', shape="box", style="rounded")

    ## tables
    dot.node('word_embeddings_table', u'word_embeddings_table\n[vocab_size, embedding_size]', shape="box", style="rounded,filled", fillcolor="gray")
    dot.node('token_type_embeddings_table', u'token_type_embeddings_table\n[token_type_vocab_size, embedding_size]', shape="box", style="rounded,filled", fillcolor="gray")
    dot.node('position_embeddings_table', u'position_embeddings_table\n[max_position_embeddings, embedding_size]', shape="box", style="rounded,filled", fillcolor="gray")

    dot.edge('input_ids', 'embedding_lookup', )
    dot.edge('embedding_lookup', 'word_embeddings', )
    dot.edge('embedding_lookup', 'word_embeddings_table', )
    dot.edge('word_embeddings', 'embedding_postprocessor', )
    dot.edge('token_type_ids', 'embedding_postprocessor', )
    dot.edge('embedding_postprocessor', 'token_type_embeddings', )
    dot.edge('embedding_postprocessor', 'position_embeddings', )
    dot.edge('embedding_postprocessor', 'token_type_embeddings_table', )
    dot.edge('embedding_postprocessor', 'position_embeddings_table', )
    dot.edge('word_embeddings', 'add_op', )
    dot.edge('token_type_embeddings', 'add_op', )
    dot.edge('position_embeddings', 'add_op', )
    dot.edge('add_op', 'embedding_output', )

    dot.render('dots/' + title)


if __name__ == "__main__":
    gen_rl_overview()
    gen_rl_overview_value_function()
    gen_rl_overview_policy_search()
    gen_tf_code_tensorshape()
    gen_bert_flow()


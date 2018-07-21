---
layout: post
category: "rl"
title: "深入浅出强化学习-chap5 基于时间差分的强化学习方法"
tags: [时间差分, TD, Q-learning, sarsa, ]
---

目录

<!-- TOC -->

- [1. 基于时间差分的强化学习方法](#1-%E5%9F%BA%E4%BA%8E%E6%97%B6%E9%97%B4%E5%B7%AE%E5%88%86%E7%9A%84%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0%E6%96%B9%E6%B3%95)
- [2. python和gym的实例](#2-python%E5%92%8Cgym%E7%9A%84%E5%AE%9E%E4%BE%8B)

<!-- /TOC -->



参考**《深入浅出强化学习》**

## 1. 基于时间差分的强化学习方法


sarsa和Qlearning的最大区别在于:

+ sarsa是用`\(\varepsilon -greedy\)`得到动作`\(a\)`回报`\(r\)`和下一个状态`\(s'\)`，并对`\(s'\)`也使用`\(\varepsilon -greedy\)`得到动作`\(a'\)`和状态行为值函数`\(Q(s',a')\)`，并计算TD目标`\(r+\gamma Q(s',a')\)`
+ Qlearning是用`\(\varepsilon -greedy\)`得到动作`\(a\)`回报`\(r\)`和下一个状态`\(s'\)`【这部分和sarsa一样】，然后计算TD目标`\(r+\gamma max_{a'}Q(s',a')\)`，可见这里不再是通过`\(\varepsilon-greedy\)`选出的`\(a'\)`来算的`\(Q(s',a')\)`，而是`\(max_{a'}Q(s',a')\)`，也就是强制选使Q最大的那个action带来的Q，而非随机策略。
+ 注意，这里二者的`\(Q(s',a')\)`都是基于第一个`\(\varepsilon-greedy\)`得到的新状态`\(s'\)`来搞的。

## 2. python和gym的实例

[https://github.com/daiwk/reinforcement-learning-code/blob/master/qlearning.py](https://github.com/daiwk/reinforcement-learning-code/blob/master/qlearning.py)

代码如下：

```python
import sys
import gym
import random
random.seed(0)
import time
import matplotlib.pyplot as plt

grid = gym.make('GridWorld-v0')
#grid=env.env                     #创建网格世界
states = grid.env.getStates()        #获得网格世界的状态空间
actions = grid.env.getAction()      #获得网格世界的动作空间
gamma = grid.env.getGamma()       #获得折扣因子
#计算当前策略和最优策略之间的差
best = dict() #储存最优行为值函数
def read_best():
    f = open("best_qfunc")
    for line in f:
        line = line.strip()
        if len(line) == 0: continue
        eles = line.split(":")
        best[eles[0]] = float(eles[1])
#计算值函数的误差
def compute_error(qfunc):
    sum1 = 0.0
    for key in qfunc:
        error = qfunc[key] -best[key]
        sum1 += error *error
    return sum1

#  贪婪策略
def greedy(qfunc, state):
    amax = 0
    key = "%d_%s" % (state, actions[0])
    qmax = qfunc[key]
    for i in range(len(actions)):  # 扫描动作空间得到最大动作值函数Q(s,a)
        key = "%d_%s" % (state, actions[i])
        q = qfunc[key]
        if qmax < q:
            qmax = q
            amax = i
    return actions[amax]


#######epsilon贪婪策略
def epsilon_greedy(qfunc, state, epsilon):
    amax = 0
    key = "%d_%s"%(state, actions[0])
    qmax = qfunc[key]
    for i in range(len(actions)):    #扫描动作空间得到最大动作值函数
        key = "%d_%s"%(state, actions[i])
        q = qfunc[key]
        if qmax < q:
            qmax = q
            amax = i
    #概率部分，除了max的为加上1-epsilon，其他的概率一样
    pro = [0.0 for i in range(len(actions))]
    pro[amax] += 1-epsilon
    for i in range(len(actions)):
        pro[i] += epsilon/len(actions)

    ##选择动作
    r = random.random()
    s = 0.0
    for i in range(len(actions)):
        s += pro[i]
        if s>= r: return actions[i]
    return actions[len(actions)-1]

def qlearning(num_iter1, alpha, epsilon):
    x = []
    y = []
    qfunc = dict()   #行为值函数为字典
    #初始化行为值函数为0
    for s in states:
        for a in actions:
            key = "%d_%s"%(s,a)
            qfunc[key] = 0.0
    for iter1 in range(num_iter1):
        x.append(iter1)
        y.append(compute_error(qfunc))

        #初始化初始状态
        s = grid.reset()
        a = actions[int(random.random()*len(actions))] # 应该改成epsilon-greedy?
        t = False
        count = 0
        while False == t and count <100:
            key = "%d_%s"%(s, a)
            #与环境进行一次交互，从环境中得到新的状态及回报
            s1, r, t1, i =grid.step(a)
            key1 = ""
            #s1处的最大动作
            a1 = greedy(qfunc, s1)
            key1 = "%d_%s"%(s1, a1) # 这个时候的qfunc[key1]就是max的
            #利用qlearning方法更新值函数，注意！！这里更新的是key，而不是key1
            qfunc[key] = qfunc[key] + alpha*(r + gamma * qfunc[key1]-qfunc[key])
            #转到下一个状态
            s = s1;
            a = epsilon_greedy(qfunc, s1, epsilon)
            count += 1
    plt.plot(x,y,"-.,",label ="q alpha=%2.1f epsilon=%2.1f"%(alpha,epsilon))
    return qfunc

```

主流程的代码在[https://github.com/daiwk/reinforcement-learning-code/blob/master/learning_and_test.py](https://github.com/daiwk/reinforcement-learning-code/blob/master/learning_and_test.py)中。

```python
import sys
import gym
from qlearning import *
import time
from gym import wrappers
#main函数
if __name__ == "__main__":
   # grid = grid_mdp.Grid_Mdp()  # 创建网格世界
    #states = grid.getStates()  # 获得网格世界的状态空间
    #actions = grid.getAction()  # 获得网格世界的动作空间
    sleeptime=0.5
    terminate_states= grid.env.getTerminate_states()
    #读入最优值函数
    read_best()
#    plt.figure(figsize=(12,6))
    #训练
    qfunc = dict()
    qfunc = qlearning(num_iter1=500, alpha=0.2, epsilon=0.2)
    #画图
    plt.xlabel("number of iterations")
    plt.ylabel("square errors")
    plt.legend()
   # 显示误差图像
    plt.show()
    time.sleep(sleeptime)
    #学到的值函数
    for s in states:
        for a in actions:
            key = "%d_%s"%(s,a)
            print("the qfunc of key (%s) is %f" %(key, qfunc[key]) )
            qfunc[key]
    #学到的策略为：
    print("the learned policy is:")
    for i in range(len(states)):
        if states[i] in terminate_states:
            print("the state %d is terminate_states"%(states[i]))
        else:
            print("the policy of state %d is (%s)" % (states[i], greedy(qfunc, states[i])))
    # 设置系统初始状态
    s0 = 1
    grid.env.setAction(s0)
    # 对训练好的策略进行测试
    grid = wrappers.Monitor(grid, './robotfindgold', force=True)  # 记录回放动画
   #随机初始化，寻找金币的路径
    for i in range(20):
        #随机初始化
        s0 = grid.reset()
        grid.render()
        time.sleep(sleeptime)
        t = False
        count = 0
        #判断随机状态是否在终止状态中
        if s0 in terminate_states:
            print("reach the terminate state %d" % (s0))
        else:
            while False == t and count < 100:
                a1 = greedy(qfunc, s0)
                print(s0, a1)
                grid.render()
                time.sleep(sleeptime)
                key = "%d_%s" % (s0, a)
                # 与环境进行一次交互，从环境中得到新的状态及回报
                s1, r, t, i = grid.step(a1)
                if True == t:
                    #打印终止状态
                    print(s1)
                    grid.render()
                    time.sleep(sleeptime)
                    print("reach the terminate state %d" % (s1))
                # s1处的最大动作
                s0 = s1
                count += 1

```
---
layout: post
category: "rl"
title: "深入浅出强化学习-chap1 绪论"
tags: [深入浅出强化学习, 绪论 ]
---

目录

<!-- TOC -->

- [1. 强化学习可以解决什么问题](#1-强化学习可以解决什么问题)
- [2. 强化学习如何解决问题](#2-强化学习如何解决问题)
- [3. 强化学习算法分类及发展趋势](#3-强化学习算法分类及发展趋势)
- [4. gym](#4-gym)
    - [reset()函数](#reset函数)
    - [render()函数](#render函数)
    - [step()函数](#step函数)
- [5. 整体脉络](#5-整体脉络)
    - [强化学习的基本算法](#强化学习的基本算法)
        - [第一部分 强化学习基础](#第一部分-强化学习基础)
            - [chap2 马尔科夫决策过程](#chap2-马尔科夫决策过程)
            - [chap3 基于动态规划的强化学习算法](#chap3-基于动态规划的强化学习算法)
        - [第二部分 基于值函数的强化学习方法](#第二部分-基于值函数的强化学习方法)
            - [chap4 基于蒙特卡罗的强化学习算法](#chap4-基于蒙特卡罗的强化学习算法)
            - [chap5 基于时间差分的强化学习算法](#chap5-基于时间差分的强化学习算法)
            - [chap6 基于值函数逼近的强化学习算法](#chap6-基于值函数逼近的强化学习算法)
        - [第三部分 基于直接策略搜索的强化学习方法](#第三部分-基于直接策略搜索的强化学习方法)
            - [chap7 策略梯度理论](#chap7-策略梯度理论)
            - [chap8 TRPO](#chap8-trpo)
            - [chap9 确定性策略搜索](#chap9-确定性策略搜索)
            - [chap10 引导策略搜索的强化学习算法](#chap10-引导策略搜索的强化学习算法)
        - [第四部分 强化学习研究及前沿](#第四部分-强化学习研究及前沿)
        - [chap11 逆向强化学习算法](#chap11-逆向强化学习算法)
    - [chap12 组合策略梯度和值函数方法](#chap12-组合策略梯度和值函数方法)
    - [chap13 值迭代网络](#chap13-值迭代网络)
    - [chap14 PILCO方法及其扩展](#chap14-pilco方法及其扩展)
    - [强化学习算法所用到的基础知识](#强化学习算法所用到的基础知识)

<!-- /TOC -->



参考**《深入浅出强化学习》**

## 1. 强化学习可以解决什么问题

强化学习解决的是**智能决策**（即**序贯决策**）问题，也就是说需要**连续不断**地做出决策，才能实现**最终目标**。

## 2. 强化学习如何解决问题

监督学习解决的是**智能感知**的问题，学习的是输入长得像什么（特征），以及和它对应的是什么（标签）。需要的是多样化的标签数据。

强化学习不关心输入长什么样，只关心**当前输入下**应该采取什么**动作**，才能实现**最终目标**。需要智能体**不断**地与**环境**交互，不断尝试。需要的是**带有回报**的**交互数据**。

几个关键时间点：

1. 1998年，Richard S. Sutton出版了《强化学习导论》第一版，总结了1998年以前强化学习的各种进展。关注和发展最多的是**表格型强化学习算法**。基于**直接策略搜索**的方法也被提出来了，例如1992年的Rinforce算法，直接对策略梯度进行估计。
2. 2013年，deepmind提出了**DQN(deep Q network)**，将深度学习与强化学习结合形成深度强化学习。
3. 2016、2017年，alphago连续两年击败围棋世界冠军。

## 3. 强化学习算法分类及发展趋势

根据强化学习是否依赖模型：

+ 基于模型的强化学习：利用与环境交互得到的数据学习系统或者环境模型，再**基于模型**进行序贯决策。效率会比无模型的高
+ 无模型的强化学习：**直接**利用与环境交互获得的数据改善自身的行为。有些根本无法建模的任务，只能利用无模型的强化学习算法，也更具有通用性。

根据策略的更新和学习方法：

+ 基于值函数的强化学习：学习**值函数**，最终的策略根据值函数**贪婪**得到。任意状态下，**值函数最大的动作就是当前最优策略。**
+ 基于直接策略搜索的强化学习：将策略**参数化**，学习实现目标的**最优参数**。
+ 基于AC的方法：**联合**使用值函数和直接策略搜索。

根据环境返回的回报函数是否已知：

+ 正向强化学习：**回报函数**是**人为指定**的
+ 逆向强化学习：通过机器学习的方法由函数自己学出回报，因为很多时候回报无法人为指定，例如无人机的表演

其他强化学习：分层强化学习、元强化学习、多智能体强化学习、关系强化学习、迁移强化学习等。

强化学习的发展趋势：

+ 强化学习和深度学习的结合会更加紧密

机器学习算法分为三大类：监督学习、无监督学习、强化学习。三类算法联合使用的效果更好，例如基于深度强化学习的对话生成等。

+ 强化学习和专业知识的结合会更加紧密

对于不同领域，可以重塑回报函数，或者修改网络结构。代表作：NIPS2016的最佳论文：值迭代网络【[Value Iteration Networks](https://arxiv.org/pdf/1602.02867.pdf)，github代码：[https://github.com/TheAbhiKumar/tensorflow-value-iteration-networks](https://github.com/TheAbhiKumar/tensorflow-value-iteration-networks)】。

+ 强化学习算法理论分析会更强，算法会更稳定和高效

代表作有：基于深度能量的策略方法、值函数与策略方法的等价性等。

+ 强化学习与脑科学、认知神经科学、记忆的联系会更紧密

DeepMind和伦敦大学学院是这一流派的代表。

## 4. gym

以CartPoleEnv为例：

```python
env = gym.make('CartPole-v0')
env.reset()
env.render()
```

CartPoleEnv的环境文件位于```/gym/gym/envs/classic_control/cartpole.py```
注册的代码在```/gym/gym/envs/__init__.py```中

### reset()函数

智能体需要一次次尝试病积累经验，然后从经验中学到好的动作。**每一次尝试**称为**一条轨迹**，或者**一个episode**。每次尝试都需要达到终止状态，一次尝试结束后，就需要智能体重新初始化。

reset()是重新初始化函数。实现如下：

```python
    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        return np.array(self.state)
```

即，利用均匀随机分布初始化环境状态，然后设置当前步数为None，病返回环境的初始化状态。

### render()函数

render()扮演图像引擎的角色。为了便于直观显示环境中物理的状态，需要除了物理引擎之外的图像引擎。源码如下:

注释参考：[https://github.com/daiwk/reinforcement-learning-code/blob/master/cartpole_notes.py](https://github.com/daiwk/reinforcement-learning-code/blob/master/cartpole_notes.py)

```python
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            # 创建台车
            l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
            axleoffset =cartheight/4.0
            cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)]) ## 填充一个矩形
            #添加台车转换矩阵属性
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            #加入几何体台车
            self.viewer.add_geom(cart)
            #创建摆杆
            l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
            pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            pole.set_color(.8,.6,.4)
            #添加摆杆转换矩阵属性
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            #加入几何体
            self.viewer.add_geom(pole)
            #创建摆杆和台车之间的连接
            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5,.5,.8)
            self.viewer.add_geom(self.axle)
            #创建台车来回滑动的轨道，即一条直线
            self.track = rendering.Line((0,carty), (screen_width,carty))
            self.track.set_color(0,0,0)
            self.viewer.add_geom(self.track)

        if self.state is None: return None

        x = self.state
        cartx = x[0]*scale+screen_width/2.0 # MIDDLE OF CART
        #设置平移属性
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array = mode=='rgb_array')
```

创建完cart的形状，给cart添加平移属性和旋转属性，将车的位移设置到cart的平移属性中，cart就会根据系统的状态变化左右移动。

### step()函数

step()函数扮演物理引擎的角色。

+ **输入**：**动作**a
+ **输出**：**下一步的状态**、**立即回报**、**是否终止**、**调试项**。调试信息可能为空，但要填默认值{}。

描述了智能体与环境交互的所有信息。利用智能体的运动学模型和动力学模型计算下一步的状态和立即回报，并判断是否达到终止状态。

```python
    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state
        x, x_dot, theta, theta_dot = state # 系统的当前状态
        force = self.force_mag if action==1 else -self.force_mag # 输入动作，即作用到车上的力
        costheta = math.cos(theta) # cos
        sintheta = math.sin(theta) # sin
        # 车摆的动力学方程式，即加速度与动作之间的关系
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta* temp) / (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass)) # 摆的角加速度
        xacc  = temp - self.polemass_length * thetaacc * costheta / self.total_mass # 小车的平移加速度
        x  = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc # 积分球下一步的状态
        self.state = (x,x_dot,theta,theta_dot)
        done =  x < -self.x_threshold \
                or x > self.x_threshold \
                or theta < -self.theta_threshold_radians \
                or theta > self.theta_threshold_radians
        done = bool(done)

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {}
```

## 5. 整体脉络

### 强化学习的基本算法

#### 第一部分 强化学习基础

##### chap2 马尔科夫决策过程

强化学习解决的是序贯决策问题，一般的序贯决策问题可以用**马尔科夫决策过程（MDP）**的框架来表示。

##### chap3 基于动态规划的强化学习算法

对于**模型已知**的MDP问题，**动态规划**是一个不错的解。由此引出**广义策略迭代**的方法。而广义策略迭代的方法也适用于无模型的方法，是**基于值函数强化学习的基本框架**。因此有chap4的基于蒙特卡罗方法、chap5的基于时间查分方法、chap6的基于值函数逼近方法。

#### 第二部分 基于值函数的强化学习方法

##### chap4 基于蒙特卡罗的强化学习算法

**无模型**的强化学习算法，是整个强化学习算法的**核心**。基于**值函数**的强化学习算法的**核心**是计算**值函数的期望**。值函数是一个**随机变量**，其**期望**的计算可以通过**蒙特卡罗方法**得到。

##### chap5 基于时间差分的强化学习算法

基于**蒙特卡罗**的强化学习算法通过蒙特卡罗模拟计算期望，需要等每次**试验结束后**再**对值函数进行估计**，**收敛速度漫**。而时间差分的方法只需要**一步**便**更新**，**效率高、收敛速度快**。

##### chap6 基于值函数逼近的强化学习算法

chap4和chap5介绍的是**表格型**强化学习，即，**状态**空间和**动作**空间都是**有限集**，动作值函数可以用一个表格来描述，表格的索引分别为**状态量**和**动作量**。但当状态空间和动作空间都很大时，甚至两个空间都是连续空间时，这就无法用表格表示，可以用**函数逼近理论**对值函数进行逼近。本章介绍了**DQN**及其变种（Double DQN、Prioritized Replay DQN、Dueling DQN等）。

#### 第三部分 基于直接策略搜索的强化学习方法

##### chap7 策略梯度理论

区别于基于值函数的方法，强化学习的第二大类算法是**直接策略搜索**方法。就是将策略进行**参数化**，然后在**参数空间**直接**搜索最优策略**。直接策略搜索方法中，最直接最简单的方法是**策略梯度**方法 。

##### chap8 TRPO

**基于策略梯度**方法最具挑战性的是**更新步长的确定**。TRPO方法通过理论分析得到**单调非递减**的策略更新方法.

##### chap9 确定性策略搜索

当**动作空间**维数很高时，智能体的探索效率会很低，利用**确定性策略**可以**免除对动作空间的探索**，提升算法的收敛速度。

##### chap10 引导策略搜索的强化学习算法

chap7（策略梯度）\8（TRPO）\9（确定性策略搜索）章讲的是**无模型**的**直接策略搜索方法**。而对于机器人等复杂系统，无模型的方法随机初始化很难找到成功的解，所以算法很难收敛。此时，可以利用**传统控制器**来**引导策略进行搜索**。

#### 第四部分 强化学习研究及前沿

#### chap11 逆向强化学习算法

很多实际问题，往往不知道回报函数，所以可以通过逆向强化学习来**学习回报函数**。

### chap12 组合策略梯度和值函数方法

将策略梯度方法和值函数方法相组合。

### chap13 值迭代网络

先介绍DQN，然后介绍值迭代网络。

### chap14 PILCO方法及其扩展

PLICO(probalistic inference for learning control)是一种**基于模型**的强化学习算法，将**模型误差**纳入考虑范围内，一般只需要训练几次到几十次就可以成功实现对单摆等典型非线性系统的稳定性控制，而基于无模型的强化学习则需要训练上万次。

### 强化学习算法所用到的基础知识

+ 第2章：概率学基础、**随机策略**。
+ 第3章：**模型已知**时，值函数的求解可转化为线性方程组的求解。**线性方程组的数值求解方法**——**高斯·赛尔德迭代法**，并利用时变与泛函分析中的压缩映射证明了算法的收敛性。
+ 第4章：值函数是**累积回报的期望**。统计学中利用**采样数据**可以用来**计算期望**：**重要性**采样、**拒绝性**采样、**MCMC**方法。
+ 第8章：TRPO中，替代目标函数用了**信息论**的**熵**和**相对熵**的概念，同时TRPO的求解需要用到各种优化算法。
+ 第10章：引导策略搜索强化学习的优化目标用到了**KL散度**和**变分推理**，及大型的**并行优化算法**，例如，**LBFGS**优化算法、**ADMM**方法（交替方向乘子法）

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

render()扮演图像引擎的角色。为了便于直观显示环境中物理的状态，需要除了物理引擎之外的图像引擎。源码如下（注释参考：[https://github.com/gxnk/reinforcement-learning-code/blob/master/%E7%AC%AC%E4%B8%80%E8%AE%B2%20%20gym%20%E5%AD%A6%E4%B9%A0%E5%8F%8A%E4%BA%8C%E6%AC%A1%E5%BC%80%E5%8F%91/cartpole_notes.py](https://github.com/gxnk/reinforcement-learning-code/blob/master/%E7%AC%AC%E4%B8%80%E8%AE%B2%20%20gym%20%E5%AD%A6%E4%B9%A0%E5%8F%8A%E4%BA%8C%E6%AC%A1%E5%BC%80%E5%8F%91/cartpole_notes.py)）：

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

step()函数扮演物理引擎的角色。输入是动作a，输出是下一步的状态、立即回报、是否终止、调试项。描述了智能体与环境交互的所有信息。利用智能体的运动学模型和动力学模型计算下一步的状态和立即回报，并判断是否达到终止状态。

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


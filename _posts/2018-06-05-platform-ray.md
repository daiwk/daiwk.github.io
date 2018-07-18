---
layout: post
category: "platform"
title: "ray"
tags: [ray, ]
---

目录

<!-- TOC -->

- [简介](#%E7%AE%80%E4%BB%8B)
- [ray-rllib](#ray-rllib)
- [gym](#gym)

<!-- /TOC -->

## 简介

参考[伯克利AI分布式框架Ray，兼容TensorFlow、PyTorch与MXNet](https://www.jiqizhixin.com/articles/2018-01-10-2)

github: [https://github.com/ray-project](https://github.com/ray-project)

tutorials: [http://ray.readthedocs.io/en/latest/tutorial.html](http://ray.readthedocs.io/en/latest/tutorial.html)


## ray-rllib

[RLlib: Abstractions for Distributed Reinforcement Learning](https://arxiv.org/abs/1712.09381)

最基础用法(使用```lib/python2.7/site-packages/ray/rllib/train.py```)：

```shell
python ./train.py --run DQN --env CartPole-v0 
python ./train.py --run APEX --env CartPole-v0
python ./train.py --run APEX_DDPG --env Pendulum-v0
python ./train.py --run DDPG --env Pendulum-v0
python ./train.py --run DDPG2 --env Pendulum-v0
python ./train.py --run A3C --env CartPole-v0 
```

<html>
<br/>
<img src='../assets/rllib-stack.svg' style='max-height: 250px'/>
<br/>
</html>

<html>
<br/>
<img src='../assets/rllib-api.svg' style='max-height: 250px'/>
<br/>
</html>


## gym

gym的所有自带的环境（注意，rllib里的ddpg适用的问题是Box的，Discrete的不能解）

[https://github.com/openai/gym/wiki/Table-of-environments](https://github.com/openai/gym/wiki/Table-of-environments)

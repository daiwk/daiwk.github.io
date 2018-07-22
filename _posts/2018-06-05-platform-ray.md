---
layout: post
category: "platform"
title: "ray"
tags: [ray, ]
---

目录

<!-- TOC -->

- [简介](#简介)
- [使用](#使用)
    - [简单的数据并行](#简单的数据并行)
- [ray-rllib](#ray-rllib)
- [gym自带的环境](#gym自带的环境)
- [ray的ui](#ray的ui)

<!-- /TOC -->

## 简介

参考[伯克利AI分布式框架Ray，兼容TensorFlow、PyTorch与MXNet](https://www.jiqizhixin.com/articles/2018-01-10-2)

github: [https://github.com/ray-project](https://github.com/ray-project)

tutorials: [http://ray.readthedocs.io/en/latest/tutorial.html](http://ray.readthedocs.io/en/latest/tutorial.html)

## 使用

### 简单的数据并行

定义：

```python
# A regular Python function.
def regular_function(x):
    return x

# A Ray remote function.
@ray.remote
def remote_function(x):
    return x
```

运行时，```remote_function.remote()```返回的是一个objectID，然后create了一个task。想要拿到结果，就要执行```ray.get```：

```python
 >>> regular_function()
 1

 >>> remote_function.remote(1)
 ObjectID(1c80d6937802cd7786ad25e50caf2f023c95e350)

 >>> ray.get(remote_function.remote(1))
 1
```

数据并行：

```python
results = [slow_function.remote(i) for i in range(7)]
ray.get(results)
```

会发现最后的结果是保持了原来的顺序的，应该是并行执行，然后最后会按先后顺序聚合。



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


## gym自带的环境

gym的所有自带的环境（注意，rllib里的ddpg适用的问题是Box的，Discrete的不能解）

[https://github.com/openai/gym/wiki/Table-of-environments](https://github.com/openai/gym/wiki/Table-of-environments)

## ray的ui

首先参考[http://ipywidgets.readthedocs.io/en/latest/user_install.html](http://ipywidgets.readthedocs.io/en/latest/user_install.html)

```shell
pip install ipywidgets
```

然后设置一下：

```shell
jupyter nbextension enable --py widgetsnbextension
```

再然后，启动jupyter:

```shell
nohup jupyter notebook &
```

还需要装个chrome的插件trace-viewer：

[https://github.com/catapult-project/catapult](https://github.com/catapult-project/catapult)

```shell
git clone https://github.com/catapult-project/catapult.git
```


然后还要安装bokeh

```shell
pip install bokeh
```

在我们执行```ray.init(num_cpus=4, redirect_output=True)```的时候会有提示，例如：

```shell
View the web UI at http://localhost:8889/notebooks/ray_ui82961.ipynb?token=43724f05dc0a2b1897bf50e9c9d01541a1bfef8ba9030eac
```

点开这个url，就可以看到ui的几个用法了：

```python
import ray.experimental.ui as ui
ray.init(redis_address=os.environ["REDIS_ADDRESS"]) ## 从刚init的那个环境再init一下下

```

使用```ui.object_search_bar()```可以查看objectid的信息，例如：

```python
Search for an object: 4fc7f1f5b7a05060629b65daa57dc36f4142c2db
## 输出
{'DataSize': 516,
 'Hash': 'cb38c7a61a24275500000000140000004fc7f1f5b7a05060629b65daa57dc36f',
 'IsPut': False,
 'ManagerIDs': ['897fa656adc4b7b9d7490502c77d97235146c871'],
 'TaskID': '4ec7f1f5b7a05060629b65daa57dc36f4142c2db'}
```

使用```ui.task_search_bar()```可以查看taskid的信息

而使用```ui.task_timeline()```则稍微有点复杂。。点击"View task timeline"，会生成一个json文件，例如```/var/folders/9q/91xmxq4d1zl__l2w9lsp22mj6x47pl/T/tmpr6x81_js.json```，然后就需要执行：

```shell
catapult/tracing/bin/trace2html /var/folders/9q/91xmxq4d1zl__l2w9lsp22mj6x47pl/T/tmpr6x81_js.json --output=my_trace.html && open my_trace.html
```

这样就可以在浏览器中打开啦~

<html>
<br/>
<img src='../assets/ray-ui-trace-viewer.png' style='max-height: 250px'/>
<br/>
</html>

参考：[https://daiwk.github.io/assets/my_trace.html](https://daiwk.github.io/assets/my_trace.html)

从图中可以看出，我跑了三次，

+ 第一次是20多个tasks，分布在4个不同的worker上并行执行
+ 第二次1个task，建了一个新的worker
+ 第三次7个task，分布在4个worker上并行执行

另外，下面这几个是基于bokeh的，可以直接在jupyter里看：

```python
ui.task_completion_time_distribution()
ui.cpu_usage()
ui.cluster_usage()
```

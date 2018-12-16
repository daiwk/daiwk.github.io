---
layout: post
category: "nlp"
title: "pytext"
tags: [pytext, caffe2, pytext-nlp]
---

目录

<!-- TOC -->

- [安装](#安装)
- [训练](#训练)
- [导出模型](#导出模型)
- [部署predictor服务](#部署predictor服务)

<!-- /TOC -->


## 安装

使用python3

```shell
pip install pytext-nlp
```

注意：[https://github.com/facebookresearch/pytext/issues/115](https://github.com/facebookresearch/pytext/issues/115)


## 训练

```shell
pytext train < demo/configs/docnn.json
```

## 导出模型

训练完了后，可以导出模型

```shell
mkdir ./models
pytext export --output-path ./models/demo.c2 < ./demo/configs/docnn.json
```

## 部署predictor服务


可以自己build一个镜像，

```shell
cd demo/predictor_service/
docker build -t predictor_service .
```

当然也可以用我编好的啦[https://hub.docker.com/r/daiwk/caffe2](https://hub.docker.com/r/daiwk/caffe2)

```shell
docker pull daiwk/caffe2
docker run -it -v ~/git_daiwk/pytext/models/:/models -p 8080:8080 daiwk/caffe2
```

然后在container中

```shell
/app/server /models/demo.c2
```

然后新开一个窗口直接curl就行：

```shell
curl -G "http://localhost:8080" --data-urlencode "doc=Flights from Seattle to San Francisco"
```

会得到输出：

```shell
doc_scores:alarm/modify_alarm:-2.13494
doc_scores:alarm/set_alarm:-2.02492
doc_scores:alarm/show_alarms:-2.05924
doc_scores:alarm/snooze_alarm:-2.02332
doc_scores:alarm/time_left_on_alarm:-2.11147
doc_scores:reminder/set_reminder:-2.00476
doc_scores:reminder/show_reminders:-2.21686
doc_scores:weather/find:-2.07725
```

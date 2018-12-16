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
    - [c++的predictor部分](#c的predictor部分)
        - [thrift api](#thrift-api)
- [...](#)
- [Copy local files to /app](#copy-local-files-to-app)
- [Compile app](#compile-app)
- [Add library search paths](#add-library-search-paths)

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

## c++的predictor部分

### thrift api

```predictor.thrift```如下

```c++
namespace cpp predictor_service

service Predictor {
   // Returns list of scores for each label
   map<string,list<double>> predict(1:string doc),
}
```

### 实现server

完整代码见[https://github.com/facebookresearch/pytext/blob/master/demo/predictor_service/server.cpp](https://github.com/facebookresearch/pytext/blob/master/demo/predictor_service/server.cpp)

```c++
class PredictorHandler : virtual public PredictorIf {
  private:
    NetDef mPredictNet;
    Workspace mWorkspace;

    NetDef loadAndInitModel(Workspace& workspace, string& modelFile) {
      auto db = unique_ptr<DBReader>(new DBReader("minidb", modelFile));
      auto metaNetDef = runGlobalInitialization(move(db), &workspace);
      const auto predictInitNet = getNet(
        *metaNetDef.get(),
        PredictorConsts::default_instance().predict_init_net_type()
      );
      CAFFE_ENFORCE(workspace.RunNetOnce(predictInitNet));

      auto predictNet = NetDef(getNet(
        *metaNetDef.get(),
        PredictorConsts::default_instance().predict_net_type()
      ));
      CAFFE_ENFORCE(workspace.CreateNet(predictNet));

      return predictNet;
    }
// ...
  public:
    PredictorHandler(string &modelFile): mWorkspace("workspace") {
      mPredictNet = loadAndInitModel(mWorkspace, modelFile);
    }
// ...
}
```

实现predict函数

```c++
class PredictorHandler : virtual public PredictorIf {
// ...
  public:
    void predict(map<string, vector<double>>& _return, const string& doc) {
      // Pre-process: tokenize input doc
      vector<string> tokens;
      string docCopy = doc;
      tokenize(tokens, docCopy);

      // Feed input to model as tensors
      Tensor valTensor = TensorCPUFromValues<string>(
        {static_cast<int64_t>(1), static_cast<int64_t>(tokens.size())}, {tokens}
      );
      BlobGetMutableTensor(mWorkspace.CreateBlob("tokens_vals_str:value"), CPU)
        ->CopyFrom(valTensor);
      Tensor lensTensor = TensorCPUFromValues<int>(
        {static_cast<int64_t>(1)}, {static_cast<int>(tokens.size())}
      );
      BlobGetMutableTensor(mWorkspace.CreateBlob("tokens_lens"), CPU)
        ->CopyFrom(lensTensor);

      // Run the model
      CAFFE_ENFORCE(mWorkspace.RunNet(mPredictNet.name()));

      // Extract and populate results into the response
      for (int i = 0; i < mPredictNet.external_output().size(); i++) {
        string label = mPredictNet.external_output()[i];
        _return[label] = vector<double>();
        Tensor scoresTensor = mWorkspace.GetBlob(label)->Get<Tensor>();
        for (int j = 0; j < scoresTensor.numel(); j++) {
          float score = scoresTensor.data<float>()[j];
          _return[label].push_back(score);
        }
      }
    }
// ...
}
```

### 编译

需要有```libthrift.so, libcaffe2.so, libprotobuf.so and libc10.so```。

Makefile文件如下

```shell
CPPFLAGS += -g -std=c++11 -std=c++14 \
  -I./gen-cpp \
  -I/pytorch -I/pytorch/build \
      -I/pytorch/aten/src/ \
      -I/pytorch/third_party/protobuf/src/
CLIENT_LDFLAGS += -lthrift
SERVER_LDFLAGS += -L/pytorch/build/lib -lthrift -lcaffe2 -lprotobuf -lc10

# ...

server: server.o gen-cpp/Predictor.o
      g++ $^ $(SERVER_LDFLAGS) -o $@

clean:
      rm -f *.o server
```

在Dockerfile中，有如下命令

```shell
# Copy local files to /app
COPY . /app
WORKDIR /app

# Compile app
RUN thrift -r --gen cpp predictor.thrift
RUN make

# Add library search paths
RUN echo '/pytorch/build/lib/' >> /etc/ld.so.conf.d/local.conf
RUN echo '/usr/local/lib/' >> /etc/ld.so.conf.d/local.conf
RUN ldconfig
```

## 部署predictor服务

参考[https://pytext-pytext.readthedocs-hosted.com/en/latest/serving_models_in_production.html#test-run-the-server](https://pytext-pytext.readthedocs-hosted.com/en/latest/serving_models_in_production.html#test-run-the-server)

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

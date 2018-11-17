---
layout: post
category: "nlp"
title: "bert代码解读——application"
tags: [bert代码解读, bert code, application]
---

目录

<!-- TOC -->

- [basics](#basics)
    - [TPUEstimator](#tpuestimator)
        - [train](#train)
        - [predict](#predict)
        - [evaluate](#evaluate)
- [extract-features](#extract-features)
    - [extract-features.py](#extract-featurespy)
- [pretrain](#pretrain)
    - [create-pretraining-data.py](#create-pretraining-datapy)
    - [run-pretraining.py](#run-pretrainingpy)
- [classification](#classification)
    - [run-classifier.py](#run-classifierpy)
- [squad](#squad)
    - [run-squad.py](#run-squadpy)

<!-- /TOC -->

## basics

### TPUEstimator

类的定义在```site-packages/tensorflow/contrib/tpu/python/tpu/tpu_estimator.py```中

#### train

在```tensorflow/tensorflow/contrib/tpu/python/tpu/tpu_estimator.py```文件中（如果装的是cpu/gpu版的，好像没这个函数）：

```python
  def train(self,
            input_fn,
            hooks=None,
            steps=None,
            max_steps=None,
            saving_listeners=None):
    rendezvous = error_handling.ErrorRendezvous(num_sources=3)
    self._rendezvous[model_fn_lib.ModeKeys.TRAIN] = rendezvous
    try:
      return super(TPUEstimator, self).train(
          input_fn=input_fn, hooks=hooks, steps=steps, max_steps=max_steps,
          saving_listeners=saving_listeners
      )
    except Exception:  # pylint: disable=broad-except
      rendezvous.record_error('training_loop', sys.exc_info())
    finally:
      rendezvous.record_done('training_loop')
      rendezvous.raise_errors()
```

#### predict

在```tensorflow/tensorflow/contrib/tpu/python/tpu/tpu_estimator.py```文件中（如果装的是cpu/gpu版的，好像没这个函数）：


```python
  def predict(self,
              input_fn,
              predict_keys=None,
              hooks=None,
              checkpoint_path=None,
              yield_single_examples=True):
    rendezvous = error_handling.ErrorRendezvous(num_sources=3)
    self._rendezvous[model_fn_lib.ModeKeys.PREDICT] = rendezvous
    try:
      for result in super(TPUEstimator, self).predict(
          input_fn=input_fn,
          predict_keys=predict_keys,
          hooks=hooks,
          checkpoint_path=checkpoint_path,
          yield_single_examples=yield_single_examples):
        yield result
    except Exception:  # pylint: disable=broad-except
      rendezvous.record_error('prediction_loop', sys.exc_info())
    finally:
      rendezvous.record_done('prediction_loop')
      rendezvous.raise_errors()

    rendezvous.record_done('prediction_loop')
    rendezvous.raise_errors()
```

#### evaluate

看tf的源码，在```tensorflow/tensorflow/contrib/tpu/python/tpu/tpu_estimator.py```文件中（如果装的是cpu/gpu版的，好像没这个函数）：

```python
  def evaluate(self, input_fn, steps=None, hooks=None, checkpoint_path=None,
               name=None):
    rendezvous = error_handling.ErrorRendezvous(num_sources=3)
    self._rendezvous[model_fn_lib.ModeKeys.EVAL] = rendezvous
    try:
      return super(TPUEstimator, self).evaluate(
          input_fn, steps=steps, hooks=hooks, checkpoint_path=checkpoint_path,
          name=name
      )
    except Exception:  # pylint: disable=broad-except
      rendezvous.record_error('evaluation_loop', sys.exc_info())
    finally:
      rendezvous.record_done('evaluation_loop')
      rendezvous.raise_errors()
```

## extract-features

### extract-features.py

## pretrain

### create-pretraining-data.py

### run-pretraining.py

## classification

### run-classifier.py

## squad

### run-squad.py

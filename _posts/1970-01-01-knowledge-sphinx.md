---
layout: post
category: "knowledge"
title: "sphinx"
tags: [sphinx, ]
---

目录

<!-- TOC -->


<!-- /TOC -->

以[https://github.com/daiwk/demo-code](https://github.com/daiwk/demo-code)为例，讲解一下sphinx的使用咯~

目录结构如下：

```
.
├── docs
│   ├── Makefile
│   ├── build
│   ├── gen_doc.sh
│   ├── make.bat
│   └── source
│       ├── _static
│       ├── _templates
│       ├── conf.py
│       ├── demo.models.rst
│       ├── demo.rst
│       ├── demo.tools.rst
│       ├── index.rst
│       ├── introduction.rst
│       └── modules.rst
├── python
│   └── demo
│       ├── __init__.py
│       ├── models
│       │   ├── __init__.py
│       │   └── demo1.py
│       └── tools
│           ├── __init__.py
│           └── demo2.py
└── requirements.txt
```

其中的```requirements.txt```写了需要的依赖包，例如tensroflow。然后在demo1.py和demo2.py中，我们使用```'''xxx'''```或者```"""xxx"""```这样的注释，让sphinx可以自动捕捉到。

而docs目录下，是通过```sphinx-quickstart```生成的，我们直接copy就行了。

然后在```docs/source/conf.py```中加上：

```python
sys.path.insert(0, os.path.abspath('./../../python/')) 
```

然后使用gen_doc.sh:

```shell
cd docs
sh -x gen_doc.sh
```

可以看到这个shell里有下面一句命令：

```shell
sphinx-apidoc -o ./source ../python -f
```


然后在```docs/source/index.srt```中记得加上下面这段，表示要显示modules.rst这个文件

```shell
.. toctree::
   :maxdepth: 1
   :caption: API docs

   modules
```

首先，用github登录[https://readthedocs.org/](https://readthedocs.org/)。

然后import一个project，

然后『管理』->『高级设置』

在『所需求的文件:』中把```requirements.txt```填进去

在『Python 配置文件:』中把```docs/source/conf.py```填进去

然后只要一提交，就会自动触发构建啦

然后去对应的网址看看，例如我的这个叫```https://demo-code.readthedocs.io/zh_CN/latest/```，当然，你应该取另一个名字，比如```https://yyy-demo-code.readthedocs.io/zh_CN/latest/```


如果只想暴露某些函数/类（注意：只能链接到类Demo1，而没法到Demo1的get函数），当然，也可以直接在py文件里声明一个函数（不属于某个类），这样就可以暴露这个函数啦：

```
.. autofunction:: demo.models.demo1.Demo1
```

另外，类的private函数（```__xxx```开头的函数）以及类函数第一个变量self是不会生成文档的


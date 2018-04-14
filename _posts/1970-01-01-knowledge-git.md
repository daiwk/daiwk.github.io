---
layout: post
category: "knowledge"
title: "git常用操作"
tags: [git, ]
---

目录

<!-- TOC -->

- [git常用操作](#git常用操作)
    - [如何把fork的源代码的更新merge到自己的来](#如何把fork的源代码的更新merge到自己的来)
    - [如何回滚到某一次commit](#如何回滚到某一次commit)
    - [如何把一个分支的代码合并到另一个分支](#如何把一个分支的代码合并到另一个分支)
- [如何往github的某一个分支push代码](#如何往github的某一个分支push代码)

<!-- /TOC -->

## git常用操作

### 如何把fork的源代码的更新merge到自己的来

```shell
git remote -v 
git remote add upstream git@github.com:xxx/xxx.git
git fetch upstream
git merge upstream/master
git push 
```

### 如何回滚到某一次commit 

```shell
git reset --hard HEAD^         #回退到上个版本
git reset --hard HEAD~3        #回退到前3次提交之前，以此类推，回退到n次提交之前
git reset --hard commit_id     #退到/进到 指定commit的sha码
```

强行推送到远端

```
git push origin HEAD --force
```

### 如何把一个分支的代码合并到另一个分支

```shell
git clone xxx
git checkout branch_a
git merge origin/branch_b
# 然后本地处理conflict
git add .
git commit -m "合并分支"
git push origin HEAD:refs/for/branch_a
```

## 如何往github的某一个分支push代码

+ case1

如果已经在github在界面上建好了分支```test-dwk```，那么

```shell
git pull
git branch -a
```

创建本地分支test-dwk(名字必须一致！！)，并且和远程origin/test-dwk分支关联

```shell 
git checkout test-dwk
```

提交
```shell
git push --set-upstream origin test-dwk
```
+ case2

如果没有在github在界面上建好了分支```test-dwk```，那么

```shell
git branch -b test-dwk
```

这样就创建了本地分支test-dwk

提交

```shell
git push --set-upstream origin test-dwk
```

这样，远程也就多了origin/test-dwk这样一个分支

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

git revert 和 git reset的区别：

1. git revert是用一次新的commit来回滚之前的commit，git reset是直接删除指定的commit。 
2. 在回滚这一操作上看，效果差不多。但是在日后继续merge以前的老版本时有区别。因为git revert是用一次逆向的commit“中和”之前的提交，因此日后合并老的branch时，导致这部分改变不会再次出现，但是git reset是之间把某些commit在某个branch上删除，因而和老的branch再次merge时，这些被回滚的commit应该还会被引入。 
3. git reset 是把HEAD向后移动了一下，而git revert是HEAD继续前进，只是新的commit的内容和要revert的内容正好相反，能够抵消要被revert的内容。

revert操作：
[https://blog.csdn.net/secretx/article/details/51461972](https://blog.csdn.net/secretx/article/details/51461972)

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

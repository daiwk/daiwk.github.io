---
layout: post
category: "nlp"
title: "自然语言处理基本常识"
tags: [nlp, natural language processing, 自然语言处理]
---

目录

<!-- TOC -->

- [**1. 初识NLP**](#1-初识nlp)
    - [**1.1 关于理解的标准(Turing test)**](#11-关于理解的标准turing-test)
    - [**1.2 自然语言处理研究的内容和面临的困难**](#12-自然语言处理研究的内容和面临的困难)
        - [**1.2.1 研究方向**](#121-研究方向)
        - [**1.2.2 层次**](#122-层次)
        - [**1.2.3 困难**](#123-困难)
    - [**1.3 《统计自然语言处理（第2版）》内容安排**](#13-统计自然语言处理第2版内容安排)

<!-- /TOC -->

# **1. 初识NLP**
 
## **1.1 关于理解的标准(Turing test)**
 
英国数学家**图灵**在1950年，提出了判断计算机是否可以被认为“能思考”的标准（图灵测试，Turing test）：如果一个计算机系统的***表现(act)、反应(react)和相互作用(interact)***都和有意识的个体一样，那么，这个计算机系统就应该被认为是有意识的。
 
## **1.2 自然语言处理研究的内容和面临的困难** 
 
### **1.2.1 研究方向**
 

1. **机器翻译**(machine translation, MT): 一种语言到另一种语言的自动翻译
2. **自动文摘**(automatic summarizing/automatic abstracting):自动归纳、提炼原文档的摘要和缩写。
3. **信息检索**(information retrieval, IR)：海量文档中检索到符合用户需求的相关文档，面对两种或两种以上语言的IR叫做跨语言信息检索(cross-language/trans-lingual information retrieval)
4. **文档分类**(document/text categorization/classification)：文档/文本自动分类。近年来，情感分类（sentiment classification）或称文本倾向性识别（text orientation identification）比较火，已经成为支撑舆情分析（public opinion analysis）的基本技术。
5. **问答系统**(question-answering system)：人机对话系统。
6. **信息过滤**(information filtering)：自动识别和过滤。
7. **信息抽取**(information extraction)：从文本中抽取特定的事件(event)或事实信息，又称事件抽取(event extraction)。
8. **文本挖掘**(text mining/data mining)：挖掘高质量信息。涉及文本分类、文本聚类(text clustering)、概念或实体抽取(concept/entity extraction)、粒度分类、情感分析、自动文摘和实体关系建模(entity relation modeling)等多种技术。 
9. **舆情分析**(public opinion analysis)：一定的社会空间内，围绕中介性社会事件的发生、发展及变化，民众对社会管理者产生和持有的社会政治态度。主要信息来源有新闻评论、论坛(bulletin board system, BBS)、聚合新闻(简易供稿，really simple syndication, RSS)、twitter等。
10. **隐喻计算**(metaphorical computation)：隐喻指的是，用乙事物或某些特征来描述甲事物的语言现象。隐喻计算就是研究自然语言语句或篇章中隐喻修辞的理解方法。
11. **文字编辑和自动校对**(automatic proffreading)：对文字拼写、用词、语法、文档格式等进行自动检查、校对和编排。
12. **作文自动评分** 
13. **光读字符识别**(optical character recognition, OCR)：印刷/手写字体识别。
14. **语音识别**(speech recognition)：语音转成文字。
15. **文语转换**(text-to-speech conversion)：书面文本自动转换成对应的语音表征，即***语音合成(speech synthesis)***
16. **说话人识别/认证/验证**(speaker recognition/identification/verification)：对说话人的言语样本做声学分析，从而推断/确定/验证说话人的身份。
 
### **1.2.2 层次**
 
1. **形态学(morphology)**：即词法，研究此内部结构，包括屈折变化和构词法两部分。
2. **语法学(syntax)**：研究句子结构成分之间的相互关系和组成句子序列的规则。关注：**为什么一句话可以这么说，也可以那么说？** 
3. **语义学(semantics)**：研究语言的意义，研究语言的各级单位（词素、词、词组、句子、句子群、整段整篇的话语和文章乃至整个著作）的意义，以及语义与语音、语法、修辞、文字、语境、哲学思想、社会环境、个人修养的关系等。关注：**这个语言单位到底说了什么？**
4. **语用学(pragmatics)**：可以集中在句子层次上进行语用研究，也可以是超出句子，对语言的实际使用情况进行调查研究，甚至与会话分析、语篇分析相结合，研究在不同上下文中的语句应用，以及上下文对语句理解所产生的影响。关注：**为什么在特定的上下文中，要说这样的话？**
 
### **1.2.3 困难**
 
1. **歧义消解(disambiguation)**：例如，如何划分分词边界。
歧义结构分析结果的数量是随介词短语的数目的增加而呈现指数上升的。歧义结构的组合数称为开塔兰数(Catalan numbers, 记作`\(C_{n}\)`)，即，如果句子中存在这样`\(n\)`个介词短语，那么：
`\[C_n = C^{n}_{2n} \dfrac{1}{n+1}\]`

2. **未知语言现象**：对于一个特定系统来说，总是有可能遇到未知词汇、未知结构等意外情况，而同时，每一种语言又随着社会的发展而动态变化着，**新的词汇（新的人名、地名、组织机构名和专用词汇）、新的词义、新的词汇用法（新词类）**，甚至**新的句子结构**都在不断地出现，尤其在口语对话或者计算机网络对话中，稀奇古怪的词汇和话语结构更是司空见惯。因此，鲁棒性(robustness)很重要。
 
## **1.3 《统计自然语言处理（第2版）》内容安排**

 ![](http://i.imgur.com/0V4CvSu.jpg)

自动生成代码sketch-code------


基础相关------------
1.  https://github.com/googlecreativelab/teachable-machine   基础

2.  深度学习中文版本阅读
    https://github.com/exacity/deeplearningbook-chinese/releases 理论基础



神经网络的可解释性-------

LIME：Local Interpretable Model-Agnostic Explanations 
更广为所知的名称是 LIME，它是一个涉及以多种方式操作数据变量以查看什么把分值提高的最多的技术。在 LIME 中，局部指的是局部保真度，即解释应该反映分类器在被预测实例「周围」的行为。这一解释是无用的除非它是可阐释的——也就是说，除非人可以理解它。Lime 可以解释任何模型而无需深入它，因此它是模型不可知论的。 DARPA 的可解释性 AI（Explainable AI）：现在，DARPA 创建了一套机器学习技术来产生更可解释的模型，同时维持一个高水平的学习表现。可解释性 AI 可以使人类用户理解和管理即将到来的 AI 伙伴，其核心优势是新技术可以潜在地回避掉对额外层的需求。另一个解释组件可以从训练神经网络关联带有隐藏层节点的语义属性——这可以促进对可解释功能的学习。

论文ICML2017---------

http://www.sohu.com/a/162831043_473283  


迁移学习-------------

情形1：目标数据集很小，目标任务与源任务相似：这种情况使用特征提取，因为目标数据集小容易造成过拟合。 
情形2：目标数据集很小，目标任务与源任务不同：这时我们微调底层网络，并移除高层网络。换句话说，我们使用较早的特征提取。 
情形3：目标数据集很大，目标任务与源任务相似：我们有了大量的数据，我们可以随机初始化参数，从头开始训练网络。然而，最好还是使用预训练的网络初始化参数并微调几层。 
情形4：目标数据集很大，目标任务与源任务不同。这时，我们微调大部分层甚至整个网络

https://arxiv.org/pdf/1707.08475 论文：使用增强学习来提高0样本的迁移学习


制定源域 和目标域
Ds   Dt

每个域中都符合马尔可夫

Ds=(Ss,As,Ts,Rs)
Dt=(St,At,Tt,Rt)
假设折扣因数是一样的 r

T 为 transition function 中文暂时翻译成 转换函数

当前行为空间被分享，且转换函数和反馈函数差不多的情况下，两个域中的状态是不一致的

1.探究卷积神经网络中的冗余性，并介绍如何通过改进网络结构减少冗余计算量；
2.通过引入动态预测方法，提升网络的推理效率； 
3.介绍一种新的剪枝方法及卷积结构，训练面向移动端的轻量卷积网络；



疗]完全不写代码： Neural Turing Machines Neural Symbolic Machines/Neural Programmer DeepCoder: Learning to Write Programs 程序员给出少量代码，用神经网络拟合得到中间缺失的函数： Programming with a Differentiable Forth Interpreter Differentiable Programs with Neural Libraries 相关工作--如何利用人工写的程序来提高神经网络『写代码』的能力： Coupling Distributed and Symbolic Execution for Natural Language Queries

https://github.com/lzrobots/DeepEmbeddingModel_ZSL


自己驾驶模拟器-----
https://github.com/18605973470/rl-with-carla/
https://github.com/kvasnyj/carla 用ros 控制carla

carla conditional imitation learning
基于autoware 项目https://github.com/xfqbuaa/PIX-Hackathon-Autoware
https://github.com/facebookresearch/Detectron
deepdrive.io 又一个模拟器

自定驾驶感知之对象侦测和图像分割--------
https://arxiv.org/pdf/1612.03144.pdf Feature Pyramid Networks for Object Detection
https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/voc-fcn-alexnet/train.prototxt
https://github.com/jinfagang/keras_frcnn
http://blog.csdn.net/happyer88/article/details/47205839 fcn 理解

http://blog.csdn.net/zhangjunhit/article/details/65629974  目标检测--Feature Pyramid Networks for Object Detection
Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks
https://arxiv.org/pdf/1506.01497.pdf 


Retinanet-------------
https://github.com/tensorflow/tpu/blob/master/models/official/retinanet/retinanet_architecture.py
http://blog.csdn.net/qq_34564947/article/details/77200104
https://arxiv.org/pdf/1708.02002.pdf

感知融合-----------

https://github.com/martinruenz/co-fusion  多目标融合

智能模拟-----------

https://pathak22.github.io/noreward-rl/  好奇心
https://github.com/MultiAgentLearning/playground 基于炸弹人游戏的多代理的ai研究环境

https://blog.csdn.net/u013236946/article/details/73243310 深度强化学习——连续动作控制DDPG、NAF
https://github.com/tensorflow/tfjs-core 深度强化学习——连续动作控制DDPG、NAF

https://github.com/yanpanlau/DDPG-Keras-Torcs

https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/6-3-A1-A3C/


https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/9_Deep_Deterministic_Policy_Gradient_DDPG/DDPG.py
https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/9_Deep_Deterministic_Policy_Gradient_DDPG/DDPG_update.py

https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/tree/master/experiments/Robot_arm

智能问答------

https://github.com/hankcs/multi-criteria-cws  简单有效的多标准中文分词

工具-------

http://sklearn.apachecn.org  sklearn 中文版本

模型设计的一般流程--------

研究人员决定尝试一个新的图像分类架构。 她从先前的项目里复制粘贴一些代码来处理她要使用的数据集的输入。 数据集在网络中的其中一个她的文件夹中，这可能是从 ImageNet 下载的一个数据集，但不确定具体是哪一个。也许不知道什么时候会有人把其中那些非 JPEG 格式的图像删掉，或者做一些其它的小改动，但是这些操作都没有历史记录 她会尝试很多种稍有区别的想法，改 bug 和稍微调整算法。这些在她自己的本地电脑上完成，当她想训练这些模型的时候，她就直接把一大堆源代码复制到 GPU 集群上。 她会把训练过程执行很多遍，通常程序在跑的时候，要花几天或者几个星期去完成，通常在这期间她还会在自己的本地电脑上修改一些代码。 可能在集群上跑到快结束的时候出现了 bug，那么在跑的过程，她需要修改一个文件的代码，然后把这个改动拷贝到所有的机器上，然后继续运行程序。 她可能从一个跑出来的程序里拿出部分训练到的权重，然后在这个新的起点上，运行不同的代码。 她会记录所有运行过程得到的权重和对应的评分，然后当她没有时间做更多实验的时候就从里面挑出一组作为最终的模型。这些权重可能来自任何一个跑出来的结果，甚至来自于和她现在手上跑着的代码非常不同的代码。 她可能会把最终的程序代码在源代码控制中做个登记，不过这是在她的个人文件夹上。 她发表她的结果，附上代码和训练权重























 
 

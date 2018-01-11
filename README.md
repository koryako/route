课程大纲第一课：SLAM概论和架构
1.从机器人的体系结构讨论SLAM的提出和发展
2.滤波器是什么，谁真正的推动了SLAM？
3.SLAM的新突破-图优化
4.SLAM的完整知识体系结构介绍，基于Linux和ROS进行SLAM的进行本课程学习
5.ROS基础：RGB-D点云示例

第二课：SLAM基本理论一：坐标系、刚体运动和李群
1.SLAM的数学表达
2.欧式坐标系和刚体姿态表示
3.李群和李代数
4.实例：Eigen和Sophus在滤波器上的应用

第三课：SLAM基本理论二：从贝叶斯开始学滤波器

1.随机状态和估计
2.卡尔曼滤波器
3.扩展卡尔曼滤波器和SLAM
4.粒子滤波器和SLAM

5.实例：基于卡尔曼滤波器的SLAM实例

第四课：SLAM基本理论三：图优化

1.从滤波器的痛来谈图优化
2.CovisibilityGraph和最小二乘

3.浅谈Marginlization
4.实例：G2O图优化实战

第五课：SLAM的传感器
1.SLAM传感器综述
2.视觉类传感器（单目、双目和RGBD相机）a.相机模型和标定b.特征提取和匹配
3.主动类传感器--激光a.激光模型和不同激光特性b.激光特征和匹配
4.实例：a.特征提取和立体视觉的深度结算；b.激光数据的基本处理

第六课：视觉里程计和回路检测
1.视觉里程计的综述
2.基于特征法的视觉里程计：PNP
3.基于直接法的视觉里程计：PhotometricError
4.基于立体视觉法的:ICP
5.基于词袋模型的回路检测
6.实例：a.PNP位姿估计b.直接法位姿估计c.回路检测

第七课：激光里程计和回路检测
1.激光里程计简介
2.激光里程计算法LOAM和VLOAM简单介绍
3.激光回路检测的特殊性和主要难点
4.伯克利的BLAM和谷歌Cartographer中回路检测的核心思路介绍
5.实例:LOAM,Cartographer测试

第八课：地图以及无人驾驶系统
1.SLAM中的不同地图系统介绍
2.高精度地图介绍
3.语义地图介绍
4.拓扑地图介绍
5.实例：粒子滤波定位实现

第九课：视觉和无人机、室内辅助导航和AR/VR

1.视觉SLAM的整体重述和实战

2.SLAM、无人机和状态机

3.GoogleTango和盲人导航
4.SLAM的小刺激：AR/VR
5.实例：视觉SLAM的AR实例

第十课：深度学习和SLAM

1.SLAM的过去、现在和未来
2.长航程SLAM的可能性
3.单目深度估计和分割和场景语义
4.动态避障
5.新的特征表达
6.课程总结 







1.
     了解 https://github.com/googlecreativelab/quickdraw-dataset  图标数据集，看看可以做成什么应用

     https://github.com/LMescheder/AdversarialVariationalBayes  贝叶斯对抗网络可以做什么

2
     使用 unity 创建环境  使用api 创建agent   https://github.com/unity-Technologies/ml-Agents

     https://blogs.unity3d.com/cn/2017/09/19/introducing-unity-machine-learning-agents/

3
     python 的外部api 控制游戏框架
     https://github.com/SerpentAI/SerpentAI
     https://github.com/SerpentAI/SerpentAI/wiki

4    深度学习中文版本阅读
     https://github.com/exacity/deeplearningbook-chinese/releases

5    
     大数据自动驾驶 
     End-to-end Learning of Driving Models from Large-scale Video Datasets
     https://arxiv.org/abs/1612.01079

6    https://github.com/openai/multiagent-competition

     
7    Mask R-CNN
    https://arxiv.org/abs/1703.06870

8    vispy.org

9    https://challenger.ai


10
[1] Large-ScaleEvolutionofImageClassifiers,
[2]NeuralArchitectureSearchwithReinforcementLearning

11   注意力

     论文《A Differentiable Transition Between Additive and Multiplicative Neurons》对这一概念进行了探索，
     参阅：https://arxiv.org/abs/1604.03736。

     另外，《深度|深度学习能力的拓展，GoogleBrain讲解注意力模型和增强RNN》这篇文章也对软注意机制进行了很好的概述。 

     软注意最简单的形式在图像方面和向量值特征方面并无不同，还是和上面的（1）式一样。

     论文《Show,AttendandTell:NeuralImageCaptionGenerationwithVisualAttention》是最早使用这种类型的注意的研究之一：
     
     https://arxiv.org/abs/1502.03044 


     下面这两个机制解决了这个问题，它们分别是由DRAW（https://arxiv.org/abs/1502.04623）和
     SpatialTransformerNetworks（https://arxiv.org/abs/1506.02025）这两项研究引入的。
     
     它们也可以重新调整输入的大小，从而进一步提升性能。 


     最近一篇关于使用带有注意机制的RNN进行生物启发式目标跟踪的论文HART中就使用了这种机制，参阅：https://arxiv.org/abs/1706.09262。


12
     对星际二agent的同学，欢迎大家围观与star：https://github.com/wwxFromTju/sc2-101-zh/tree/master/sc2-code/first_agent

     https://zhuanlan.zhihu.com/p/29222384知乎专栏的阅读效果更好，但是github会持续更新

13
https://github.com/loliverhennigh/Early-Visual-Concept-Learning-Recreation-of-Some-Results

https://github.com/crcrpar/chainer-VAE
论文：https://arxiv.org/abs/1606.05579

http://openreview.net/forum?id=Sy2fzU9gl 

14
https://github.com/googlecreativelab/teachable-machine 

15  量子api
OpenFermion，本文的主角https://github.com/quantumlib/OpenFermionOpenFermion
论文：OpenFermion:TheElectronicStructurePackageforQuantumComputers
https://arxiv.org/abs/1710.07629OpenFermion-Psi4
https://github.com/quantumlib/OpenFermion-Psi4Psi4
https://github.com/psi4/psi4OpenFermion-PySCF
https://github.com/quantumlib/OpenFermion-PySCFPySCF
https://github.com/sunqm/pyscfOpenFermion-ProjectQ
https://github.com/quantumlib/OpenFermion-ProjectQProjectQ
https://github.com/ProjectQ-Framework/ProjectQForest-OpenFermion
https://github.com/rigetticomputing/forestopenfermionForest
https://www.rigetti.com/forest


16
用于训练GAN的数据集：Caltech-UCSD-200-2011是一个具有200种鸟类照片、总数为11,788的图像数据集。

Oxford-102花数据集由102个花的类别组成，每个类别包含40到258张图片不等。

17
 论文地址：https://arxiv.org/pdf/1705.07962.pdf
 
 Github项目地址：https://github.com/tonybeltramelli/pix2code
 
 申请试用地址：https://uizard.io/?email_field=mmill06%40gmail.com 
 
 18
本文参考了：HighQuality3DObjectReconstructionfromaSingleColorImage

相关论文连接：1、HierarchicalSurfacePredictionfor3DObjectReconstruction（ChristianHäne等）
2、3D-R2N2:AUnifiedApproachforSingleandMulti-view3DObjectReconstruction（Choy等）
3、LearningaPredictableandGenerativeVectorRepresentationforObjects（Girdhar等）
4、ShapeNet:AnInformation-Rich3DModelRepository（关于ShapeNet数据集的论文） 

19  动作视频数据集
    1.数据集地址：https://research.google.com/ava/

      论文：https://arxiv.org/abs/1705.08421 
    许多基准数据集，
    UCF101、
    activitynet
    和DeepMind的Kinetics，
    都是采用图像分类的标记方案，

    http://www.cvlibs.net/datasets/kitti/index.php

    http://cs.stanford.edu/people/teichman/stc/ 



20


21

DeepMind发布的WaveNet的表现相当不错，而且现在还有百度的DeepVoice3，以及最近谷歌开发的Tacotron2： 

34-layer1DResNet诊断心律失常的模型 

AutoML技术，以及新智元分别报道过的“将好奇心加入AI”和“让AI自动分享学到的经验”，将成为狭义AI迈向AGI和超级智能的基础。 


21
FusionofLiDAR3DPointsCloudwith2DDigitalCameraImage

22
ZhuSuan: A Library for Bayesian Deep Learning
https://arxiv.org/abs/1709.05870 

[1]BayesianMethodLecture,UTDallas.http://www.utdallas.edu/~nrr150130/cs7301/2016fa/lects/Lecture_14_Bayes.pdf
[2]MLE,MAP,BayesclassificationLecture,CMU.http://www.cs.cmu.edu/~aarti/Class/10701_Spring14/slides/MLE_MAP_Part1.pdf

附录为什么说频率学派求硬币概率的算法本质是在优化NLL？因为抛硬币可以表示为参数为θ的Bernoulli分布，即：其中xi=1表示第i次抛出正面。那么，求导数并使其等于零，得到﻿即，也就是出现正面的次数除以总共的抛掷次数。 

23

https://www.wired.com/story/this-ai-fortified-bot-will-build-the-first-homes-for-humans-on-mars/?mbid=social_twitter_onsiteshare

http://www.dlr.de/rmc/rm/en/desktopdefault.aspx/tabid-11427/#gallery/29202 



24

这两个是pytorch版本的；官方https://github.com/LantaoYu/SeqGAN；


seqgan的升级版leakgan

25

https://github.com/googlecreativelab/teachable-machine 


http://www.cvlibs.net/datasets/kitti/index.php

http://cs.stanford.edu/people/teichman/stc/

论文：https://www.autodeskresearch.com/sites/default/files/paper.pdf 

GitHub地址：https://github.com/apple/darwin-xnu 
https://www.cc.gatech.edu/~riedl/pubs/ijcai17.pdf 

https://github.com/baldassarreFe/deep-koalarization。 

https://devblogs.nvidia.com/parallelforall/photo-editing-generative-adversarial-networks-1/ 






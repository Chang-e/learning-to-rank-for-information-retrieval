# 第八章 半监督排序

**摘要：**在本章中，我们介绍了用于排序的半监督学习。 该主题的动机来自于这样一个事实，我们总是可以以低成本收集大量未贴标签的文档或查询。 如果可以在学习排序过程中利用这些未标记的数据，那将非常有帮助。 在本章中，我们主要回顾用于此任务的归纳法和转化法，并讨论如何通过考虑排序的独特属性来改进这些方法。

到目前为止，在本书的前几章中，我们主要讨论了排序中的监督学习。 但是，就像分类中的情况一样，有时未标记的数据将帮助我们减少所需标记数据的数量。 在半监督排名方面已有一些初步尝试[1、2]。

## 8.1 归纳法

[1]采用了归纳法。 更具体地，根据标签文档的基础标签在特征空间中的相互相似性，将其传播到未标签文档。 相同的技术已广泛用于半监督分类[4、7、8]。

为了利用来自未标记数据集的信息，假定与标记文档相似的未标记文档应具有与该标记文档相似的标签。 首先选择与标签文档$x$最相似的未标签文档，然后为它们分配相应的相关性判断$y$。 为了便于讨论，我们将未标记的文档称为自动标记的文档，而原始标记的文档则称为人工标记的文档。

标签传播后，一种简单的方法是将这些自动标记的文档添加到原始训练集中，然后像在监督案例中那样，例如使用RankBoost [3]，学习排序功能。但是，该训练方案具有以下缺点。由于自动标记的文档具有容易出错的标签，因此排序效果将高度取决于训练方案对嘈杂的标签的鲁棒性。因此，与其将人工标记的文档和自动标记的文档混合起来，不如分别处理这两种类型的文档。也就是说，当使用RankBoost时，仅在同一类型的文档中构造文档对，并且为两种类型的文档对维护单独的分布。这样，两种类型的数据将分别对总体损失函数做出贡献（参数$λ$用于权衡对应于不同类型文档的损失）。已经证明，通过这种处理，训练过程仍然可以收敛，并且可以继承RankBoost的一些原始的好属性。

所提出的方法已经在几个二分排名任务上进行了测试，以AUC（ROC曲线下的面积）作为评估指标。 实验结果表明，所提出的方法可以提高二分法排序的准确性，即使只有少量标记数据，也可以很好地利用未标记数据来实现良好的性能。

## 8.2 转化法

在[2]中，采用了一种转化法。 关键思想是使用未标记的测试数据自动得出更好的特征，以提高模型训练的有效性。 特别是，一种无监督的学习方法（特别是[2]中的内核PCA）被用于发现检索到的测试文档的每个列表中的显着模式。 总共使用了四个不同的内核：多项式内核，RBF内核，扩散内核和线性内核（在这种特定情况下，内核PCA完全成为PCA）。 然后将训练数据投影到这些模式的方向上，并将所得的数值添加为新特征。 这种方法的主要假设是，这种新的训练集（在投影之后）可以更好地表征测试数据，因此在学习等级功能时应优于原始训练集。 然后将RankBoost [3]作为示例算法来证明这种转化方法的有效性。

在文献[2]中对LETOR基准数据集进行了广泛的实验（参见第10章），以测试所提出的转导方法的有效性。 实验结果表明，以这种方式使用未标记的数据可以提高排序性能。 同时，已经对与该半监督学习过程相关的一些问题进行了详细的分析，例如，是否可以解释内核PCA特征，非线性内核PCA是否有帮助，性能在查询中的变化情况以及什么？ 计算复杂度是多少？总体结论如下。

- 内核PCA特征通常很难解释，并且大多数内核PCA特征与原始特征几乎没有关联。
- 在大多数情况下，非线性很重要，但是不应指望非线性内核总是优于线性内核。 最好的策略是采用多个内核。
- 提出的转导方法并未在所有查询中都进行改进。 与改进查询相比，改进查询所占的比例更大。
- 转化法需要在线计算。 对于正在调查的数据集，每个查询的总时间约为数百秒。 因此，希望可以使用更好的代码优化或新颖的分布式算法来使该方法切实可行。

## 8.3 小结

我们可以看到，以上工作从半监督分类中借鉴了一些概念和算法。尽管已经获得了良好的排名表现，但这样做的有效性可能还需要进一步的证明。例如，由于相似性对于许多分类算法都是必不可少的（即“相似文档应具有相同的类别标签”），因此通过相似文档传播标签看起来非常自然且合理。但是，在排序中，相似性并不发挥相同的核心作用。似乎偏好比相似性更根本。然后，问题是进行基于相似度的标签传播以进行半监督排名是否仍然自然合理。

此外，在分类中，如果没有分类标签，则对条件概率$p(y | x)$一无所知。但是，在排序中，即使我们没有基准标签，我们仍然有几个非常强大的排名，例如BM25 [6]和LMIR [5]，这可以让我们相对合理地猜测应该使用哪个文档排序更高。换句话说，我们对未标记的数据有一些了解。如果我们可以将此类知识纳入半监督排序过程，那么我们就有机会做得更好。

## 参考文献

1. Amini, M.R., Truong, T.V., Goutte, C.: A boosting algorithm for learning bipartite ranking functions with partially labeled data. In: Proceedings of the 31st Annual International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR 2008), pp. 99–106 (2008)
2. Duh, K., Kirchhoff, K.: Learning to rank with partially-labeled data. In: Proceedings of the 31st Annual International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR 2008), pp. 251–258 (2008)
3. Freund, Y., Iyer, R., Schapire, R., Singer, Y.: An efficient boosting algorithm for combining preferences. Journal of Machine Learning Research **4**, 933–969 (2003)
4. Niu,Z.Y.,Ji,D.H.,Tan,C.L.:Word sense disambiguation using label propagation based semi supervised learning. In: Proceedings of the 403rd Annual Meeting of the Association for Com- putational Linguistics (ACL 2005), pp. 395–402 (2005)
5. Ponte,J.M.,Croft,W.B.:A language modeling approach to information retrieval.In:Proceed-ings of the 21st Annual International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR 1998), pp. 275–281 (1998)
6. Robertson,S.E.:Overview of the okapi projects.Journal of Documentation 53(1),3–7(1997)
7. Tong, W., Jin, R.: Semi-supervised learning by mixed label propagation. In: Proceedings of the 22nd AAAI Conference on Artificial Intelligence (AAAI 2007), pp. 651–656 (2007)
8.  Xiaojin Zhu, Z.G.: Learning from labeled and unlabeled data with label propagation. Ph.D. thesis, Carnegie Mellon University (2002)


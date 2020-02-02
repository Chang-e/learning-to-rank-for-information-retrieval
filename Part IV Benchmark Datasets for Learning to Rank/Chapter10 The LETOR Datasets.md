# 第十章 LETOR数据集

**摘要：**在本章，我们将介绍LETOR基准数据集，包括以下几个方面：文档语料库（以及查询集），文档采样，特征提取，元信息，交叉验证和支持的主要排序任务。

## 10.1 概述

众所周知，具有标准功能和评估措施的基准数据集对机器学习的研究非常有帮助。 例如，有基准数据集，例如Reuters（http://www.daviddlewis.com/resources/testcollections/reuters21578/.）和RCV-1（http://jmlr.csail.mit.edu/papers/volume5/lewis04a/lyrl2004_rcv1v2_README.htm.）用于文本分类，而UCI（http://archive.ics.uci.edu/ml/.）用于常规机器学习。 但是，直到2007年初发布了LETOR数据集[8]为止，还没有用于排名的基准数据集。近年来，LETOR数据集已广泛用于学习等级论文的实验中，并极大地帮助了他们。 推进关于学习排名的研究。 截止本书编写之日，LETOR已经发布了多个版本。 在本章中，我们将主要介绍两个最常用的版本，即LETOR 3.0和4.0。 特别是，我们将描述这些数据集的详细信息，包括文档语料库，文档采样，特征提取，元信息和学习任务。

## 10.2 文档语料

LETOR数据集中使用了三个文档语料库和九个查询集。 前两个文档语料库在LETOR 3.0中使用，而第三个文档语料库在LETOR 4.0中使用。

### 10.2.1 “Gov”语料库和六个查询集

在TREC 2003和2004中，组织了一个用于Web信息检索的特殊渠道，称为Web track（http://trec.nist.gov/tracks.html.）。 该渠道使用了“ Gov”语料库，该语料库基于2002年1月对.gov域。 该语料库总共有1,053,110个html文档。

Web轨道中有三个搜索任务：主题提炼（TD），主页查找（HP）和命名页面查找（NP）。 主题提炼旨在找到主要致力于该主题的优质网站的入口点列表。 查找主页旨在返回查询的主页。 查找命名页面旨在返回名称与查询完全相同的页面。 一般而言，主页查找和命名页面查找只有一个答案。 这三个任务中的查询数量如表10.1所示。 为了便于参考，我们将这两年中的查询集分别表示为TD2003，TD2004，HP2003，HP2004，NP2003和NP2004。


表10.1 TREC网页轨道的query个数
| 任务 | TREC2003 | TREC2004 |
| ------ | ------ | ------ |
| 话题提炼 | 50 | 75 |
| 主页查找 | 150 | 75 |
| 命名页面查找 | 150 | 75 |

由于语料库规模较大，因此检查每个文档并判断其与给定查询的相关性是不可行的。 TREC中的做法如下。 给定一个查询，仅选择在参与者提交的运行中排名最高的一些“可能”相关文档进行标记。 给定一个查询，要求评估人员标记这些可能相关的文件是否真正相关。 在评估过程中，所有其他文件，包括那些未经评估人员检查但未标记为相对的文件以及在提交的运行中根本没有排名最高的文件，都被认为是无关的[2]。

许多研究论文[11、13、19、20]已经使用“ Gov”语料库上的三个任务作为其实验平台。

### 10.2.2 OHSUMED 语料库

OHSUMED语料库[5]是MEDLINE的子集，MEDLINE是有关医学出版物的数据库。 它包含1987年至1991年期间270种医学期刊的348,566条记录（超过7,000,000条）。记录的字段包括标题，摘要，MeSH索引词，作者，来源和出版物类型。

在先前的许多工作中[11，19]，已经使用了OHSUMED语料库上的106个查询的查询集，每个查询都描述了医疗搜索需求（与患者信息和主题信息相关联）。 文件与查询的相关程度由评估人员从三个层次进行判断：绝对相关，部分相关和不相关。 有相关判断的总共有16140个查询文档对。

### 10.2.3 “Gov2” 语料库和两个查询集

Million Query（MQ）方法在TREC 2007中首次运行，然后在随后的几年中成为规范。 MQ的设计有两个目的。 首先，它是对大量文档的即时检索的探索。 其次，它研究了系统评估的问题，特别是使用许多浅层的判断还是少量的全面判断方面的探索。

MQ方法使用所谓的“ terabyte”或“ Gov2”语料库作为其文档集合。 该语料库是2004年初从.gov域的网站中爬取的Web数据的集合。该集合包括426 GB的大约25,000,000个文档。

2007年的MQ方法中大约有1700个带有标签文档的查询（简称为MQ2007），而2008年的MQ方法中大约有800个查询（称为MQ2008）。 判断分为三个级别，即高度相关，相关和不相关。

## 10.3 文档采样

由于与选择用于标记的文档类似，不可能提取语料库中所有文档的特征向量。 合理的策略是对一些“可能”相关的文档进行采样，然后为相应的查询-文档对提取特征向量。

对于TD2003，TD2004，NP2003，NP2004，HP2003和HP2004，按照[9]和[12]中的建议，以以下方式采样文档。 首先，使用BM25模型对每个查询排序所有文档，然后选择每个查询的前1000个文档进行特征提取。 请注意，这种抽样策略是为了简化实验研究，但这绝不是说学习排序只能适用于这种重新排名的情况。

与上述将未判断的文档视为不相关的任务不同，在OHSUMED，MQ2007和MQ2008中，判断明确包含“不相关”的类别，并且在评估中忽略了未判断的文档。 相应地，在LETOR中，仅将已判断的文档用于特征提取，而这些语料库将忽略所有未判断的文档。

最近有一些关于学习排序的文档抽样策略的讨论，例如[1]。 不同的采样策略可能会导致不同的训练效果，但是，目前这些策略尚未应用于LETOR数据集中。

## 10.4 特征提取

本节中，我们介绍LETOR中文档的特征表示。 在特征提取过程中使用以下原理。

1. 在信息检索中涵盖尽可能多的经典特征。
2. 为了重现SIGIR最新论文中建议的尽可能多的函数，他们使用OHSUMED，“ Gov”或“ Gov2”语料进行实验。
3. 符合原始文件中的设置。

对于Gov语料库，每个查询文档对都提取了64个特征，如表10.2所示。 其中一些特征既依赖于查询又依赖于文档，某些仅依赖于文档，而另一些仅依赖于查询。 在表中，$q$表示查询，其中包含项$t_1,...,t_m$，$TF(t_i,d)$表示文档$d$中查询词$t_i$的出现次数。 请注意，如果从流（例如标题或URL）中提取特征，则$TF(t_i,d)$表示流中$t_i$的出现次数。

从上表中，我们可以找到许多经典的信息检索功能，例如术语频率和BM25 [16]。 同时，根据最新的SIGIR论文还提取了许多特征。 例如，主题页面排名和主题HITS是根据[10]计算的； 根据[17]和[13]计算基于站点地图和基于超链接的分数/特征传播，根据[20]计算HostRank，并根据[6]生成提取的标题。 有关这些功能的更多详细信息，请访问LETOR网站http://research.microsoft.com/~LETOR/。

对于OHSUMED语料库，总共提取了45个特征，如表10.3所示。 在表中，$| C |$ 表示语料库中文档的总数。 有关这些特征的更多详细信息，请访问LETOR网站。

对于“ Gov2”语料库，提取了46个特征，如表10.4所示。 同样，可以在LETOR网站上找到有关这些特征的更多详细信息。

表10.2 Gov语料库特征
| ID   | 特征描述                                                |
| ---- | ------------------------------------------------------- |
| 1    | 主体的$\sum_{t_i\in q \cap d}TF(t_i,d)$                 |
| 2    | 锚点的$\sum_{t_i\in q \cap d}TF(t_i,d)$                 |
| 3    | 标题的$\sum_{t_i\in q \cap d}TF(t_i,d)$                 |
| 4    | URL的$\sum_{t_i\in q \cap d}TF(t_i,d)$                  |
| 5    | 全文档的$\sum_{t_i\in q \cap d}TF(t_i,d)$               |
| 6    | 主体的$\sum_{t_i\in q}IDF(t_i)$                         |
| 7    | 锚点的$\sum_{t_i\in q}IDF(t_i)$                         |
| 8    | 标题的$\sum_{t_i\in q}IDF(t_i)$                         |
| 9    | URL的$\sum_{t_i\in q}IDF(t_i)$                          |
| 10   | 全文档的$\sum_{t_i\in q}IDF(t_i)$                       |
| 11   | 主体的$\sum_{t_i\in q \cap d}TF(t_i,d)\cdot IDF(t_i)$   |
| 12   | 锚点的$\sum_{t_i\in q \cap d}TF(t_i,d)\cdot IDF(t_i)$   |
| 13   | 标题的$\sum_{t_i\in q \cap d}TF(t_i,d)\cdot IDF(t_i)$   |
| 14   | URL的$\sum_{t_i\in q \cap d}TF(t_i,d)\cdot IDF(t_i)$    |
| 15   | 全文档的$\sum_{t_i\in q \cap d}TF(t_i,d)\cdot IDF(t_i)$ |
| 16   | 主体的$LEN(d)$                                          |
| 17   | 锚点的$LEN(d)$                                          |
| 18   | 标题的$LEN(d)$                                          |
| 19   | URL的$LEN(d)$                                           |
| 20   | 全文的$LEN(d)$                                          |
| 21   | 主体的BM25                                              |
| 22   | 锚点的BM25                                              |
| 23   | 标题的BM25                                              |
| 24   | URL的BM25                                               |
| 25   | 全文档的BM25                                            |
| 26   | 主体的LMIR.ABS                                          |
| 27   | 锚点的LMIR.ABS                                          |
| 28   | 标题的LMIR.ABS                                          |
| 29   | URL的LMIR.ABS                                           |
| 30   | 全文档的LMIR.ABS                                        |
| 31   | 主体的LMIR.DIR                                          |
| 32   | 锚点的LMIR.DIR                                          |
| 33   | 标题的LMIR.DIR                                          |
| 34   | URL的LMIR.DIR                                           |
| 35   | 全文档的LMIR.DIR                                        |
| 36   | 主体的LMIR.JM                                           |
| 37   | 锚点的LMIR.JM                                           |
| 38   | 标题的LMIR.JM                                           |
| 39   | URL的LMIR.JM                                            |
| 40   | 全文档的LMIR.JM                                         |
| 41   | 基于站点网络的术语传播                                  |
| 42   | 基于站点网络的得分传播                                  |
| 43   | 基于超链接的分数传播：加权链接                          |
| 44   | 基于超链接的分数传播：加权出链                          |
| 45   | 基于超链接的分数传播：标准化出链                        |
| 46   | 基于超链接的特征传播：加权链接                          |
| 47   | 基于超链接的特征传播：加权出链                          |
| 48   | 基于超链接的特征传播：标准化出链                        |
| 49   | HITS权限                                                |
| 50   | HITS中心                                                |
| 51   | PageRank                                                |
| 52   | HostRank                                                |
| 53   | 主题PageRank                                            |
| 54   | 主题HITS权限                                            |
| 55   | 主题HITS中心                                            |
| 56   | 内链个数                                                |
| 57   | 外链个数                                                |
| 58   | URL中的斜线数                                           |
| 59   | URL长度                                                 |
| 60   | 子页面个数                                              |
| 61   | 所提取标题的BM25                                        |
| 62   | 所提取标题的LMIR.ABS                                    |
| 63   | 所提取标题的LMIR.DIR                                    |
| 64   | 所提取标题的LMIR.JM                                     |

表10.3  OHSUMED语料库特征
| ID   | 特征描述                                                     |
| ---- | ------------------------------------------------------------ |
| 1    | 标题的$\sum_{t_i\in q \cap d}TF(t_i,d)$                      |
| 2    | 标题的$\sum_{t_i\in q \cap d} log(TF(t_i,d)+1)$              |
| 3    | 标题的$\sum_{t_i\in q \cap d}\frac{TF(t_i,d)}{LEN(d)}$       |
| 4    | 标题的$\sum_{t_i\in q \cap d}log(\frac{TF(t_i,d)}{LEN(d)}+1)$ |
| 5    | 标题的$\sum_{t_i\in q} log(|C|\cdot IDF(t_i))$               |
| 6    | 标题的$\sum_{t_i\in q} log(log(|C|\cdot IDF(t_i)))$          |
| 7    | 标题的$\sum_{t_i\in q} log(\frac{|C|}{TF(t_i,C)}+1)$         |
| 8    | 标题的$\sum_{t_i\in q \cap d}log(\frac{TF(t_i,d)}{LEN(d)}\cdot log(|C|\cdot IDF(t_i))+1)$ |
| 9    | 标题的$\sum_{t_i\in q \cap d}TF(t_i,d\cdot log(|C|\cdot IDF(t_i)))$ |
| 10   | 标题的$\sum_{t_i\in q \cap d}log(\frac{TF(t_i,d)}{LEN(d)}\cdot \frac{|C|}{TF(t_i,C)}+1)$ |
| 11   | 标题的BM25                                                   |
| 12   | 标题的$log(BM25)$                                            |
| 13   | 标题的LMIR.DIR                                               |
| 14   | 标题的LMIR.JM                                                |
| 15   | 标题的LMIR.ABS                                               |
| 16   | 摘要的$\sum_{t_i\in q \cap d}TF(t_i,d)$                      |
| 17   | 摘要的$\sum_{t_i\in q \cap d} log(TF(t_i,d)+1)$              |
| 18   | 摘要的$\sum_{t_i\in q \cap d}\frac{TF(t_i,d)}{LEN(d)}$       |
| 19   | 摘要的$\sum_{t_i\in q \cap d}log(\frac{TF(t_i,d)}{LEN(d)}+1)$ |
| 20   | 摘要的$\sum_{t_i\in q} log(|C|\cdot IDF(t_i))$               |
| 21   | 摘要的$\sum_{t_i\in q} log(log(|C|\cdot IDF(t_i)))$          |
| 22   | 摘要的$\sum_{t_i\in q} log(\frac{|C|}{TF(t_i,C)}+1)$         |
| 23   | 摘要的$\sum_{t_i\in q \cap d}log(\frac{TF(t_i,d)}{LEN(d)}\cdot log(|C|\cdot IDF(t_i))+1)$ |
| 24   | 摘要的$\sum_{t_i\in q \cap d}TF(t_i,d\cdot log(|C|\cdot IDF(t_i)))$ |
| 25   | 摘要的$\sum_{t_i\in q \cap d}log(\frac{TF(t_i,d)}{LEN(d)}\cdot \frac{|C|}{TF(t_i,C)}+1)$ |
| 26   | 摘要的BM25                                                   |
| 27   | 摘要的$log(BM25)$                                            |
| 28   | 摘要的LMIR.DIR                                               |
| 29   | 摘要的LMIR.JM                                                |
| 30   | 摘要的LMIR.ABS                                               |
| 31   | 摘要+标题的$\sum_{t_i\in q \cap d}TF(t_i,d)$                 |
| 32   | 摘要+标题的$\sum_{t_i\in q \cap d} log(TF(t_i,d)+1)$         |
| 33   | 摘要+标题的$\sum_{t_i\in q \cap d}\frac{TF(t_i,d)}{LEN(d)}$  |
| 34   | 摘要+标题的$\sum_{t_i\in q \cap d}log(\frac{TF(t_i,d)}{LEN(d)}+1)$ |
| 35   | 摘要+标题的$\sum_{t_i\in q} log(|C|\cdot IDF(t_i))$          |
| 36   | 摘要+标题的$\sum_{t_i\in q} log(log(|C|\cdot IDF(t_i)))$     |
| 37   | 摘要+标题的$\sum_{t_i\in q} log(\frac{|C|}{TF(t_i,C)}+1)$    |
| 38   | 摘要+标题的$\sum_{t_i\in q \cap d}log(\frac{TF(t_i,d)}{LEN(d)}\cdot log(|C|\cdot IDF(t_i))+1)$ |
| 39   | 摘要+标题的$\sum_{t_i\in q \cap d}TF(t_i,d\cdot log(|C|\cdot IDF(t_i)))$ |
| 40   | 摘要+标题的$\sum_{t_i\in q \cap d}log(\frac{TF(t_i,d)}{LEN(d)}\cdot \frac{|C|}{TF(t_i,C)}+1)$ |
| 41   | 摘要+标题的BM25                                              |
| 42   | 摘要+标题的$log(BM25)$                                       |
| 43   | 摘要+标题的LMIR.DIR                                          |
| 44   | 摘要+标题的LMIR.JM                                           |
| 45   | 摘要+标题的LMIR.ABS                                          |

表10.4 Gov2语料库的特征
表10.2 Gov语料库特征
| ID   | 特征描述                                                |
| ---- | ------------------------------------------------------- |
| 1    | 主体的$\sum_{t_i\in q \cap d}TF(t_i,d)$                 |
| 2    | 锚点的$\sum_{t_i\in q \cap d}TF(t_i,d)$                 |
| 3    | 标题的$\sum_{t_i\in q \cap d}TF(t_i,d)$                 |
| 4    | URL的$\sum_{t_i\in q \cap d}TF(t_i,d)$                  |
| 5    | 全文档的$\sum_{t_i\in q \cap d}TF(t_i,d)$               |
| 6    | 主体的$\sum_{t_i\in q}IDF(t_i)$                         |
| 7    | 锚点的$\sum_{t_i\in q}IDF(t_i)$                         |
| 8    | 标题的$\sum_{t_i\in q}IDF(t_i)$                         |
| 9    | URL的$\sum_{t_i\in q}IDF(t_i)$                          |
| 10   | 全文档的$\sum_{t_i\in q}IDF(t_i)$                       |
| 11   | 主体的$\sum_{t_i\in q \cap d}TF(t_i,d)\cdot IDF(t_i)$   |
| 12   | 锚点的$\sum_{t_i\in q \cap d}TF(t_i,d)\cdot IDF(t_i)$   |
| 13   | 标题的$\sum_{t_i\in q \cap d}TF(t_i,d)\cdot IDF(t_i)$   |
| 14   | URL的$\sum_{t_i\in q \cap d}TF(t_i,d)\cdot IDF(t_i)$    |
| 15   | 全文档的$\sum_{t_i\in q \cap d}TF(t_i,d)\cdot IDF(t_i)$ |
| 16   | 主体的$LEN(d)$                                          |
| 17   | 锚点的$LEN(d)$                                          |
| 18   | 标题的$LEN(d)$                                          |
| 19   | URL的$LEN(d)$                                           |
| 20   | 全文的$LEN(d)$                                          |
| 21   | 主体的BM25                                              |
| 22   | 锚点的BM25                                              |
| 23   | 标题的BM25                                              |
| 24   | URL的BM25                                               |
| 25   | 全文档的BM25                                            |
| 26   | 主体的LMIR.ABS                                          |
| 27   | 锚点的LMIR.ABS                                          |
| 28   | 标题的LMIR.ABS                                          |
| 29   | URL的LMIR.ABS                                           |
| 30   | 全文档的LMIR.ABS                                        |
| 31   | 主体的LMIR.DIR                                          |
| 32   | 锚点的LMIR.DIR                                          |
| 33   | 标题的LMIR.DIR                                          |
| 34   | URL的LMIR.DIR                                           |
| 35   | 全文档的LMIR.DIR                                        |
| 36   | 主体的LMIR.JM                                           |
| 37   | 锚点的LMIR.JM                                           |
| 38   | 标题的LMIR.JM                                           |
| 39   | URL的LMIR.JM                                            |
| 40   | 全文档的LMIR.JM                                         |
| 41   | PageRank                                                |
| 42   | 内链个数                                                |
| 43   | 外链个数                                                |
| 44   | URL中的斜线数                                           |
| 45   | URL长度                                                 |
| 46   | 子页面个数                                              |

## 10.5 元信息

除了这些特征之外，LETOR中还提供了以下元信息。

- 有关语料库的统计信息，例如文档总数，流数以及每个流中（唯一）术语的数。
- 与每个查询关联的文档的原始信息，例如术语频率和文档长度。
- 关系信息，例如超链接图，站点网络信息以及语料库的相似关系矩阵。

利用元信息，人们可以重现现有特征，调整其参数，研究新特征并进行一些高级研究，例如关系排序[14，15]。

## 10.6 学习任务

LETOR支持的主要学习任务是监督排行。 也就是说，给定训练集已完全标记，则采用按等级学习算法来学习排序模型。 使用验证集选择这些模型，最后在测试集上进行评估。

为了使评估更加全面，在LETOR中建议进行五重交叉验证。 特别地，将LETOR中的每个数据集分为五个部分，使用大约相同数量的查询，分别表示为S1，S2，S3，S4和S5，以进行五重交叉验证。 对于每一折，三部分用于训练排序模型，一部分用于调整排名算法的超参数（例如，RankBoost [3]中的迭代次数以及在Rank SVM目标函数中的组合系数） [4，7]），其余部分用于评估学习模型的排名性能（请参见表10.5）。 五次折叠的平均性能用于衡量学习排名算法的整体性能。

可能已经注意到，所有LETOR数据集中的自然标签都是相关度。 如前所述，有时文档的成对偏好甚至全部顺序也是有效的标签。 为了促进使用此类标签的学习，在LETOR 4.0中，通过启发式方法得出MQ2007和MQ2008中带标签文档的总顺序，并将其用于训练。

除了标准的监督排序之外，LETOR还支持半监督排序和排序汇总。 与监督排序的任务不同，半监督排序考虑了已判断和未判断的查询-文档对的训练。 对于排名汇总，查询与一组输入排序列表关联，但与单个文档的功能不关联。 任务是通过汇总多个输入列表来输出排序更好的列表。

可以从官方LETOR网站http://research.microsoft.com/~LETOR/.5下载LETOR数据集，其中包含上述文档的特征表示，其与查询的相关性判断以及分区的训练，验证和测试集（请注意，LETOR数据集正在频繁更新。 预计将来会添加更多的数据集。 此外，LETOR网站已发展成为学习排序研究的门户网站，不仅限于数据发布。 您可以在学习领域中找到代表性论文，教程，活动，研究小组等，然后从网站上进行排名）。

## 10.7 小结

LETOR已在学习排名研究社区中广泛使用。 但是，其当前版本也有局限性。 这里我们列出其中一些。

- 文件抽样策略。对于基于“ Gov”语料库的数据集，检索问题本质上是作为LETOR中的重排任务（对于前1000个文档）。一方面，这是现实世界中Web搜索引擎的一种常见做法。通常，为了提高效率，搜索引擎会使用两个排序：首先使用一个简单的排名（例如BM25 [16]）来选择一些候选文档，然后使用一个更复杂的排名（例如，学习到书中介绍的排名算法）用于产生最终的排名结果。但是，另一方面，也有一些检索应用程序不应该作为重新排序任务使用。将来最好将数据集重新排序以外的数据集添加到LETOR中。
- 特征。在学术界和工业界，越来越多的功能已经得到研究和应用，以提高排序准确性。 LETOR中提供的功能列表远非全面。例如，文档特征（例如文档长度）未包含在OHSUMED数据集中，而邻近特征[18]也未包含在所有数据集中。将来在LETOR数据集中添加更多功能会很有帮助。
- 数据集的规模和多样性。与Web搜索相比，LETOR中数据集的规模（查询数量）要小得多。为了验证用于实际Web搜索的按等级学习技术的性能，需要大规模的数据集。此外，尽管有9个查询集，但只涉及3个文档语料库。将来最好使用更多文档语料库来创建新的数据集。

## 参考文献

1. Aslam, J.A., Kanoulas, E., Pavlu, V., Savev, S., Yilmaz, E.: Document selection methodolo- gies for efficient and effective learning-to-rank. In: Proceedings of the 32nd Annual Interna- tional ACM SIGIR Conference on Research and Development in Information Retrieval (SI- GIR 2009), pp. 468–475 (2009)

2. Craswell, N., Hawking, D., Wilkinson, R., Wu, M.: Overview of the trec 2003 Webtrack. In: Proceedings of the 12th Text Retrieval Conference (TREC 2003), pp. 78–92 (2003)

3. Freund, Y., Iyer, R., Schapire, R., Singer, Y.: An efficient boosting algorithm for combining preferences. Journal of Machine Learning Research **4**, 933–969 (2003)

4. Herbrich, R., Obermayer, K., Graepel, T.: Large margin rank boundaries for ordinal regres- sion. In: Advances in Large Margin Classifiers, pp. 115–132 (2000)

5. Hersh, W., Buckley, C., Leone, T.J., Hickam, D.: Ohsumed: an interactive retrieval evalua- tion and new large test collection for research. In: Proceedings of the 17th Annual Interna- tional ACM SIGIR Conference on Research and Development in Information Retrieval (SI- GIR 1994), pp. 192–201 (1994)

6. Hu, Y., Xin, G., Song, R., Hu, G., Shi, S., Cao, Y., Li, H.: Title extraction from bodies of html documents and its application to web page retrieval. In: Proceedings of the 28th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR 2005), pp. 250–257 (2005)

7. Joachims, T.: Optimizing search engines using clickthrough data. In: Proceedings of the 8th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD 2002), pp. 133–142 (2002)

8. Liu, T.Y., Xu, J., Qin, T., Xiong, W.Y., Li, H.: LETOR: benchmark dataset for research on learning to rank for information retrieval. In: SIGIR 2007 Workshop on Learning to Rank for Information Retrieval (LR4IR 2007) (2007)

9. Minka, T., Robertson, S.: Selection bias in the LETOR datasets. In: SIGIR 2008 Workshopon Learning to Rank for Information Retrieval (LR4IR 2008) (2008)

10. Nie, L., Davison, B. D., Qi, X.: Topical link analysis for web search. In: Proceedings of the 29th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR 2006), pp. 91–98 (2006)

11. Qin, T., Liu, T.Y., Lai, W., Zhang, X.D., Wang, D.S., Li, H.: Ranking with multiple hyper- planes. In: Proceedings of the 30th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR 2007), pp. 279–286 (2007)

12. Qin, T., Liu, T.Y., Xu, J., Li, H.: How to make LETOR more useful and reliable. In: SIGIR 2008 Workshop on Learning to Rank for Information Retrieval (LR4IR 2008) (2008)

13. Qin, T., Liu, T. Y., Zhang, X. D., Chen, Z., Ma, W. Y.: A study of relevance propagation for web search. In: Proceedings of the 28th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR 2005), pp. 408–415 (2005)

14. Qin, T., Liu, T.Y., Zhang, X.D., Wang, D., Li, H.: Learning to rank relational objects and its application to web search. In: Proceedings of the 17th International Conference on World Wide Web (WWW 2008), pp. 407–416 (2008)

15. Qin, T., Liu, T.Y., Zhang, X.D., Wang, D.S., Li, H.: Global ranking using continuous condi- tional random fields. In: Advances in Neural Information Processing Systems 21 (NIPS 2008), pp. 1281–1288 (2009)

16. Robertson, S. E.: Overview of the okapi projects. Journal of Documentation 53(1),3–7(1997)

17. Shakery, A., Zhai, C.: A probabilistic relevance propagation model for hypertext retrieval.In: Proceedings of the 15th International Conference on Information and Knowledge Management (CIKM 2006), pp. 550–558 (2006)

18. Tao, T., Zhai, C.: An exploration of proximity measures in information retrieval. In: Proceedings of the 30th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR 2007), pp. 295–302 (2007)

19. Xu, J., Li, H.: Adarank: a boosting algorithm for information retrieval. In: Proceedings of the 30th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR 2007), pp. 391–398 (2007)

20. Xue, G.R., Yang, Q., Zeng, H.J., Yu, Y., Chen, Z.: Exploiting the hierarchical structure for link analysis. In: Proceedings of the 28th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR 2005), pp. 186–193 (2005)
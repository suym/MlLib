# 将常见的机器学习和深度学习算法用于数据挖掘

初步的框架

## 目录介绍

> ./mllib/classModel目录存放分类算法

> ./mllib/clusterModel目录存放聚类算法

> ./mllib/regressionModel目录存放回归算法

> ./mllib/feature目录存放相关特征工程程序

> ./mllib/visualData目录存放数据可视化程序

> ./mllib/src目录存放公共的代码库

## 将mllib目录下的算法打包成可执行文件mlModel

> pyinstaller config.spec

> 在生成的mlModel.spec配置文件中，加入hiddenimports=['\_sysconfigdata',cython','sklearn','sklearn.ensemble','sklearn.neighbors.typedefs','sklearn.neighbors.quad\_tree','sklearn.tree.\_utils','scipy.\_lib.messagestream']。

> 在pathex选项中加入mllib目录的绝对路径。

> 如果遇到Exception: Versioning for this project requires either an sdist tarball错误，这是pbr问题，加上pbr的版本号(pbr -v)，export PBR\_VERSION=X.Y.Z。

> 如果遇到tensorflow找不到相关动态链接库问题，请参考config.spec文件中的add\_tensorflow\_so函数。

> 如果遇到pygal找不到相关css文件问题，请参考config.spec文件的add\_pygal\_data函数。


## 字符串

python 字符串 format , {x} 会被当做变量，{{x}} 不会，最终输出{x}

## 数组

x_temp = [i for i in range(0,300)]
x = [f(tvParams[0],i) for i in x_temp]
> 语法 [expression for i in iterable]

## 函数

### sum

可对一维数组进行加和

## 循环

### for in

**enumerate**

将数组进行转化，获取索引

```py
for i, image in enumerate(faces.images[:5]):
    # 既获得数据又获得 索引
```

## 库

### matplotlib.pyplot

plot: 画线
scatter：画散点
legend: 绘制LabeL
```py
from matplotlib import pyplot as plt

# 不弹窗画图，jupyter book中？
%matplotlib inline
```

imshow：绘制图像，只要是二维数据即可，不需要额外处理

### numpy

numpy.sum: 可以处理多维数组的加和

poly1d：接收数组生成多项式

np.linspace: 通过定义均匀间隔创建数值序列,
> linspace(start,end,size) size是间隔，算上终点和起始
> 0,100,11, 即 0 ~ 100，并且加上0、80总共11个值， 刚好是0、1、2、3、4、5、6、7、8、9、10， 11个数值

np.martix.I: 逆矩阵 A^-1

argmin: 排序后最小的元素的索引

argsort：从小到大排序后的索引数组

### pandas

#### [[]]

取出多列,数据类型是dataframe，列名会作为表头，在展示时使用，计算时会忽略列名

#### []

取出单列,类型是Series，列名只作为标记？不会被展示

#### loc、iloc

loc：按照行、列 label进行选取
loc[[False, False, True]]：False意为跳过，不选取，True自然为选取
iloc 是基于行列的位置，而非label,所有行，到 列位置 -1？

ature_data = lilac_data.iloc[:, :-1]
loc(): 基于label（或者是boolean数组）进行数据选择
iloc(): 基于position(整数-integer)进行数据选择

#### get_dummies

独热编码

#### head

读取前5行

#### tail
读取最后5行

#### to_numeric

```python

# 将某一列数据转成数值
df['Rings'] = pd.to_numeric(df['Rings'])
```
#### columns

```python

df.columns # 读取或者更新表头
```

#### drop

```python

# 根据index删除？删除最后一行
dx.drop(dx.index[-1])
```

#### replace

```python
# 将某一列的某些值进行更新，replace方法已经被废弃！
df['Sex'] = df.Sex.replace({'M':0, 'F':1, 'I':2})
```

#### cut

```python

# 根据bins对数据值划分区间，然后分别替换
df['Rings'] = pd.cut(df.Rings, bins=[0, 10, 20, 30], labels=['small','middle','large'])
```

#### describe

它用于生成有关数据的统计摘要。这个统计摘要包括了数据列的数量、均值、标准差、最小值、25% 分位数、中位数（50% 分位数）、75% 分位数和最大值

#### concat

axis 默认为0,也就是纵向上进行合并。沿着连接的轴,1 就是横向合并
1： concat就是行对齐，然后将不同列名称的两张表合并
print(pd.concat([features, target], axis=1).head())

### jupyter notebook

命令jupyter notebook运行

`%matplotlib inline`:的作用是将Matplotlib图形嵌入到Notebook单元格中，使得图形能够在Notebook中直接显示，而不是在新窗口中弹出

### sklearn.cluster

#### k_means

聚类直接实现

- `X`：表示需要聚类的数据。
    
- `n_clusters`：表示聚类的个数，也就是 K 值。



### sklearn.decomposition

#### PCA

数据降维

### sklearn.linear_model

#### LogisticRegression


#### Ridge

岭回归

```py
ridge_model = Ridge(fit_intercept=False)  # 参数代表不增加截距项
ridge_model.fit(x, y)
ridge_model.coef_  # 打印模型参数
```

#### Lasso

```py
lasso = Lasso(alpha=a, fit_intercept=False)
lasso.fit(x, y)
lasso.coef_
```

#### LinearRegression

线性回归模型

```py
x = np.array([56, 72, 69, 88, 102, 86, 76, 79, 94, 74])
y = np.array([92, 102, 86, 110, 130, 99, 96, 102, 105, 92])

model = LinearRegression()
model.fit(x.reshape(x.shape[0], 1), y)  # 训练, reshape 操作把数据处理成 fit 能接受的形状

# 得到模型拟合参数
model.intercept_, model.coef_
```

#### mean_absolute_error

mae

#### mean_squared_error

mse

### scipy.linalg

#### hilbert

```py
from scipy.linalg import hilbert

x = hilbert(10)
```

### scipy.optimize

####  leastsq

```py
from scipy.optimize import leastsq

func = lambda p, x: np.dot(x, p)  # 函数公式
err_func = lambda p, x, y: func(p, x) - y  # 残差函数
p_init = np.random.randint(1, 2, 10)  # 全部参数初始化为 1

parameters = leastsq(err_func, p_init, args=(x, y))  # 最小二乘法求解
```

### sklearn.preprocessing

#### scale

规范化处理

将特征数据的分布调整成标准正太分布，也叫高斯分布
即使得数据的均值维0，方差为1



#### PolynomialFeatures

构造特征矩阵

```py
from sklearn.preprocessing import PolynomialFeatures
X = [2, -1, 3]
X_reshape = np.array(X).reshape(len(X), 1)  # 转换为列向量
# 使用 PolynomialFeatures 自动生成特征矩阵
PolynomialFeatures(degree=2, include_bias=False).fit_transform(X_reshape)

x = np.array(x).reshape(len(x), 1)  # 转换为列向量
y = np.array(y).reshape(len(y), 1)

# 使用 sklearn 得到 2 次多项式回归特征矩阵
poly_features = PolynomialFeatures(degree=2, include_bias=False)
poly_x = poly_features.fit_transform(x)
```

### sklearn.pipeline

#### make_pipeline

多模型组合

### sklearn.tree

#### DecisionTreeClassifier

建立 决策树

### sklearn.ensemble

#### BaggingClassifier

建立 Bagging Tree

#### RandomForestClassifier

建立随机森林

#### AdaBoostClassifier

建立 AdaBoost 

#### GradientBoostingClassifier

梯度提升树 GBDT

#### VotingClassifier

投票分类器，组合多个分类器进行投票

### sklearn.metrics

#### accuracy_score

判断输入两个数据间的相同率？
> 判断模型预测的准确率

#### precision_score

查准率计算

#### recall_score

计算召回率

#### f1_score

f1计算

#### roc_curve

计算ROC曲线

#### auc

计算auc

#### r2_score

R方计算

```py
from sklearn.metrics import r2_score

# 分别传入真实观测值和模型预测值
r2_score(y1, model1.predict(x)), r2_score(y2, model2.predict(x))
```

### sklearn.svm

#### SVC

支持向量机分类器

### statsmodels.api

#### OLS

普通最小二乘法

### statsmodels.formula.api

#### smf

```py
import statsmodels.formula.api as smf

model_smf_full = smf.ols(formula='Sales ~ TV + Radio + Newspaper', data=data)
results_smf_full = model_smf_full.fit()

results_smf_full.summary2()  # 输出模型摘要
```


### sklearn.model_selection

#### KFold

进行K折数据

#### cross_val_score

k折数据，交叉验证


### sklearn.naive_bayes

伯努利模型


#### train_test_split

```py
# X_train,X_test, y_train, y_test 分别表示，切分后的特征的训练集，特征的测试集，标签的训练集，标签的测试集；其中特征和标签的值是一一对应的。

# train_data,train_target分别表示为待划分的特征集和待划分的标签集。

# test_size：测试样本所占比例。

# random_state：随机数种子,在需要重复实验时，保证在随机数种子一样时能得到一组一样的随机数。

X_train, X_test, y_train, y_test = train_test_split(
    feature_data, label_data, test_size=0.3, random_state=2
)
```

### sklearn.datasets

#### make_blobs

生成特定的团状数据

- `n_samples`：表示生成数据总个数,默认为 100 个。
    
- `n_features`：表示每一个样本的特征个数，默认为 2 个。
    
- `centers`：表示中心点的个数，默认为 3 个。
    
- `center_box`：表示每一个中心的边界,默认为 -10.0到10.0。
    
- `random_state`：表示生成数据的随机数种子。

#### make_circles

生成线性不可分数据

### jieba

结巴分词模块

### re

正则

### tqdm

通过子线程实现进度显示？

### gensim.models

#### Word2Vec

文字转向量

### ipywidgets

#### interact

允许在图表中，增加可交互的可调节参数，看不同参数下效果

```py
def change_c(c):
    linear_svc.C = c
    linear_svc.fit(x, y)
    plt.figure(figsize=(10, 8))
    plt.scatter(x[:, 0], x[:, 1], c=y, s=40, cmap="bwr")
    svc_plot(linear_svc)


interact(change_c, c=[1, 10000, 1000000])
```
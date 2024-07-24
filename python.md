
## datetime 对象

```python
import datetime

datetime.datetime.now()  # 获取当前时间
# datetime.datetime.now() 返回了一个时间对象，依次为：年、月、日、时、分、秒、毫秒。其中，毫秒的取值范围在 0 <= microsecond < 1000000。

# 等效方法 datetime.datetime.now()
datetime.datetime.today()  # 获取当前时间

# 只取一部分数值
datetime.datetime.now().year  # 返回当前年份

# 手动指定时间
datetime.datetime(2017, 10, 1, 10, 59, 30)  # 指定任意时间
```

### 计算

```python
# 返回了 datetime.timedelta 对象。timedelta 可以用于表示时间上的不同，但最多只保留 3 位，分别是：天、秒、毫秒
datetime.datetime(2018, 10, 1) - datetime.datetime(2017, 10, 1)  # 计算时间间隔

datetime.datetime.now() - datetime.datetime(2017, 10, 1)  # timedelta 表示间隔时间
# datetime.timedelta(days=2232, seconds=48988, microseconds=37705)

# 通过 timedelta 增加一年时间
datetime.datetime.now() + datetime.timedelta(365)  # 需将年份转换为天
```

### 格式化

```python
datetime.datetime.now().strftime("%Y-%m-%d")  # 转换为自定义样式
datetime.datetime.now().strftime("%Y 年 %m 月 %d 日")  # 转换为自定义样式

# 字符串转时间对象
datetime.datetime.strptime("2018-10-1", "%Y-%m-%d")
```

参数：
%y 两位数的年份表示（00-99）

%Y 四位数的年份表示（000-9999）

%m 月份（01 - 12）

%d 月内中的一天（0 - 31）

%H 24 小时制小时数（0 - 23）

%I 12 小时制小时数（01 - 12）

%M 分钟数（00 - 59）

%S 秒（00 - 59）

%a 本地简化星期名称

%A 本地完整星期名称

%b 本地简化的月份名称

%B 本地完整的月份名称

%c 本地相应的日期表示和时间表示

%j 年内的一天（001 - 366）

%p 本地 A.M. 或 P.M. 的等价符

%U 一年中的星期数（00 - 53）星期天为星期的开始

%w 星期（0 - 6），星期天为星期的开始

%W 一年中的星期数（00 - 53）星期一为星期的开始

%x 本地相应的日期表示

%X 本地相应的时间表示

%Z 当前时区的名称


### 时区

协调世界时（UTC）,使用 UTC 偏移量来定义不同时区的时间

```python
datetime.datetime.utcnow()  # 获取 UTC 时间

# 进行北京时间 + 8 处理
utc = datetime.datetime.utcnow()  # 获取 UTC 时间
tzutc_8 = datetime.timezone(datetime.timedelta(hours=8))  # + 8 小时
utc_8 = utc.astimezone(tzutc_8)  # 添加到时区中
print(utc_8)
```

### 时间戳 Timestamp

```python
import pandas as pd

pd.Timestamp("2018-10-1")
pd.Timestamp(datetime.datetime.now())
pd.to_datetime("1-10-2018")
# Timestamp('2018-01-10 00:00:00') 默认 月、日

pd.to_datetime("1-10-2018", dayfirst=True)
# Timestamp('2018-10-01 00:00:00') 日优先
```

### 时间索引 DatetimeIndex

```python
pd.to_datetime(["2018-10-1", "2018-10-2", "2018-10-3"])  # 生成时间索引
# DatetimeIndex(['2018-10-01', '2018-10-02', '2018-10-03'], dtype='datetime64[ns]', freq=None)


# 将 Seris 中字符串转换为时间, 这样处理得到的不是严格意义上的时间索引
s = pd.Series(["2018-10-1", "2018-10-2", "2018-10-3"])
pd.to_datetime(s) 

pd.Series(index=pd.to_datetime(s)).index  # 当时间位于索引时，就是 DatetimeIndex

```

#### date_range

生成任何以一定规律变化的时间索引

pandas.date_range(start=None, end=None, periods=None, freq=’D’, tz=None, normalize=False, name=None, closed=None, **kwargs)
start= ：设置起始时间

end= ：设置截至时间

periods= ：设置时间区间，若 None 则需要单独设置起止和截至时间。

freq= ：设置间隔周期。

tz= ：设置时区。

特别地，freq= 频度参数非常关键，可以设置的周期有：

freq='s' : 秒

freq='min' : 分钟

freq='H' : 小时

freq='D' : 天

freq='w' : 周

freq='m' : 月

freq='BM' : 每个月最后一天

freq='W' ：每周的星期日

```python
pd.date_range("2018-10-1", "2018-10-2", freq="H")  # 按小时间隔生成时间索引
# DatetimeIndex(['2018-10-01 00:00:00', '2018-10-01 01:00:00',
#                '2018-10-01 02:00:00', '2018-10-01 03:00:00',
#                '2018-10-01 04:00:00', '2018-10-01 05:00:00',
#                '2018-10-01 06:00:00', '2018-10-01 07:00:00',
#                '2018-10-01 08:00:00', '2018-10-01 09:00:00',
#                '2018-10-01 10:00:00', '2018-10-01 11:00:00',
#                '2018-10-01 12:00:00', '2018-10-01 13:00:00',
#                '2018-10-01 14:00:00', '2018-10-01 15:00:00',
#                '2018-10-01 16:00:00', '2018-10-01 17:00:00',
#                '2018-10-01 18:00:00', '2018-10-01 19:00:00',
#                '2018-10-01 20:00:00', '2018-10-01 21:00:00',
#                '2018-10-01 22:00:00', '2018-10-01 23:00:00',
#                '2018-10-02 00:00:00'],
#               dtype='datetime64[ns]', freq='H')

# 从 2018-10-1 开始，以天为间隔，向后推 10 次
pd.date_range("2018-10-1", periods=10, freq="D")

# 从 2018-10-1 开始，以 1H20min 为间隔，向后推 10 次
pd.date_range("2018-10-1", periods=10, freq="1H20min")


# offset 对象可以让时间序列进行一定的偏移
from pandas import offsets

# offset 对象让 time_index 依次增加 1 个月 + 2 天 + 3 小时。
time_index + offsets.DateOffset(months=1, days=2, hours=3)

# 统一推后两周
time_index + 2 * offsets.Week()

```

#### 时间间隔 Periods

生成按特定时间跨度的周期对象，后续进行计算时会根据这个跨度进行计算

```python
# 1 年跨度
pd.Period("2018")

# 1 个月跨度
pd.Period("2018-1")

# 1 天跨度
pd.Period("2018-1-1")
```

#### 时间间隔索引 PeriodsIndex

```python
p = pd.period_range("2018", "2019", freq="M")  # 生成 2018-1 到 2019-1 序列，按月分布

# S: start、E: end
p.asfreq(freq="D", how="S")  # 频度从 M → D
```

#### 时序数据选择、切片、偏移

```python
import numpy as np

timeindex = pd.date_range("2018-1-1", periods=20, freq="M")
s = pd.Series(np.random.randn(len(timeindex)), index=timeindex)

# 选择 2008 年的数据
s["2018"]

# 选择 2018 年 7 月到 2019 年 3 月之间的所有数据
s["2018-07":"2019-03"]

# 索引整体偏移
s.shift(3)  # 时间索引以默认 Freq: M 向后偏移 3 个单位

s.shift(-3, freq="D")  # 时间索引以 Freq: D 向前偏移 3 个单位
```

#### 时序数据重采样

调整时间索引序列的频率

```python
dateindex = pd.period_range("2018-10-1", periods=20, freq="D")
s = pd.Series(np.random.randn(len(dateindex)), index=dateindex)

s.resample("2D").sum()  # 降采样，并将删去的数据依次合并到保留数据中

s.resample("2D").asfreq()  # 降采样，直接舍去数据


# open、high、low、close 分别对应开盘价、最高价、最低价和收盘价
s.resample("2D").ohlc()  

s.resample("H").ffill()  # 升采样，使用相同的数据对新增加行填充

s.resample("H").ffill(limit=3)  # 升采样，最多填充临近 3 行
```


#### 时序数据时区处理


```python
naive_time = pd.date_range("1/10/2018 9:00", periods=10, freq="D")

# 转本地时间
utc_time = naive_time.tz_localize("UTC")

# 转特定时区？
utc_time.tz_convert("Asia/Shanghai")


pd.date_range("1/10/2018 9:00", periods=10, freq="D", tz="Asia/Shanghai")
```

## 字符串

python 字符串 format , {x} 会被当做变量，{{x}} 不会，最终输出{x}

```python
year = 1
f"{year}-第一季度"
```

## 数组

x_temp = [i for i in range(0,300)]
x = [f(tvParams[0],i) for i in x_temp]
> 语法 [expression for i in iterable]

## 类

```python
# 继承 nn.Module
class Net(nn.Module):
	pass
```

## 函数

### sum

可对一维数组进行加和

### zip

将两个数组合并进行遍历？

```python

cluster_names = ['KMeans', 'MiniBatchKMeans', 'AffinityPropagation', 
                 'MeanShift', 'SpectralClustering', 'AgglomerativeClustering', 'Birch', 'DBSCAN']


cluster_estimators = [
    cluster.KMeans(n_clusters=2),
    cluster.MiniBatchKMeans(n_clusters=2),
    cluster.AffinityPropagation(),
    cluster.MeanShift(),
    cluster.SpectralClustering(n_clusters=2),
    cluster.AgglomerativeClustering(n_clusters=2),
    cluster.Birch(n_clusters=2),
    cluster.DBSCAN()
]

for name,e in zip(cluster_names, cluster_estimators):
	pass
```

## 循环

### for in

**enumerate**

将数组进行转化，获取索引

```py
for i, image in enumerate(faces.images[:5]):
    # 既获得数据又获得 索引
```

## 重载

### __getitem__

重载 [] 能力，pandas 通过重载实现大量数据 hack 操作


## 使用

### 请求图片并且保存本地，用于进行推理.....

```python
res = requests.get(IMAGE_URL)
with open("test.jpg", "wb") as f:
    f.write(res.content)
```


## 库
### PyTorch

机器之心 [有一篇文章](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650726576&idx=3&sn=4140ee7afc67928333e971062d042c59&chksm=871b24ceb06cadd8922cde50cbc5da6a04fd3f00a78964381c593b2dcf62bb78835159a00f27&scene=0#rd) 对各个框架介绍的非常详细
[PyTorch vs TensorFlow — spotting the difference](https://towardsdatascience.com/pytorch-vs-tensorflow-spotting-the-difference-25c75777377b)


#### 模块

| Package（包）                 | Description（描述）                                                                                                                                                                                                                |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `torch`                    | 张量计算组件, 兼容 NumPy 数组，且具备强大的 GPU 加速支持                                                                                                                                                                                            |
| `torch.autograd`           | 自动微分组件, 是 PyTorch 的核心特点，支持 torch 中所有可微分的张量操作                                                                                                                                                                                   |
| `torch.nn`                 | 深度神经网络组件, 用于灵活构建不同架构的深度神经网络<br>- 全连接层：`torch.nn.Linear()`<br>- MSE 损失函数类：`torch.nn.MSELoss()`<br>- torch.nn.functional<br>  - 神经网络层，激活函数，损失函数<br>  - torch.nn.functional.linear()<br>  - torch.nn.functionalmse_loss()<br><br> |
| `torch.optim`              | 优化计算组件, 囊括了 SGD, RMSProp, LBFGS, Adam 等常用的参数优化方法                                                                                                                                                                               |
| `torch.multiprocessing`    | 多进程管理组件，方便实现相同数据的不同进程中共享视图                                                                                                                                                                                                     |
| `torch.utils`              | 工具函数组件，包含数据加载、训练等常用函数                                                                                                                                                                                                          |
| `torch.legacy(.nn/.optim)` | 向后兼容组件, 包含移植的旧代码                                                                                                                                                                                                               |
#### 支持的类型

https://pytorch.org/docs/stable/tensors.html

| 数据类型 dtype            | CPU 张量               | GPU 张量                    |
| --------------------- | -------------------- | ------------------------- |
| 32-bit 浮点             | `torch.FloatTensor`  | `torch.cuda.FloatTensor`  |
| 64-bit 浮点             | `torch.DoubleTensor` | `torch.cuda.DoubleTensor` |
| 16-bit 半精度浮点          | N/A                  | `torch.cuda.HalfTensor`   |
| 8-bit 无符号整形(0~255)    | `torch.ByteTensor`   | `torch.cuda.ByteTensor`   |
| 8-bit 有符号整形(-128~127) | `torch.CharTensor`   | `torch.cuda.CharTensor`   |
| 16-bit 有符号整形          | `torch.ShortTensor`  | `torch.cuda.ShortTensor`  |
| 32-bit 有符号整形          | `torch.IntTensor`    | `torch.cuda.IntTensor`    |
| 64-bit 有符号整形          | `torch.LongTensor`   | `torch.cuda.LongTensor`   |
|                       |                      |                           |

#### 便捷方法

| 方法                      | 描述                              |
| ----------------------- | ------------------------------- |
| `ones(*sizes)`          | 创建全为 1 的 Tensor                 |
| `zeros(*sizes)`         | 创建全为 0 的 Tensor                 |
| `eye(*sizes)`           | 创建对角线为 1，其他为 0 的 Tensor         |
| `arange(s, e, step)`    | 创建从 s 到 e，步长为 step 的 Tensor     |
| `linspace(s, e, steps)` | 创建从 s 到 e，均匀切分成 steps 份的 Tensor |
| `rand/randn(*sizes)`    | 创建均匀/标准分布的 Tensor               |
| `normal(mean, std)`     | 创建正态分布分布的 Tensor                |
| `randperm(m)`           | 创建随机排列的 Tensor                  |

| 方法                                 | 描述                |
| ---------------------------------- | ----------------- |
| `mean` / `sum` / `median` / `mode` | 均值 / 和 / 中位数 / 众数 |
| `norm` / `dist`                    | 范数 / 距离           |
| `std` / `var`                      | 标准差 / 方差          |
| `cumsum` / `cumprod`               | 累加 / 累乘           |

| 方法                                                      | 描述                             |
| ------------------------------------------------------- | ------------------------------ |
| `abs` / `sqrt` / `div` / `exp` / `fmod` / `log` / `pow` | 绝对值 / 平方根 / 除法 / 指数 / 求余 / 求幂… |
| `cos` / `sin` / `asin` / `atan2` / `cosh`               | 三角函数                           |
| `ceil` / `round` / `floor` / `trunc`                    | 上取整 / 四舍五入 / 下取整 / 只保留整数部分     |
| `clamp(input, min, max)`                                | 超过 min 和 max 部分截断              |
| `sigmod` / `tanh`                                       | 常用激活函数                         |
#### 线性代数

| 方法              | 描述          |
| --------------- | ----------- |
| `trace`         | 对角线元素之和     |
| `diag`          | 对角线元素       |
| `triu` / `tril` | 上三角 / 下三角矩阵 |
| `mm`            | 矩阵乘法        |
| `t`             | 转置          |
| `inverse`       | 求逆矩阵        |
| `svd`           | 奇异值分解       |

```python
# 矩阵的叉乘
b.mm(a), b.matmul(a)
```


squeeze: 减少维度
unsqueeze: 前面增加维度

#### 索引、切片、变换


`reshape()`，`resize()` 和 `view()`，三者直接的区别在于：`resize()` 和 `view()` 执行变换时和原 Tensor 共享内存，即修改一个，另外一个也会跟着改变。而 `reshape()` 则会复制到新的内存区块上

```python
c = t.rand(5, 4)

# 取第 1 行
c[0]

# 取第 1 列
c[:, 0]

# 形状做出改变
c.reshape(4, 5)
c.view(4, 5)
```


#### 张量结构

![image.png](https://fastly.jsdelivr.net/gh/rquanx/my-statics@master/images/20240613003057.png)

#### 自动微分 Autograd

根据前向传播过程自动构建计算图，并自动完成反向传播而不需要手动去实现反向传播的过程

torch.Tensor
- `data`：数据，也就是对应的 Tensor。
- `grad`：梯度，也就是 Tensor 对应的梯度，注意 `grad` 同样是 `torch.Tensor` 对象。
- `grad_fn`：梯度函数，用于构建计算图并自动计算梯度。

#### 数据加载器

```python
import torch

# 训练数据打乱，使用 64 小批量
train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=64, shuffle=True)

# 测试数据无需打乱，使用 64 小批量
test_loader = torch.utils.data.DataLoader(dataset=test, batch_size=64, shuffle=False)
```

#### 构建神经网络

```python
# 通过基础 api 一步一步构建
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 512)  # 784 是因为训练是我们会把 28*28 展平
        self.fc2 = nn.Linear(512, 128)  # 使用 nn 类初始化线性层（全连接层）
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 直接使用 relu 函数，也可以自己初始化一个 nn 下面的 Relu 类使用
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # 输出层一般不激活
        return x


# Sequential 容器结构
model_s = nn.Sequential(
    nn.Linear(784, 512),  # 线性类
    nn.ReLU(),  # 激活函数类
    nn.Linear(512, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
)


```

#### 基础用法

```python
# 创建张量
t.Tensor([1, 2, 3])

import numpy as np
t.Tensor(np.random.randn(3))

# 通过 `shape` 查看 Tensor 的形状
t.Tensor([[1, 2], [3, 4], [5, 6]]).shape

# 数学运算
a = t.Tensor([[1, 2], [3, 4]])
b = t.Tensor([[5, 6], [7, 8]])

print(a + b)
print(a - b)

# 对 a 求列平均
a.mean(dim=0)

# a 中的每个元素求平方
a.pow(2)


```




### TensorFlow

基础属性：数据，数据类型和形状

- tf.Variable：变量 Tensor，需要指定初始值，常用于定义可变参数，例如神经网络的权重。

- tf.constant：常量 Tensor，需要指定初始值，定义不变化的张量。

```python
import tensorflow as tf

v = tf.Variable([[1, 2], [3, 4]])  # 形状为 (2, 2) 的二维变量
v
```

```python
c = tf.constant([[1, 2], [3, 4]])  # 形状为 (2, 2) 的二维常量
c
```

```python
# 输出张量的 NumPy 数组
c.numpy()
```

#### 常用方法

- tf.zeros：新建指定形状且全为 0 的常量 Tensor

- tf.zeros_like：参考某种形状，新建全为 0 的常量 Tensor

- tf.ones：新建指定形状且全为 1 的常量 Tensor

- tf.ones_like：参考某种形状，新建全为 1 的常量 Tensor

- tf.fill：新建一个指定形状且全为某个标量值的常量 Tensor

- tf.linspace：创建一个等间隔序列。

- tf.range：创建一个数字序列。

- tf.matmul: 矩阵乘法

```python
a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])
b = tf.constant([7.0, 8.0, 9.0, 10.0, 11.0, 12.0], shape=[3, 2])

c = tf.linalg.matmul(a, b)  # 矩阵乘法
c
```

- tf.linalg.matrix_transpose：转置矩阵
- tf.cast：数据类型转换
	- `tf.cast(tf.constant(df[["X0", "X1"]].values), tf.float32)`
- tf.nn.sigmoid: sigmoid 函数
- tf.losses.mean_squared_error：MSE 损失函数
- tf.reduce_mean：求和?
- tf.optimizers.SGD：随机梯度下降优化器
- tf.random.normal：生成随机数
- tf.nn.relu：relu 函数
- tf.optimizers.Adam： adam 优化器
- tf.argmax：取值最大的索引
- matrix_inverse: 

#### 常用模块

- tf.：包含了张量定义，变换等常用函数和类。

- tf.data：输入数据处理模块，提供了像 tf.data.Dataset 等类用于封装输入数据，指定批量大小等。

- tf.image：图像处理模块，提供了像图像裁剪，变换，编码，解码等类。

- tf.keras：原 Keras 框架高阶 API。包含原 tf.layers 中高阶神经网络层。

- tf.linalg：线性代数模块，提供了大量线性代数计算方法和类。

- tf.losses：损失函数模块，用于方便神经网络定义损失函数。

- tf.math：数学计算模块，提供了大量数学计算函数。

- tf.saved_model：模型保存模块，可用于模型的保存和恢复。

- tf.train：提供用于训练的组件，例如优化器，学习率衰减策略等。

- tf.nn：提供用于构建神经网络的底层函数，以帮助实现深度神经网络各类功能层。

- tf.estimator：高阶 API，提供了预创建的 Estimator 或自定义组件


####  tf.keras

- `tf.keras.layers.Conv1D` [🔗](https://tensorflow.google.cn/api_docs/python/tf/keras/layers/Conv1D)：一般用于文本或时间序列上的一维卷积。
    
- `tf.keras.layers.Conv2D` [🔗](https://tensorflow.google.cn/api_docs/python/tf/keras/layers/Conv2D)：一般用于图像空间上的二维卷积。
    
- `tf.keras.layers.Conv3D` [🔗](https://tensorflow.google.cn/api_docs/python/tf/keras/layers/Conv3D)。一般用于处理视频等包含多帧图像上的三维卷积。

	- `filters`: 卷积核数量，整数。
    
	- `kernel_size`: 卷积核尺寸，元组。
    
	- `strides`: 卷积步长，元组。
    
	- `padding`: `"valid"` 或 `"same"`。
		- valid: 无法被卷积的像素将被丢弃
		- same: 通过 `0` 填补保证每一个输入像素都能被卷积
- tf.keras.layers.AveragePooling2D: 平均池化
- tf.keras.layers.Flatten()：展开数据，最后进行全连接时使用？
- tf.keras.preprocessing.image.load_img： load image to PIL Image instance，可以直接用于展示
- tf.keras.utils.img_to_array： Converts a PIL Image instance to a NumPy array.



##### 顺序模型

大大简化了模型定义过程

[全部优化器列表](https://tensorflow.google.cn/api_docs/python/tf/keras/optimizers)
[全部损失函数列表](https://tensorflow.google.cn/api_docs/python/tf/losses)
[名称列表](https://keras.io/zh/losses/)


```python
model = tf.keras.models.Sequential()  # 定义顺序模型

# 添加全连接层
model.add(tf.keras.layers.Dense(units=30, activation=tf.nn.relu))  # 输出 30，relu 激活
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))  # 输出 10，softmax 激活

# adam 优化器 + 交叉熵损失 + 准确度评估
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# 损失函数需要根据网络的输出形状和真实值的形状来决定

# 模型训练
model.fit(X_train, y_train, batch_size=64, epochs=5)

# 模型评估
model.evaluate(X_test, y_test)

# 使用参数传入测试数据
model.fit(X_train, y_train, batch_size=64, epochs=5, validation_data=(X_test, y_test))
```


**全连接层**

Dense(units, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)

- **units**: 正整数，输出空间维度。
- **activation**: 激活函数。若不指定，则不使用激活函数(即， 线性激活: `a(x) = x`)。
- **use_bias**: 布尔值，该层是否使用偏置项量。
- **kernel_initializer**: `kernel` 权值矩阵的初始化器。
- **bias_initializer**: 偏置项量的初始化器.
- **kernel_regularizer**: 运用到 `kernel` 权值矩阵的正则化函数。
- **bias_regularizer**: 运用到偏置项的正则化函数。
- **activity_regularizer**: 运用到层的输出的正则化函数。
- **kernel_constraint**: 运用到 `kernel` 权值矩阵的约束函数。
- **bias_constraint**: 运用到偏置项量的约束函数。


##### 函数模型

```python
inputs = tf.keras.Input(shape=(64,))  # 输入层
x = tf.keras.layers.Dense(units=30, activation="relu")(inputs)  # 中间层
outputs = tf.keras.layers.Dense(units=10, activation="softmax")(x)  # 输出层

# 函数式 API 需要指定输入和输出
model = tf.keras.Model(inputs=inputs, outputs=outputs)
model


model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

model.fit(X_train, y_train, batch_size=64, epochs=20, validation_data=(X_test, y_test))
```


##### 模型保存

TensorFlow 模型一般包含 3 类要素，分别是：模型权重值、模型配置乃至优化器配置

```python
model.save_weights("./weights/model")  # 保存检查点名称为 model，路径为 ./weights

model.load_weights("./weights/model")  # 恢复检查点

# 保存完整的模型，即包含模型权重值、模型配置乃至优化器配置等，模型直接拿去推理
model.save("model.h5")  # 保存完整模型

model_ = tf.keras.models.load_model("model.h5")  # 调用模型

# 查看 Keras 模型结构，包含神经网络层和参数等详细数据
model_.summary()

preds = model_.predict(X_test[:3])  # 预测前 3 个测试样本
preds
```

#### Estimator 高阶 API

TensorFlow 中的高阶 API，它可以将模型的训练、预测、评估、导出等操作封装在一起，构成一个 Estimator


一般步骤：
1. 创建一个或多个输入函数。
2. 定义模型的特征列。
3. 实例化 Estimator，指定特征列和各种超参数。
4. 在 Estimator 对象上调用一个或多个方法，传递适当的输入函数作为数据的来源


#### api 总结

封装程度： tf.estimator > tf.keras > tf.nn


### mlxtend

#### mlxtend.preprocessing

**TransactionEncoder**

将列表数据转换为 Apriori 算法 API 可用的格式

类似于独热编码，可以提取数据集中的不重复项，并将每个数据转换为等长度的布尔值表示

#### mlxtend.frequent_patterns

- apriori：寻找频繁项集
- association_rules：生成关联规则

### matplotlib

#### matplotlib.pyplot

- plot: 画线
- scatter：画散点
  - alpha： 绘制点的透明度
  - c: 绘制点的颜色？
  - cmap：按类别进行颜色合并？
- legend: 绘制LabeL
- imshow：绘制图像，只要是二维数据即可，不需要额外处理
- ylim：限定 y 轴范围

```py
from matplotlib import pyplot as plt

# 不弹窗画图，jupyter book中？
%matplotlib inline
```


### numpy

- numpy.sum: 可以处理多维数组的加和
- np.random.shuffle：洗牌算法?
- poly1d：接收数组生成多项式
- np.linspace: 通过定义均匀间隔创建数值序列,
  - linspace(start,end,size) size是间隔，算上终点和起始
  - 0,100,11, 即 0 ~ 100，并且加上0、80总共11个值， 刚好是0、1、2、3、4、5、6、7、8、9、10， 11个数值
- np.martix.I: 逆矩阵 A^-1
- argmin: 排序后最小的元素的索引
- argsort：从小到大排序后的索引数组
- cov：协方差矩阵计算
- mat：转矩阵？
- np.linalg.eig：计算特征值、特征向量
- np.linalg.norm: 计算欧氏距离
- dot: 点乘
- ones：根据shape，生成全 1 的数组
- subtract: 减法
- mean: 平均值
- expand_dims：在指定的位置插入一个新的轴，从而扩展数组的维度，假设 image_a 原来的形状是 (height, width, channels)，那么经过 np.expand_dims 之后，它的形状将变成 (1, height, width, channels)

### pandas

#### 运算符

- [[]]：取出多列,数据类型是dataframe，列名会作为表头，在展示时使用，计算时会忽略列名
- []：取出单列,类型是Series，列名只作为标记？不会被展示

#### 常用方法

- get_dummies：独热编码
- head：读取前5行
- tail: 读取最后5行
- columns: 读取或者更新表头
- describe: 它用于生成有关数据的统计摘要。这个统计摘要包括了数据列的数量、均值、标准差、最小值、25% 分位数、中位数（50% 分位数）、75% 分位数和最大值
- 

**loc、iloc**

loc：按照行、列 label进行选取
loc[[False, False, True]]：False意为跳过，不选取，True自然为选取
iloc 是基于行列的位置，而非label,所有行，到 列位置 -1？

ature_data = lilac_data.iloc[:, :-1]
loc(): 基于label（或者是boolean数组）进行数据选择
iloc(): 基于position(整数-integer)进行数据选择

**read_csv**

```python
pd.read_csv(xx) # 读取数据，行默认以数字作为索引
pd.read_csv(xx，head=None) # 忽略行？

    # 读取数据，并且以第一列作为索引，不要默认的 1、2、3.....了
    df = pd.read_csv("GOOGL.csv", index_col=0)
```

**resample**

```python
# 重采样，对每一列进行聚合，取平均或则总和，
# Q 是按季度的意思
df = df.resample('Q').agg({"Open": 'mean', "High": 'mean', "Low": 'mean',
                               "Close": 'mean', "Adj Close": 'mean', "Volume": 'sum'})
```
**sort_values**

```python
# 按 Volume 排序
df = df.sort_values(by='Volume', ascending=False)
```

**to_numeric**

```python

# 将某一列数据转成数值
df['Rings'] = pd.to_numeric(df['Rings'])
```

**drop**

```python

# 根据index删除？删除最后一行
dx.drop(dx.index[-1])
```

**replace**

```python
# 将某一列的某些值进行更新，replace方法已经被废弃！
df['Sex'] = df.Sex.replace({'M':0, 'F':1, 'I':2})
```

**cut**

```python

# 根据bins对数据值划分区间，然后分别替换
df['Rings'] = pd.cut(df.Rings, bins=[0, 10, 20, 30], labels=['small','middle','large'])
```

**concat**

axis 默认为0,也就是纵向上进行合并。沿着连接的轴,1 就是横向合并
1： concat就是行对齐，然后将不同列名称的两张表合并
print(pd.concat([features, target], axis=1).head())

#### pandas.plotting

- autocorrelation_plot：绘制自相关图

### jupyter notebook

命令jupyter notebook运行

- `%matplotlib inline`:的作用是将Matplotlib图形嵌入到Notebook单元格中，使得图形能够在Notebook中直接显示，而不是在新窗口中弹出
- input('xxx'): 在 note book 上显示输入框


### sklearn

#### sklearn.cluster

- MeanShift：均值漂移聚类

- AffinityPropagation：亲和传播聚类
  - damping：阻尼因子，避免数值振荡。
  - max_iter：最大迭代次数。
  - affinity：亲和评价方法，默认为欧式距离。
- SpectralClustering：谱聚类
- DBSCAN：密度聚类
- hierarchy
- linkage：进行层次聚类/凝聚聚类
- dendrogram：绘制聚类数
- Birch：Birch 聚类
- MiniBatchKMeans

**AgglomerativeClustering**

层次聚类

n_clusters: 表示最终要查找类别的数量，例如上面的 2 类。

metric: 有 euclidean（欧式距离）, l1（L1 范数）, l2（L2 范数）, manhattan（曼哈顿距离）等可选。

linkage: 连接方法：ward（单连接）, complete（全连接）, average（平均连接）可选。

**k_means**

聚类直接实现

- `X`：表示需要聚类的数据。
    
- `n_clusters`：表示聚类的个数，也就是 K 值。


#### sklearn.decomposition

**PCA**

数据降维

n_components= 表示需要保留主成分（特征）的数量。

copy= 表示针对原始数据降维还是针对原始数据副本降维。当参数为 False 时，降维后的原始数据会发生改变，这里默认为 True。

whiten= 白化表示将特征之间的相关性降低，并使得每个特征具有相同的方差。

svd_solver= 表示奇异值分解 SVD 的方法。有 4 参数，分别是：auto, full, arpack, randomized。


#### sklearn.linear_model

- LogisticRegression
- mean_absolute_error: mae
- mean_squared_error: mse
**Ridge**

岭回归

```py
ridge_model = Ridge(fit_intercept=False)  # 参数代表不增加截距项
ridge_model.fit(x, y)
ridge_model.coef_  # 打印模型参数
```

**Lasso**

```py
lasso = Lasso(alpha=a, fit_intercept=False)
lasso.fit(x, y)
lasso.coef_
```

**LinearRegression**

线性回归模型

```py
x = np.array([56, 72, 69, 88, 102, 86, 76, 79, 94, 74])
y = np.array([92, 102, 86, 110, 130, 99, 96, 102, 105, 92])

model = LinearRegression()
model.fit(x.reshape(x.shape[0], 1), y)  # 训练, reshape 操作把数据处理成 fit 能接受的形状

# 得到模型拟合参数
model.intercept_, model.coef_
```

#### sklearn.model_selection

- KFold：进行K折数据
- cross_val_score：k折数据，交叉验证


#### sklearn.naive_bayes

伯努利模型

**train_test_split**

```py
# X_train,X_test, y_train, y_test 分别表示，切分后的特征的训练集，特征的测试集，标签的训练集，标签的测试集；其中特征和标签的值是一一对应的。

# train_data,train_target分别表示为待划分的特征集和待划分的标签集。

# test_size：测试样本所占比例。

# random_state：随机数种子,在需要重复实验时，保证在随机数种子一样时能得到一组一样的随机数。

X_train, X_test, y_train, y_test = train_test_split(
    feature_data, label_data, test_size=0.3, random_state=2
)
```


#### sklearn.neural_network


**MLPClassifier**

实现了具有反向传播算法的多层神经网络结构

- hidden_layer_sizes: 定义隐含层及包含的神经元数量，(20, 20) 代表 2 个隐含层各有 20 个神经元。
- activation: 激活函数，有 identity（线性）, logistic, tanh, relu 可选。
- solver: 求解方法，有 lbfgs（拟牛顿法），sgd（随机梯度下降），adam（改进型 sgd） 可选。adam 在相对较大的数据集上效果比较好（上千个样本），对小数据集而言，lbfgs 收敛更快效果也很好。 
- alpha: 正则化项参数。
- learning_rate: 学习率调整策略，constant（不变），invscaling（逐步减小），adaptive（自适应） 可选。
- learning_rate_init: 初始学习率，用于随机梯度下降时更新权重。
- max_iter: 最大迭代次数。
- shuffle: 决定每次迭代是否重新打乱样本。
- random_state: 随机数种子。
- tol: 优化求解的容忍度，当两次迭代损失差值小于该容忍度时，模型认为达到收敛并且训练停止

#### sklearn.datasets

- fetch_california_housing：加州房价数据
- make_moons：生成月牙状数据
- make_circles：生成线性不可分数据

**load_digits**

images：8x8 矩阵，记录每张手写字符图像对应的像素灰度值

data：将 images 对应的 8x8 矩阵转换为行向量

target：记录 1797 张影像各自代表的数字

数据集：包含由 1797 张数字 0 到 9 的手写字符影像转换后的数字矩阵，目标值是 0-9

**make_blobs**

生成特定的团状数据

- `n_samples`：表示生成数据总个数,默认为 100 个。
    
- `n_features`：表示每一个样本的特征个数，默认为 2 个。
    
- `centers`：表示中心点的个数，默认为 3 个。
    
- `center_box`：表示每一个中心的边界,默认为 -10.0到10.0。
    
- `random_state`：表示生成数据的随机数种子。


### scipy

#### scipy.linalg

**hilbert**

```py
from scipy.linalg import hilbert

x = hilbert(10)
```

#### scipy.optimize

**leastsq**

```py
from scipy.optimize import leastsq

func = lambda p, x: np.dot(x, p)  # 函数公式
err_func = lambda p, x, y: func(p, x) - y  # 残差函数
p_init = np.random.randint(1, 2, 10)  # 全部参数初始化为 1

parameters = leastsq(err_func, p_init, args=(x, y))  # 最小二乘法求解
```

#### sklearn.preprocessing

**scale**

规范化处理

将特征数据的分布调整成标准正太分布，也叫高斯分布
即使得数据的均值维0，方差为1

**PolynomialFeatures**

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

#### sklearn.pipeline

- make_pipeline：多模型组合

#### sklearn.tree

- DecisionTreeClassifier：建立 决策树

#### sklearn.ensemble

- BaggingClassifier：建立 Bagging Tree
- RandomForestClassifier：建立随机森林
- AdaBoostClassifier：建立 AdaBoost 
- GradientBoostingClassifier：梯度提升树 GBDT
- VotingClassifier：投票分类器，组合多个分类器进行投票

#### sklearn.metrics

- accuracy_score：判断输入两个数据间的相同率？
  - 判断模型预测的准确率
- precision_score：查准率计算
- recall_score：计算召回率
- f1_score：f1计算
- roc_curve：计算ROC曲线
- auc：计算auc

**r2_score**

R方计算

```py
from sklearn.metrics import r2_score

# 分别传入真实观测值和模型预测值
r2_score(y1, model1.predict(x)), r2_score(y2, model2.predict(x))
```

#### sklearn.svm

- SVC：支持向量机分类器

### joblib

保存模型，模型存为 `.pkl` 二进制文件

### statsmodels

#### statsmodels.tsa.stattools

- arma_order_select_ic

#### statsmodels.stats.diagnostic

**acorr_ljungbox**

随机序列判断

计算 LB 统计量，默认会返回 LB 统计量和 LB 统计量的 P 值。如果 LB 统计量的 P 值小于 `0.05`，我们则认为该序列为非随机序列，否则就为随机序列

#### statsmodels.graphics.tsaplots

- plot_acf：绘制自相关图的函数
- OLS：普通最小二乘法

#### statsmodels.formula.api

**smf**

```py
import statsmodels.formula.api as smf

model_smf_full = smf.ols(formula='Sales ~ TV + Radio + Newspaper', data=data)
results_smf_full = model_smf_full.fit()

results_smf_full.summary2()  # 输出模型摘要
```

### jieba

结巴分词模块

### re

正则

### tqdm

通过子线程实现进度显示？

### gensim

#### gensim.models

- Word2Vec：文字转向量

#### gensim.ipywidgets

**interact**

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

### torchvision

#### torchvision.transforms

- ToPILImage：加载图片并转换为 PIL IMAGE
- Resize：尺寸变形
- RandomCrop：随机裁剪
- CenterCrop：居中裁剪
- ToTensor：转张量
- Normalize：标准化？
- Compose：组合基础方法
  - `composed = transforms.Compose([transforms.Resize(256), transforms.RandomCrop(224)])`

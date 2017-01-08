# 线性回归
让我们从经典的线性回归模型开始这份教程。在这一章里，你将使用真实的数据集建立起一个房价预测模型，并且若干重要的概念。

## 背景介绍
给定一个大小为$n$的数据集 ${\\{y_{i}, x_{i1}, ..., x_{id}\\}}_{i=1}$，其中$x_{i1}, ... x_{id}$是$d$个属性上的取值，$y_{i}$是待预测的目标。线性回归模型假设目标$y_{i}$可以被属性间的线性组合描述，即

$$y_i = \omega_1x_{i1} + \omega_2x_{i2} + \ldots + \omega_dx_{id} + b,  i=1,...,n$$

例如，在我们将要建模的房价预测问题里，$x_{ij}$是描述房子$i$的各种属性（比如房间的个数），而 $y_{i}$是房屋的价格。

初看起来，这个假设实在过于简单了，变量间的真实关系很难是线性的。但由于其形式简单、易于建模分析，线性回归模型却在实际问题中得到了大量的应用，多数的统计学习、机器学习书籍也会选择对线性模型独立成章重点讲解。

## 效果展示
我们使用从[UCI Housing Data Set](https://archive.ics.uci.edu/ml/datasets/Housing)获得的波士顿房价数据集进行模型的训练和预测。下面的散点图展示了使用模型对部分房屋价格进行的预测。其中，横轴展示了房屋的真实值，纵轴为预测值，当二者值完全相等的时候就会落在虚线上。所以模型预测的越准确，则点离虚线越近。

## 模型概览
在波士顿房价数据集中，和房屋相关的值共有14个：前13个用来描述房屋相关的各种信息，即模型中的 $x_i$；最后一个值为我们要预测的房屋价格的中值，即模型中的 $y_i$。因此，我们的模型就可以表示成：

$$\hat{Y} = \omega_1X_{1} + \omega_2X_{2} + ... + \omega_{13}X_{13} + b$$

$Y\string^$ 表示模型的预测，用来和真实值$Y$区分。模型要学习的参数即：$\omega_1, \dot, \omega_{13}, b$。

有了模型表示之后，我们还需要能够度量给定一组参数后，模型表现的好坏。也就是说，我们需要一个损失函数来指导参数的调整。对于线性回归模型来讲，最常见的损失函数就是均方误差（Mean Squared Error， MSE）了，它的形式是：

$${MSE}={\frac  {1}{n}}\sum _{{i=1}}^{n}({\hat  {Y_{i}}}-Y_{i})^{2}$$

即对于一个大小为n的测试集，$MSE$是$n$个数据预测结果误差平方的均值。

## 数据准备
执行以下命令来准备数据:
```bash
cd datas && python prepare_data.py
```
这段代码将从[UCI Housing Data Set](https://archive.ics.uci.edu/ml/datasets/Housing)下载数据并进行预处理（细节见下节-数据预处理），最后数据将被分为训练集和测试集。

这份数据集共506行，每行包含了波士顿郊区的一类房屋的相关信息及价格的中位数。其各维属性的意义如下：

| 属性名 | 解释 | 类型 |
| ------| ------ | ------ |
| CRIM | 该镇的人均犯罪率 | 连续值 |
| ZN | 占地面积超过25,000平方呎的住宅用地比例 | 连续值 |
| INDUS | 非零售商业用地比例 | 连续值 |
| CHAS | 是否临近 Charles River  | 离散值，1=邻近；0=不邻近 |
| NOX | 一氧化氮浓度 | 连续值 |
| RM | 每栋房屋的平均客房数 | 连续值 |
| AGE | 1940年之前建成的自用单位比例 | 连续值 |
| DIS | 到波士顿5个就业中心的加权距离 | 连续值 |
| RAD | 到径向公路的可达性指数 | 连续值 |
| TAX | 全值财产税率 | 连续值 |
| PTRATIO | 学生与教师的比例 | 连续值 |
| B | 1000(Bk - 0.63)^2，其中BK为黑人占比 | 连续值 |
| LSTAT | 低收入人群占比 | 连续值 |
| MEDV | 房屋价格的中位数 | 连续值 |

### 数据预处理
观察一下数据，我们的第一个发现是：所有的13维属性中，有12维的连续值和1维的离散值（CHAS）。离散值虽然也常使用类似0、1、2这样的数字表示，但是其含义与连续值是不同的，因为这里的差值没有实际意义。例如，我们用0、1、2来分别表示红色、绿色和蓝色的话，我们并不能因此说“蓝色和红色”比“绿色和红色”的距离更远。所以通常对一个有$d$ 个可能取值的离散属性，我们会将它们转为 $d$ 个取值为0或1的二值属性。不过就这里而言，因为CHAS本身就是一个二值属性，就省去了这个麻烦。

另外一个稍加观察即可发现的事实是，各维属性的取值范围差别很大。例如，属性B的取值范围是[0.32, 396.90]，而属性NOX的取值范围是[0.3850, 0.8170]。这里就要用到一个常见的操作-归一化（normalization）了。归一化的目标是把各位属性的取值范围放缩到差不多的区间，例如[-0.5,0.5]。下面的代码展示了一种最简便的操作方法：减掉均值，然后除以原取值范围。

做归一化至少有以下3个理由：
- 过大或过小的数值范围会导致计算时的浮点上溢或下溢。
- 不同的数值范围会导致不同属性对模型的重要性不同（至少在训练的初始阶段如此），而这个隐含的假设常常是不合理的。这会对优化的过程造成困难，使训练时间大大的加长。
- 很多的机器学习技巧/模型（例如L1,L2正则项，Vector Space Model）都基于这样的假设，即所有的属性取值范围都差不多是以0为均值，范围接近于1的。

最后，我们将数据按照9:1的比例分割成训练集和测试集，并将保存后的文件路径分别写入 `train.list` 和 `test.list`两个文件中，供PaddlePaddle读取。
```python
def save_list():
    with open('train.list', 'w') as f:
        f.write('datas/' + train_data + '\n')
    with open('test.list', 'w') as f:
        f.write('datas/' + test_data + '\n')


def preprocess(file_path, feature_num=14):
    data = np.fromfile(file_path, sep=' ')
    data = data.reshape(data.shape[0]/feature_num, feature_num)
    maximums, minimums, avgs = data.max(axis=0), data.min(axis=0), data.sum(axis=0)/data.shape[0]
    for i in xrange(feature_num - 1):
        data[:, i] = (data[:, i] - avgs[i]) / (maximums[i] - minimums[i])

    offset = int(data.shape[0] * 0.9)
    np.save(train_data, data[:offset])
    logging.info('saved training data to %s' % train_data)
    np.save(test_data, data[offset:])
    logging.info('saved test data to %s' % test_data)
    save_list()
```


### 提供数据给PaddlePaddle
准备好数据之后，我们使用一个Python data provider来为PaddlePaddle的训练过程提供数据。一个 data provider 就是一个Python函数，它会被PaddlePaddle的训练过程调用。在这个例子里，只需要读取已经保存好的数据，然后一行一行的返回给PaddlePaddle的训练进程即可。

```python
#dataprovider.py
from paddle.trainer.PyDataProvider2 import *
import numpy as np

# define data types of input
@provider(input_types=[dense_vector(13), dense_vector(1)])
def process(settings, input_file):
    data = np.load(input_file.strip())
    for row in data:
	    yield row[:-1].tolist(), row[-1:].tolist()

```

## 模型配置说明
我们通过一个模型配置文件来定义模型相关的各种细节。

首先，通过 `define_py_data_sources2` 来配置PaddlePaddle从上面的`dataprovider.py`里读入训练数据和测试数据。 PaddlePaddle接受从命令行读入的配置信息，例如这里我们传入一个名为`is_predict`的变量来控制模型在训练和测试时的不同结构。
```python
#trainer_config.py
from paddle.trainer_config_helpers import *

is_predict = get_config_arg('is_predict', bool, False)

# 1. read data. Suppose you saved above python code as dataprovider.py
define_py_data_sources2(
    train_list='datas/train.list',
    test_list='datas/test.list',
    module='dataprovider',
    obj='process',
    args={})

```
接着指定模型优化算法的细节。由于线性回归模型比较简单，我们只要指定基本的`batch_size`即可，它表示每次更新参数的时候使用多少条数据计算梯度信息。
```python
settings(batch_size=2)
```
最后使用`fc_layer`和`LinearActivation`来表示线性回归的模型本身。
```python
#输入数据，13维的房屋信息
x = data_layer(name='x', size=13)

#$$Y\string^ = \omega_1X_{1} + \omega_2X_{2} + ... + \omega_{13}X_{13} + b$$
y_predict = fc_layer(
    input=x,
    param_attr=ParamAttr(name='w'),
    size=1,
    act=LinearActivation(),
    bias_attr=ParamAttr(name='b'))

if not is_predict: #训练时，我们使用MSE，即regression_cost作为损失函数
    y = data_layer(name='y', size=1)
    cost = regression_cost(input=y_predict, label=y)
    outputs(cost) #训练时输出MSE来监控损失的变化
else: #测试时，输出预测值
    outputs(y_predict)
```

## 训练模型
在对应代码的根目录下执行PaddlePaddle的命令行训练程序。这里指定模型配置文件为`trainer_config.py`，训练30轮，结果保存在`output`路径下。
```bash
paddle train --config=trainer_config.py --save_dir=./output --num_passes=30
```

## 应用模型
现在来看下如何使用已经训练好的模型进行预测。这里我们指定一个pass保存的模型，并对测试集中的每一条数据进行预测。
```python
def predict(input_file, model_dir):
    # 准备PaddlePaddle，加载模型
    swig_paddle.initPaddle("--use_gpu=0")
    conf = parse_config('trainer_config.py', 'is_predict=1')
    network = swig_paddle.GradientMachine.createFromConfigProto(conf.model_config)
    network.loadParameters(model_dir)
    slots = [dense_vector(13)]
    converter = DataProviderConverter(slots)
	
	# 对每一条测试数据进行预测
    data = np.load(input_file)
    ys = []
    for row in data:
        result = network.forwardTest(converter([[row[:-1].tolist()]]))
        y_true = row[-1:].tolist()[0]
        y_predict = result[0]['value'][0][0]
        ys.append([y_true, y_predict])
    ys = np.matrix(ys)
    
    #计算在测试集上的MSE，大概在8.92左右
    avg_err = np.average(np.square((ys[:, 0] - ys[:, 1])))
    logging.info('MSE of test set is %f' % avg_err)
```

将模型的预测结果和真实值进行对比，就得到了在本章开头处展示的结果。
```python
    # draw a scatter plot
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    ax.scatter(ys[:, 0], ys[:, 1])
    y_range = [ys[:, 0].min(), ys[:, 0].max()]
    ax.plot(y_range, y_range, 'k--', lw=4)
    ax.set_xlabel('True')
    ax.set_ylabel('Predicted')
    fig.savefig('predictions.png')
    plt.close(fig)
```

## 总结
## 参考文献

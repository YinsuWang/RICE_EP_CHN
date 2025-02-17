# define function
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from scipy.optimize import minimize


# 损害系数
def damageelasticity(e=None, aggregateq=None):
    da = (aggregateq / 100) ** e
    # 获取aggregateq的行数与列数
    [_, n] = da.shape
    for i in range(1, n):
        da[:, i] = da[:, i] / sum(da[:, i])

    d = da * 100
    return d


# AbateCost:计算减排成本
def abatement_cost(mu, theta1, theta2):
    lambda1 = theta1 / 1000 * mu ** theta2
    return lambda1


# decline rate of population or TFP calibration
def decline2013(data, ga_min=None, delta_min=None):
    data = data.copy()
    nn = data.shape
    # data1 = np.zeros((nn[0], 3))
    # for i in range(3):
    #     data1[:, i] = np.mean(data.iloc[:, i:nn[1]-3+i], axis=1)
    # DICE 2013
    # 将data转换为np.array
    data1 = data.values
    g1 = (data1[:, 1:] / data1[:, :-1]) - 1
    delta1 = (g1[:, 1:] / g1[:, :-1]) - 1
    delta = np.mean(delta1, axis=1)
    g = np.mean(g1, axis=1)

    if ga_min is not None:
        gmin = np.ones((data.shape[0],)) * ga_min
        g = np.maximum(g, gmin)
    if delta_min is not None:
        deltamin = np.ones((data.shape[0],)) * delta_min
        delta = np.maximum(delta, deltamin)

    return g, delta


def growth2013(initial, g_0, delta, t):
    g = np.zeros((g_0.shape[0], t + 1))  # para['deltaA'] 是int，shape[1])会超出范围
    data = np.zeros((initial.shape[0], t + 1))
    g[:, 0] = g_0
    data[:, 0] = initial

    # DICE 2013
    for i in range(1, t + 1):
        g[:, i] = g[:, i - 1] / (1 + delta)
        data[:, i] = data[:, i - 1] * (1 + g[:, i])

    # 将data和g转换味dataframe并且将index设置为initial的index
    data = pd.DataFrame(data, index=initial.index)
    g = pd.DataFrame(g, index=initial.index)
    return data, g


# 需要一个新的函数，用于计算delta和g，可能的思路是根据历史数据，逐个地区迭代g和delta,得到最为拟合的g和delta
def decline(data, ga_min=None, delta_min=None):  # decline和growth都有问题
    data = data.copy()
    # 先求均值，再计算g与delta
    n = data.shape[1]
    # data1 = np.zeros((data.shape[0], 3))
    # for i in range(3):
    #     data1[:, i] = np.mean(data.iloc[:, i:n - 3 + i], axis=1)  # 可能是矩阵的方向不对

    # DICE 2013
    # g1 = (data1[1:, :] / data1[:-1, :]) - 1
    # delta1 = (g1[:-1, :] / g1[1:, :]) - 1
    # delta = np.mean(delta1, axis=0)
    # g = np.mean(g1, axis=0)

    # DICE 2016
    data1 = data.values
    g1 = (data1[:, 1:] / data1[:, :-1])
    delta1 = -np.log(g1[:, 1:] / g1[:, :-1])  # RuntimeWarning: invalid value encountered in log
    delta = np.mean(delta1, axis=1)
    g = np.mean(g1, axis=1)

    if ga_min is not None:
        gmin = np.ones((data.shape[0],)) * ga_min
        g = np.maximum(g, gmin)
    if delta_min is not None:
        deltamin = np.ones((data.shape[0],)) * delta_min
        delta = np.maximum(delta, deltamin)

    g = g.T
    delta = delta.T
    return g, delta


def growth(initial, g_0, delta, t):
    if isinstance(initial, (int, float)):
        n = 1
    else:
        n = initial.shape[0]
    g = np.zeros((n, t))  # para['deltaA'] 是int，shape[1])会超出范围
    data = np.zeros((n, t))
    g[:, 0] = g_0
    data[:, 0] = initial

    # DICE 2016
    for i in range(1, t):
        g[:, i] = g_0 * np.exp(-delta * (i))
        data[:, i] = data[:, i - 1] * (1 + g[:, i])

    # 将data和g转换味dataframe并且将index设置为initial的index
    if isinstance(initial, (int, float, np.ndarray)):
        pass
    else:
        data = pd.DataFrame(data, index=initial.index)
        g = pd.DataFrame(g, index=initial.index)
    return data


def growth_sigma(initial, g_0, delta, t):
    if isinstance(initial, (int, float)):
        n = 1
        # 将initial转换位1*1的ndarray
        initial = np.array([[initial]])
    else:
        n = initial.shape[0]
    g = np.zeros((n, t))  # para['deltaA'] 是int，shape[1])会超出范围
    data = np.zeros((n, t))
    # 如果g_0的维度为2，且为n*1的ndarray，则转换为(n,)的ndarray;如果g_0维度大于二且第二维大于1，则报错
    if g_0.ndim == 2:
        if g_0.shape[1] == 1:
            g_0 = g_0.iloc[:, 0]
            delta = delta.iloc[:, 0]
        else:
            raise ValueError('g_0的维度应为1或者2且第二维度为1')

    g[:, 0] = g_0
    if isinstance(initial, pd.DataFrame):
        data[:, 0] = initial.iloc[:, 0].values
    else:
        data[:, 0] = initial
    # DICE 2016
    for i in range(1, t):
        g[:, i] = g_0 * (1 + delta) ** (i)
        data[:, i] = data[:, i - 1] * np.exp(g[:, i])

    # 将data和g转换味dataframe并且将index设置为initial的index
    if isinstance(initial, (int, float, np.ndarray)):
        pass
    else:
        data = pd.DataFrame(data, index=initial.index)
        g = pd.DataFrame(g, index=initial.index)
    return data


def growth_tfp(initial, g_0, delta, t):
    if isinstance(initial, (int, float)):
        n = 1
    else:
        n = initial.shape[0]
    g = np.zeros((n, t))  # para['deltaA'] 是int，shape[1])会超出范围
    data = np.zeros((n, t))
    g[:, 0] = g_0
    if isinstance(initial, pd.DataFrame):
        data[:, 0] = initial.iloc[:, 0].values
    else:
        data[:, 0] = initial

    # DICE 2016/2022
    for i in range(1, t):
        g[:, i] = g_0 * np.exp(-delta * (i))
        data[:, i] = data[:, i - 1] / (1 - g[:, i])

    # 将data和g转换味dataframe并且将index设置为initial的index
    if isinstance(initial, (int, float, np.ndarray)):
        pass
    else:
        data = pd.DataFrame(data, index=initial.index)
        g = pd.DataFrame(g, index=initial.index)
    return data


def growth_tfp_for_ndarray(initial, g_0, delta, t):
    n = initial.shape[0]
    g = np.zeros((n, t))  # para['deltaA'] 是int，shape[1])会超出范围
    data = np.zeros((n, t))
    g[:, 0] = g_0
    data[:, 0] = initial

    # DICE 2016/2022
    for i in range(1, t):
        g[:, i] = g_0 * np.exp(-delta * (i))
        data[:, i] = data[:, i - 1] / (1 - g[:, i])
    return data


def calibrate(data, upper, lower, growthfunction=None, chn_start=11):
    if growthfunction is None:
        def growthfunction(l, g0, delta, t):
            g = np.zeros((t,))
            data = np.zeros((t,))
            g[0] = g0
            data[0] = l[0]
            for i in range(1, t):
                g[i] = g0 * np.exp(-delta * (i))
                data[i] = data[i - 1] * (1 + g[i])
            return data

    # def objective(x, l):
    #     g0, delta = x
    #     l_fit = growthfunction(l[0], g0, delta, len(l))  # Use growth function instead of for loop
    #     error = np.sum((l_fit - l) ** 2)
    #     return error
    def objective(x, l):
        delta = x[0]
        g0 = x[1:]
        g0 = np.append(g0, np.tile(g0[-1], (l.shape[0] - len(g0), 1)))
        # 如果l是二维的
        if len(l.shape) == 2:
            l_fit = growthfunction(l[:, 0], g0, delta, l.shape[1])
            error = np.sum((l_fit[:, -1] - l[:, -1]) ** 2)
        else:
            l_fit = growthfunction(l[0], g0, delta, len(l))  # Use growth function instead of for loop
            error = np.sum((l_fit[-1] - l[-1]) ** 2)
        # error = np.sum((l_fit - l) ** 2)
        return error
    # 如果data是dataframe，那么将其转换为array
    if isinstance(data, pd.DataFrame):
        # 转换位ndarray
        data = data.values
    # 如果data是一维数据，那么将其转换为二维数据
    if len(data.shape) == 1:
        data = data.reshape((1, -1))

    # 遍历data的每一行
    # g0 = np.zeros((data.shape[0],))
    # delta = np.zeros((data.shape[0],))
    # for i in range(data.shape[0]):
    #     l = data[i, :]
    #     res = differential_evolution(objective, bounds, args=(l,))
    #     g0[i] = res.x[0]
    #     delta[i] = res.x[1]

    if chn_start == 0:
        init = np.zeros((data.shape[0] + 1))
        bounds = np.ones((data.shape[0] + 1)) * upper
    else:
        init = np.zeros((chn_start + 2))
        bounds = np.ones((chn_start + 2)) * upper

    bounds = [(min(lower, x), max(lower, x)) for x in bounds]
    # 判断upper正负
    if upper > 0:
        scal=1
    else:
        scal=-1
    # bounds[0] = (0.01*scal, 0.01*scal)
    res = minimize(objective, init, bounds=bounds, args=(data))
    delta = res.x[0]
    g0 = res.x[1:]
    return g0, delta





''' pylines = mpars.matlab2python(
    'D:/OneDrive/OneDrive - smail.swufe.edu.cn/1/RICE_EnergyPoverty/code/DataGenerate/ReadData.m', output='readdata.py')
'''

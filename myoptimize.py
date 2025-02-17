import pandas as pd
from scipy.optimize import minimize
import numpy as np
from RICE_Calculater import calibrate
from RICE_Calculater import growth_tfp
from RICE_Calculater import growth_tfp_for_ndarray
from equations import *
from pyomo.environ import value

# import equations_copy和equations的区别在于，能源支出发生在储蓄前后（是否影响实际收入）
# equations_copy能源支出在储蓄之前，即k=(income-energy_expenditure)*savingrate;
# 而equations中为k=income*savingrate, energy_expenditure + consumption = income*(1-savingrate)

# 假设了税收是在分配完capital后才考虑的，因此不会出现负的资本积累，即lambda和tax的惩罚被弱化了
# 这里考虑到capital并不是均匀的分布在不同quintile，富人的capital占比更大，并且一个地区所有L K都参与生产，而碳税收入将返还给所有人，只有
# 考虑跨区域碳税返还的情况下，需要考虑不同地区的碳税收入的分配
# 解决不了暂时不管了

def welfare_increase(argument, s, para, initial, var, regions, objfunction, step=1):
    # 如果mu为dataframe或者ini则转换为ndarray
    if isinstance(argument, pd.DataFrame):
        argument = argument.values
    elif isinstance(argument, int):
        argument = np.array([[argument]])

    if para['ifprovince'] == 1 or para['ifprovince'] == 3:
        argument = np.append(argument, np.tile(argument[-1, :], (para['I'] - argument.shape[0], 1)), axis=0)

    if argument.ndim == 1 or argument.shape[1] == 1:
        if argument.shape[0] == 1:
            argument = argument * np.ones((para['I'], para['Tmax']))
        else:
            argument = np.repeat(argument, para['Tmax'], axis=1)
    elif argument.shape[1] < para['Tmax']:
        if argument.shape[0] == 1:
            # 将mu的剩余位置补充为1
            argument = np.append(argument, np.ones((1, para['Tmax'] - argument.shape[1])), axis=1)
        elif argument.shape[0] == para['I']:
            g, delta = calibrate(argument, [(0.001, 1), (0.001, 1)], growth=growth_tfp)
            argument = growth_tfp(argument[:, -1], g, delta, para['Tmax'] - argument.shape[1])
            argument = np.append(argument, argument, axis=1)
            # 截尾mu，将大于1的值设为1，小于0的值设为0
            # argument[argument > 1] = 1
            # argument[argument < 0] = 0
    mu = np.cumsum(argument, axis=1)
    # mu = np.minimum(np.maximum(mu, 0), 1)
    result = objfunction(mu, s, para, initial, var, regions, step=step)
    # total_welfare = np.sum(result['welfare'])
    # total_welfare = np.average(result['welfare'],weights=var['L'])
    total_welfare = np.sum(result['welfare'] * var['L'])
    # total_welfare = np.sum(result['welfare'])
    return total_welfare, result


def welfare_increase_2(argument, s, para, initial, var, regions, objfunction, step=1):
    mu = growth_tfp_for_ndarray(argument[:, 0], argument[:, 1], argument[:, 2], para['Tmax'])
    mu[mu > 1] = 1
    mu[mu < 0] = 0
    # mu = np.minimum(np.maximum(growth_tfp(argument[:, 0], argument[:, 1], argument[:, 2], para['Tmax']), 0), 1)
    result = objfunction(mu, s, para, initial, var, regions, step=step)
    # total_welfare = np.sum(result['welfare'])
    # total_welfare = np.average(result['welfare'],weights=var['L'])
    total_welfare = np.sum(result['welfare'] * var['L'])
    total_welfare = np.sum(result['welfare'])
    return total_welfare, result


def welfare(argument, s, para, initial, var, regions, objfunction, step=1):
    # 如果mu为dataframe或者ini则转换为ndarray
    if isinstance(argument, pd.DataFrame):
        argument = argument.values
    elif isinstance(argument, int):
        argument = np.array([[argument]])

    # 对argument横向累加,一个尝试，记得改回来
    if para['if_cum']:
        argument = np.cumsum(argument, axis=1)

    if para['ifprovince'] == 1 or para['ifprovince'] == 3:
        argument = np.append(argument, np.tile(argument[-1, :], (para['I'] - argument.shape[0], 1)), axis=0)

    if argument.ndim == 1 or argument.shape[1] == 1:
        if argument.shape[0] == 1:
            argument = argument * np.ones((para['I'], para['Tmax']))
        else:
            argument = np.repeat(argument, para['Tmax'], axis=1)
    elif argument.shape[1] < para['Tmax']:
        if argument.shape[0] == 1:
            # 将mu的剩余位置补充为1
            argument = np.append(argument, np.ones((1, para['Tmax'] - argument.shape[1])), axis=1)
        elif argument.shape[0] == para['I']:
            g, delta = calibrate(argument, [(0.001, 1), (0.001, 1)], growth=growth_tfp)
            argument = growth_tfp(argument[:, -1], g, delta, para['Tmax'] - argument.shape[1])
            argument = np.append(argument, argument, axis=1)
            # 截尾mu，将大于1的值设为1，小于0的值设为0
            # argument[argument > 1] = 1
            # argument[argument < 0] = 0

    result = objfunction(argument, s, para, initial, var, regions, step=step)
    # total_welfare = np.sum(result['welfare'])
    # total_welfare = np.average(result['welfare'],weights=var['L'])
    # total_welfare = np.sum(result['welfare'] * var['L'])
    total_welfare = np.sum(result['welfare'])
    return total_welfare, result


def objective_function_mu_negishi_china(mu, savingrate, para, initial, var, regions, step=1):
    para = para.copy()
    # 判断是否存在IFprovince
    if 'ifprovince' not in para.keys():
        ifprovince = 0
    else:
        ifprovince = para['ifprovince']
    # ifprovince 应该是一个dataframe，对应每个区域的
    # 应该是在cr/regions中获取ifprovince的信息
    tax = np.power(mu, para['theta'][1, 0] - 1) * var['pb'] / 1000
    # mu = np.power((tax * 1000 / var['pb']), 1 / (para['theta'][1, 0] - 1))
    # mu[mu < 0] = 0
    # 人均产出、人均排放与大气海洋生物圈二氧化碳存量
    production, income, co2, temperature, emissions, damage, lambda1, consumption_pre, consumption_bar = \
        product(var['A'], initial['K'], var['Ltol'], para['alpha'], var['sigma'], mu, savingrate, para, initial, var,
                step=step)

    # 人均能源消费
    energy_expenditure, energy_expenditure_grouped, energy_consumption, hdd, cdd = \
        energy(para['HT'], para['CT'], temperature, initial, var, para)

    # 人均消费、税后收入
    emissions_oneyear = emissions/1
    consumption, income_taxed, income_bar, grossconsumption, taxpayment = \
        taxing(income, consumption_pre, consumption_bar, para['q'], para['d'], para['tau'], para['ryc'], damage,
               lambda1, tax, emissions_oneyear, var['Ltol'], regions, ifprovince, savingrate)
    # /step 是因为收入是流量（当期收入），emissions是step期的流量（step期内的总排放）
    # Energy Poverty
    energy_poverty_grouped = energy_expenditure_grouped / income_taxed
    consumption_real = (consumption - energy_expenditure_grouped) # * 1000  # 调整一下单位为 thousand USD
    energy_poverty = energy_expenditure / (income)

    # Negishi welfare
    welfare = negishi_welfare(para, consumption_real, step, var['L'])

    # 保存结果
    result = {}
    for var in locals().keys():
        if var != 'result':
            result[var] = locals()[var]

    return result


def objective(argument, _s, _para, _initial, _var, _regions, _objective_function_mu_negishi_china, _step):
    argument = argument.reshape((int(argument.shape[0] / _para['Tmax']), _para['Tmax']))
    rew, _ = welfare(argument, _s, _para, _initial, _var, _regions, _objective_function_mu_negishi_china, step=_step)
    return -rew


def objective_increase(argument, _s, _para, _initial, _var, _regions, _objective_function_mu_negishi_china, _step):
    length = _para['Tmax']
    # lengte = 3
    argument = argument.reshape((int(argument.shape[0] / length), length))
    rew, _ = welfare_increase(argument, _s, _para, _initial, _var, _regions, _objective_function_mu_negishi_china,
                                step=_step)
    return -rew


def objective_cons(argument, _s, _para, _initial, _var, _regions, _objective_function_mu_negishi_china, _step):
    argument = argument.reshape((int(argument.shape[0] / _para['Tmax']), _para['Tmax']))
    rew, res = welfare(argument, _s, _para, _initial, _var, _regions, _objective_function_mu_negishi_china, step=_step)
    if res['consumption'].min() < 0:
        return (100 - rew)
    return -rew


def objective_temp(argument, _s, _para, _initial, _var, _regions, _objective_function_mu_negishi_china, _step):
    argument = argument.reshape((int(argument.shape[0] / _para['Tmax']), _para['Tmax']))
    rew, res = welfare(argument, _s, _para, _initial, _var, _regions, _objective_function_mu_negishi_china, step=_step)
    if res['temperature'].max() > 2:
        return (100 - rew)
    return -rew


def nocons_t(argument, _s, _para, _initial, _var, _regions, _objective_function_mu_negishi_china, _step):
    argument = argument.reshape((int(argument.shape[0] / _para['Tmax']), _para['Tmax']))
    _, rew = welfare(argument, _s, _para, _initial, _var, _regions, _objective_function_mu_negishi_china, step=_step)
    ret = rew['temperature'].flatten()
    return ret


def nocons_c(argument, _s, _para, _initial, _var, _regions, _objective_function_mu_negishi_china, _step):
    argument = argument.reshape((int(argument.shape[0] / _para['Tmax']), _para['Tmax']))
    _, rew = welfare(argument, _s, _para, _initial, _var, _regions, _objective_function_mu_negishi_china, step=_step)
    taxed = rew['income'] - rew['emissions'] * rew['tax']
    ret = rew['consumption'].flatten()
    return ret


def nocons_c_r(argument, _s, _para, _initial, _var, _regions, _objective_function_mu_negishi_china, _step):
    argument = argument.reshape((int(argument.shape[0] / _para['Tmax']), _para['Tmax']))
    _, rew = welfare(argument, _s, _para, _initial, _var, _regions, _objective_function_mu_negishi_china, step=_step)
    taxed = rew['income'] - rew['emissions'] * rew['tax']
    ret = rew['consumption_real'].flatten()
    return ret


def nocons_mu(argument, _para):
    argument = argument.reshape((int(argument.shape[0] / _para['Tmax']), _para['Tmax']))
    mu = np.cumsum(argument, axis=1).flatten()
    return mu


def nocons_t_1(argument, _s, _para, _initial, _var, _regions, _objective_function_mu_negishi_china, _step):
    argument = argument.reshape((_para['I'], _para['Tmax']))
    _, rew = welfare_increase(argument, _s, _para, _initial, _var, _regions, _objective_function_mu_negishi_china, step=_step)
    ret = rew['temperature'].flatten()
    return ret

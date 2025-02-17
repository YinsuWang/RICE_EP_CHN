# Function: 主程序
import numpy as np
from my_tools import *
from RICE_Calculater import *
from myoptimize import *
import pickle
import time
import datetime
import os
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
from scipy import optimize
from multiprocessing import Pool
import multiprocessing
import pdb
from functools import partial
import threading
import re

# 是否采用新的数据
new_data = True
nice = True
if_cum = False
# 是否包括省级数据,0/-1为不包括(RecyclingTax/NoTax)，1为包括并设置统一税率，并将全部收入均等的返还（不利于高排放省份）(RecyclingFlatTax)，
# 2为包括省并设置不同税率，但是税收统一再分配(RecyclingRev)，
# 3为根据全国碳排放征收碳税，但是根据收入分配到不同地区(RecyclingIncomeTax),即根据消费征税
# 4为根据消费征税，但是各省设置不同税率，税收全国统一再分配(RecyclingIndConsTax)
ifprovince = 0
onlychn = True
Negishi = False
beta = 0.015
step = 5
regionsnum = 41
parafile = 'Inputdata/Parameters_'+str(regionsnum)+'.xlsx'
temperaturefile = 'Inputdata/TmeanandTdiff_'+str(regionsnum)+'.dta'
regionfile = 'Inputdata/Region_'+str(regionsnum)+'.xlsx'
inputfile = 'Inputdata/InputData_'+str(regionsnum)+'.txt'
# 初始参数设置
# Tstart:起始年份,考虑设置为2015;HT/CT: 设定平衡点温度用于计算HDD/CDD
set_p = {'Negishi': Negishi, 'Tend': 2100, 'Tstart': 2015, 'N': 5, 'HT': 18.3, 'CT': 23.9, 'Tpop': 0, 'TA': 10,
         'ifprovince': ifprovince, 'new_data': new_data, 'step': step, 'beta': beta, 'epsilon': 1, 'parafile': parafile,
         'temperaturefile': temperaturefile, 'regionfile': regionfile, 'inputfile': inputfile}
# 设置A L 的校准参数:采用近Tpop年的数据进行校准人口数据，Tpop=0则使用外部数据（WPP）;采用近TA年的数据进行校准TFP
set_p.update({'n': 5, 'Tmax': set_p['Tend'] - set_p['Tstart'] + 1})  # 在生成外生数据时仍需要获得完整的Tmax，对于Tmax的修改应该在readdata之后
para, initial, var, regions = inputdata(set_p, new_data)  # gengrate variables 时step为1，后续计算step才为5
s = 0.258 * np.ones((para['I'], para['Tmax']))
num_countries = regions.ParentRegion.isna().sum()
if nice:
    para['phi'] = np.array([[0.88, 0.12, np.nan], [0.04704,0.94796, 0.005], [np.nan, 0.00075, 0.99925]])
    initial['pw'] = 1260
    initial['CO2'] = np.array([[787], [1600], [10010]])
    name = 'nice'
else:
    name = 'dice'
# var['pb'] = var['pb'] * 550 / 1260  # pw = 550
# var['sigma'] = var['sigma'] / 0.75  # 假设mu0=0.25

# 设置碳税返还场景
# delta:碳税收入的返还矩阵
scenarios = ["NoTax", "RecyclingTax", "RecyclingRev", "RecyclingFlatTax", "RecyclingConsTax", "RecyclingIndConsTax"]
# scenarios = ["RecyclingFlatTax"]
# goals = ["Pos2C", "Pos", "2C", ""]  # 无限制、收入/消费大于0、温度不超过2摄氏度、收入大于0且温度不超过2摄氏度
goals = ["2C", ""]
results = {}
# bounds = [(0, np.inf)] * (para['I'] * para['Tmax'])  # argument 为tax时的bounds
# 根据技术水平设置mu的上限，将95%分位数设置为2，最低设置为0.5
# mu_limit = var['A'][:, 0] * 0.25 / np.percentile(var['A'][:, 0], 95) + 0.5
mu_limit = (np.log(var['A'] / var['sigma'])[:, 0] - np.min(np.log(var['A'] / var['sigma'])[:, 0])) * 0.25 \
           / (np.max(np.log(var['A'] / var['sigma'])[:, 0]) - np.min(np.log(var['A'] / var['sigma'])[:, 0])) + 0.75
mu_limit = np.linspace(mu_limit, 1, 8).T
mu_limit = np.append(mu_limit, np.tile(np.linspace(1, 1.5, para['Tmax'] - 7), (para['I'], 1))[:, 1:], axis=1)
mu_limit = np.tile(np.concatenate((np.linspace(0.5,1,4), np.ones(6) * 1,np.linspace(1,1.5,8) * 1)),(para['I'], 1))
# 生成一个para['I'] * para['Tmax']的矩阵，前10列为1，后面为1.5
mu_limit = np.ones((para['I'], para['Tmax'])) * 1
# mu_limit[:, :9] = 1
# mu_limit_chn_temp = np.min(mu_limit[11:-1, :], axis=0)
# mu_limit_chn = [(0, x) for x in mu_limit_chn_temp]
bounds_orgin = [(0, x) for x in mu_limit.flatten()]
bounds_orgin_cum = [(0, 10)] * (para['I'] * para['Tmax'])
bounds_orgin_dep = [(0, x) for x in mu_limit.flatten()]


def constraint_function_2C(_argument, _para):
    return 2 - (nocons_t(_argument, s, _para, initial, var, regions, objective_function_mu_negishi_china, step))
def constraint_function_consumption(_argument, _para):
    return nocons_c_r(_argument, s, _para, initial, var, regions, objective_function_mu_negishi_china, step)
def constraint_function_mu(_argument, _para):
    if _para['ifprovince'] == 1 or _para['ifprovince'] == 3:
        limit = (mu_limit[0:(regions.ParentRegion.isna().sum() + 1), :]).flatten()
    else:
        limit = mu_limit.flatten()
    return limit - nocons_mu(_argument, _para)

def create_constraints(_para, _goal, _coefficient=0.5):
    constraints1 = []
    _coefficient = 0.2
    if _goal == "2C":
        para['if_cum'] = if_cum
        constraints1 = [{'type': 'ineq', 'fun': constraint_function_2C, 'args': (_para,)}]
    elif _goal == "Pos":
        para['if_cum'] = if_cum
        constraints1 = [{'type': 'ineq', 'fun': constraint_function_consumption, 'args': (_para,)}]
    elif _goal == "Pos2C":
        para['if_cum'] = if_cum
        constraints1 = [{'type': 'ineq', 'fun': constraint_function_2C, 'args': (_para,)},
                        {'type': 'ineq', 'fun': constraint_function_consumption, 'args': (_para,)}]
    elif _goal == "":
        para['if_cum'] = if_cum
        constraints1 = []

    if para['if_cum']:
        constraints1 = constraints1 + [{'type': 'ineq', 'fun': constraint_function_mu, 'args': (_para,)}]
        _coefficient = 0

    _constraints = constraints1
    return _constraints, _coefficient


def optimize_scenario(_problem):
    options = {'disp': True, 'maxiter': 1000, }#'ftol': 1e-12, }
    _scenarioname = _problem['scenarioname']
    _constraints, coefficient = create_constraints(_problem['para'], _problem['goal'], _problem['coefficient'])
    _init = coefficient * _problem['init']
    _res = minimize(objective, _init, args=_problem['args'], method='SLSQP', bounds=_problem['bounds'],
                    constraints=_constraints, options=options)
    print(f"\n end optimization \n scenario:{_problem['scenarioname']}\n mode:{_problem['para']['ifprovince']}")
    # resultargument = _res.x.reshape((int(_res.x.shape[0] / _problem['para']['Tmax']), _problem['para']['Tmax']))
    # _, _result = welfare(resultargument, s, _problem['para'], initial, var, regions,
    #                      objective_function_mu_negishi_china, step=step)

    _runtime = record_program_runtime(_problem['start_time'])
    _result = {'scenarioname': _problem['scenarioname'], 'res': _res, 'runtime': _runtime, 'problem': _problem, }
    return _result


if __name__ == '__main__':
    start_time = time.time()
    # 获取当前时间，并以时间为文件名保存结果
    current_time = datetime.datetime.now()
    time_string = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"OutputData/results_{time_string}.txt"
    with open('OutputData/results_status.txt', 'a') as f:
        f.write(f"start time:{time_string}\n")
    # 创建一个空的问题列表
    problems = []
    scenarionames = []
    args = {}
    constraints = {}
    coefficient = 0.2
    num_if_cum = 0
    for k in range(0, len(goals)):
        for n in range(len(scenarios)):
            scenario = scenarios[n]
            if scenario == "NoTax":
                para['tau'] = np.zeros((para['I'], para['N']))
                para['ryc'] = np.zeros((para['I'], para['N']))
                para['ifprovince'] = -1
            elif scenario == "RecyclingFlatTax":
                para['ryc'] = np.ones((para['I'], para['N'])) * 0.2
                para['tau'] = para['q'].copy()
                para['ifprovince'] = 1
            elif scenario == "RecyclingRev":
                para['ryc'] = np.ones((para['I'], para['N'])) * 0.2
                para['tau'] = para['q'].copy()
                para['ifprovince'] = 2
            elif scenario == "RecyclingTax":
                para['ryc'] = np.ones((para['I'], para['N'])) * 0.2
                para['tau'] = para['q'].copy()
                para['ifprovince'] = 0
            elif scenario == "RecyclingConsTax":
                para['ryc'] = np.ones((para['I'], para['N'])) * 0.2
                para['tau'] = para['q'].copy()
                para['ifprovince'] = 3
            elif scenario == "RecyclingIndConsTax":
                para['ryc'] = np.ones((para['I'], para['N'])) * 0.2
                para['tau'] = para['q'].copy()
                para['ifprovince'] = 4

            if onlychn:
                para['tau'][0:num_countries, :] = 0
                para['ryc'][0:num_countries, :] = 0

            goal = goals[k]
            scenarioname = scenario + goal
            scenarionames.append(scenarioname)

            constraints[scenarioname], coefficient = create_constraints(para, goal)

            if para['if_cum']:
                bounds_orgin = bounds_orgin_cum.copy()
                num_if_cum += 1
            else:
                bounds_orgin = bounds_orgin_dep.copy()

            ifprovince = para['ifprovince']
            if ifprovince == 1 or ifprovince == 3:
                init = np.ones((regions.ParentRegion.isna().sum() + 1, para['Tmax'])).flatten()  * coefficient
                # init = tax_limit[0:(regions.ParentRegion.isna().sum() + 1) * para['Tmax']]  # 截取tax_limit中的部分元素
                # init[-18:] = tax_limit_chn_temp
                bounds = bounds_orgin[0:(regions.ParentRegion.isna().sum() + 1) * para['Tmax']].copy()
                # bounds[-18:] = tax_limit_chn  # 根据最低省限制tax，否则会出现负收入，会导致无法满足2摄氏度目标
                # nlc = optimize.NonlinearConstraint(con, np.zeros((init.shape[0] * para['N'])), np.inf * np.ones((init.shape[0] * para['N'])))  # 尝试使用数组约束
            elif ifprovince == 2 or ifprovince == -1 or ifprovince == 4:
                init = np.ones((para['I'], para['Tmax'])).flatten() * coefficient
                # init = tax_limit[0:para['I'] * para['Tmax']]  # 截取tax_limit中的部分元素
                bounds = bounds_orgin.copy()
            elif ifprovince == 0:
                init = np.ones((para['I'], para['Tmax'])).flatten() * coefficient
                bounds = bounds_orgin.copy()

            args[scenarioname] = (
                s.copy(), para.copy(), initial.copy(), var.copy(), regions.copy(), objective_function_mu_negishi_china,
                step)

            problem = {
                'scenario': scenarios[n],
                'goal': goals[k],
                'scenarioname': scenarios[n] + goals[k],
                'init': init.copy(),
                'bounds': bounds.copy(),
                # 'constraints': constraints[scenarioname],
                'args': args[scenarioname],
                'start_time': start_time,
                'para': para.copy(),
                'coefficient': coefficient,
            }
            problems.append(problem)

    # num_processes = int(multiprocessing.cpu_count())  # 使用所有可用的CPU核心
    num_processes = 12
    # num_processes = None
    pool = multiprocessing.Pool(processes=num_processes)

    results_temp = pool.imap_unordered(optimize_scenario, problems)
    pool.close()
    pool.join()

    results = {}

    for i, result in enumerate(results_temp):
        scenarioname = result['scenarioname']
        # results[scenarioname] = result
        _problem = result['problem']
        res = result['res']

        resultargument = res.x.reshape((int(res.x.shape[0] / _problem['para']['Tmax']), _problem['para']['Tmax']))
        _, results[scenarioname] = welfare(resultargument, s, _problem['para'], initial, var, regions,
                             objective_function_mu_negishi_china, step=step)

        results[scenarioname]['Status'] = res.status
        results[scenarioname]['Function_value'] = res.fun
        results[scenarioname]['Iteration'] = res.nit
        results[scenarioname]['Runtime'] = result['runtime']
        with open('OutputData/results_status.txt', 'a') as ff:
            # 将运行完成的原因写入文件
            ff.write(f"Scenario: {scenarioname}\n")
            ff.write(f"Success: {res.success}\n")
            ff.write(f"Message: {res.message}\n")
            if goal != "sss":
                ff.write(f"Status: {res.status}\n")
                ff.write(f"Function value: {res.fun}\n")
                ff.write(f"Iteration: {res.nit}\n")
            ff.write("\n")  # 添加一个空行，用于分隔不同的scenario结果

    results = {key: results[key] for key in scenarionames}

    argument = np.zeros((para['I'], para['Tmax']))
    para['tau'] = np.zeros((para['I'], para['N']))
    para['ryc'] = para['tau']
    reswelfare, results['NoMitigation'] = welfare(argument, s, para, initial, var, regions,
                                                  objective_function_mu_negishi_china, step=step)
    runtime = record_program_runtime(start_time)
    results['mu_limit'] = mu_limit
    # 保存结果
    if num_if_cum > 0:
        file_name_2 = "OutputData/0result"+str(regionsnum)+"region_"+name+"_cum" + ".txt"
    else:
        file_name_2 = "OutputData/0result"+str(regionsnum)+"region_"+name + ".txt" # 根据coefficient的不同保存不同的文件
    with open(file_name, 'wb') as f:
        pickle.dump(results, f)
    with open(file_name_2, 'wb') as f:
        pickle.dump(results, f)
    with open('OutputData/results_status.txt', 'a') as f:
        f.write(f"Runtime: {runtime}\n")
        f.write("\n")
    # check_shutdown_time()
    print("finish")
    print(file_name_2+'saved')
    time.sleep(100)
    os.system("shutdown /h")
    # os.system('shutdown -s -t 0')


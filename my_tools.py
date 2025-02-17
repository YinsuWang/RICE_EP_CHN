# Useful tools
import pandas as pd
import re
import time
from datetime import datetime
import numpy as np
import math
import pickle
from RICE_Calculater import *
import os
from scipy.optimize import differential_evolution
import shutil
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

# 提取字典中的每一个dataframe的第一行
def get_first_row(d):
    first_rows = []
    for df in d.values():
        first_rows.append(df.iloc[0])
    return first_rows


# 提取字典中每个为时间序列的DataFrame的共同起始年份
def get_common_start_year(d):
    d = d.copy()
    start_years = []
    for df in d.values():
        first_row = pd.Series(df.columns)
        first_col = df.iloc[:, 0]
        is_year_row = first_row.apply(lambda x: bool(re.search(r'(?<!\d)\d{4}(?!\d)', str(x))))
        is_year_col = first_col.apply(lambda x: bool(re.search(r'(?<!\d)\d{4}(?!\d)', str(x))))
        num_rows, num_cols = df.shape
        if is_year_row.sum() > is_year_col.sum() and is_year_row.sum() > 5:
            # 在df中逐列检测nan的数量，若超过一半，则将is_year_row中对应的位置设为False
            nan_list = df.isna().sum(axis=0).reset_index(drop=True)  # 重置索引
            is_year_row[nan_list > num_cols * 0.6] = False
            year_list = first_row.astype(str).str.extract('(\d{4})')
        elif is_year_col.sum() > 5:
            # 逐行检测nan的数量，若超过一半，则将is_year_col中对应的位置设为False
            nan_list = df.isna().sum(axis=1)
            is_year_col[nan_list > num_rows * 0.6] = False
            year_list = first_col.astype(str).str.extract('(\d{4})')
        else:
            year_list = pd.Series(dtype='float64')

        if is_year_row.sum() > 5 or is_year_col.sum() > 5:
            if np.nanmax(year_list.astype(float)) <= datetime.now().year:
                start_years.append(np.nanmin(year_list.astype(float)))
    start_year = max(start_years)
    return max(start_years)


# Merge regions with pre set country-region list
# 当data中包括了cr之外的国家后，data会比weight长，从而出现超范围的问题,这是数据本身有重复导致的
# 并且当data中不包括year时，会错误的删除weight的所有数据


def merge_region(data, region, weight=None, mergekey='CountryCode', regionkey='Region', option='percapital'):
    '''默认在输入时会输入：年度data+年度weight、年度data+列weight、多列data+列weight、列data+列weight
    对于其他情况，则返回错误、
    同时默认年度已经确定(在common_year中处理)，输入数据中只保留id+data'''
    # copy input data
    data = data.copy()
    region = region.copy()
    weight = weight.copy() if weight is not None else None
    custom_order = region[regionkey].unique().tolist()

    # 如果不输入weight，则默认对数据进行加总
    def customsort(data, custom_order, regionkey):
        data[regionkey] = pd.Categorical(data[regionkey], categories=custom_order, ordered=True)
        return data.sort_values(regionkey)

    if weight is None:
        data_regional = customsort(pd.merge(data, region, how='right', on=mergekey), custom_order, regionkey)
        if option == 'mean':
            data_regional = data_regional.groupby(regionkey).mean()
        else:
            data_regional = data_regional.groupby(regionkey).sum()
        return data_regional
    if option == 'sum' or option == 'total':
        data_regional = customsort(pd.merge(data, region, how='right', on=mergekey), custom_order, regionkey)
        return data_regional.groupby(regionkey).sum()

    # 判断data是否为年度数据
    data_year = data.columns.astype(str).str.extract('(\d{4})')[0]
    if len(data_year.dropna()) == 0:
        isyeardata = False
    else:
        isyeardata = True

    # 判断weight是否为年度数据
    if weight is not None:
        weight_year = weight.columns.astype(str).str.extract('(\d{4})')[0]
        if len(weight_year.dropna()) == 0:
            isyearweight = False
        else:
            isyearweight = True
    else:
        isyearweight = False

    # 判断data哪些列为数据以及数据的列数
    data_list = pd.DataFrame(index=range(len(data.columns)))
    data_list.loc[:, 0] = data.dtypes.values
    # 将year_data_list中非数值的列设为nan
    data_list.loc[0, ~data_list.iloc[0].isin([np.dtype('int64'), np.dtype('float64')])] = np.nan
    data_list.loc[:, 0] = data_list.loc[:, 0].where(data_list.loc[:, 0].isna(), data.columns)
    data_long = len(data_list.dropna())

    weight_list = pd.DataFrame(index=range(len(weight.columns)))
    weight_list.loc[:, 0] = weight.dtypes.values
    weight_list.loc[0, ~weight_list.iloc[0].isin([np.dtype('int64'), np.dtype('float64')])] = np.nan
    weight_list.loc[:, 0] = weight_list.loc[:, 0].where(weight_list.loc[:, 0].isna(), weight.columns)
    weight_long = len(weight_list.dropna())

    # 若data与weight均为年度数据，判断data与weight是否大小一致
    if isyeardata and isyearweight and weight_long > 1:
        if data_long != weight_long:
            raise ValueError('data and weight should have the same number of columns')

    # 若weight不是年度数据且weight的长度大于1，则返回错误
    if not isyearweight and weight_long > 1:
        raise ValueError('weight should be a column or a year data')

    # 若weight不是年度数据且data的长度大于1，则将weight补充至与data一样的长度
    if not isyearweight and data_long > 1:
        for i in range(data_long - 1):
            weight = weight.join(weight.iloc[:, -1], rsuffix=f'_{i}')

    # 若data非年度数据而weight为年度数据或者weight的长度大于data，则令weight等于最后一期的weight
    if not isyeardata and (isyearweight or weight_long > data_long):
        weight = weight.iloc[:, [0, -1]]
        for i in range(data_long - 1):
            weight = weight.join(weight.iloc[:, -1], rsuffix=f'_{i}')

    # 加权平均合并,只要输入weight，就执行加权平均
    data_regional = pd.merge(data, region, how='right', on=mergekey)
    data_regional = data_regional.dropna(subset=[regionkey])
    _, wide = data_regional.shape
    weight_regional = pd.merge(weight, region, how='right', on=mergekey)
    data_regional = pd.merge(data_regional, weight, how='left', on=mergekey).dropna()
    data_regional = customsort(data_regional, custom_order, regionkey)
    # 根据regionkey依次加权平均
    data_regional = data_regional.groupby(regionkey).apply(
        lambda x: pd.Series(np.sum(x.iloc[:, 1:wide - 1].values *
                                   x.iloc[:, wide:].values, axis=0) /
                            np.sum(x.iloc[:, wide:].values, axis=0)))
    data_regional = data_regional.rename(columns=dict(zip(data_regional.columns, data.columns[1:])))
    return data_regional


def common_year(data1, data2=None, exyear=None):
    # 创建两个DataFrame的副本
    data1 = data1.copy()
    # 提取两个DataFrame中的年份
    years1 = (data1.columns.astype(str).str.extract('(\d{4})')[0])
    if data2 is not None:
        data2 = data2.copy()
        years2 = (data2.columns.astype(str).str.extract('(\d{4})')[0])

    # 根据exyear参数确定要保留的年份
    if exyear is not None:
        if type(exyear) is int:
            exyear = [exyear]
        common_years = (exyear)
        # 将common_years转换成str类型
        common_years = ([str(year) for year in common_years])
    elif data2 is not None:
        common_years = years1 & years2

    # 保留两个DataFrame中的第一列和common_years中的年份
    data1 = data1.iloc[:, [0] + [i for i in range(1, len(data1.columns)) if str(years1[i]) in str(common_years)]]
    # data1 = data1.iloc[:, [0] + [col for col in years1 if str(col) in str(common_years)]]
    if data2 is not None:
        data2 = data2.iloc[:, [0] + [i for i in range(1, len(data2.columns)) if str(years2[i]) in str(common_years)]]
        return data1, data2
    else:
        return data1


def multiply_matrices(cpre, q):
    """
    将一个 I*T 的矩阵和一个 I*N 的矩阵按元素相乘，返回一个 I*T*N 的三维矩阵。
    """
    # 获取矩阵的维度
    i, t = cpre.shape
    n = q.shape[1]

    # 将 cpre 和 q 扩展为 I*T*N 和 I*N*T 的三维矩阵
    cpre_3d = np.tile(cpre, (n, 1, 1)).transpose(1, 2, 0)
    q_3d = np.tile(q, (t, 1, 1)).transpose(1, 0, 2)

    # 对 cpre_3d 和 q_3d 沿着第一个维度（即行）进行按元素相乘
    result_3d = np.multiply(cpre_3d, q_3d)

    return result_3d


def times(m1, m2):
    """
    将一个 I*T 的矩阵和一个 I*N 的矩阵按元素相乘，返回一个 I*T*N 的三维矩阵。
    """
    i, t = m1.shape
    i, n = m2.shape
    result = np.zeros((i, t, n))
    for i in range(i):
        result[i, :, :] = np.matmul(m1[i, :, np.newaxis], m2[i, :, np.newaxis].T)
    return result


def stepvar(var, step, tmax):
    if step != 1:
        timelist = np.arange(0, tmax * step, step)

        # 保留var中的每个Dataframe中对应para['T']的列
        for key in var.keys():
            # 如果var[key]是一个np.array且维度为2，则保留对应的列；如果维度为3，则保留对应的第二个维度
            if isinstance(var[key], np.ndarray) and var[key].ndim == 2:
                var[key] = var[key][:, timelist]
            elif isinstance(var[key], np.ndarray) and var[key].ndim == 3:
                var[key] = var[key][:, timelist, :]
            elif isinstance(var[key], np.ndarray) and var[key].ndim == 1:
                var[key] = var[key][timelist]

    return var


def record_program_runtime(_start_time):
    end_time = time.time()
    _runtime = end_time - _start_time
    print("程序运行时间: %.2f 秒" % _runtime)
    return _runtime


def check_shutdown_time():
    current_time = time.localtime()
    if 2 <= current_time.tm_hour <= 8:
        print("当前时间超过凌晨2点，即将关机...")
        # 使用不同操作系统的关机命令
        if os.name == "posix":  # Linux、Mac等类Unix系统
            os.system("sudo shutdown -h now")
        elif os.name == "nt":  # Windows系统
            os.system("shutdown /h")


def customsort(data, custom_order, regionkey):
    data[regionkey] = pd.Categorical(data[regionkey], categories=custom_order, ordered=True)
    return data.sort_values(regionkey)


def inputdata(set_p, new_data=True):

    if new_data is True:
        shutil.copy2(set_p['parafile'], 'Inputdata/Parameters.xlsx')
        shutil.copy2(set_p['temperaturefile'], 'Inputdata/TmeanandTdiff.dta')
        shutil.copy2(set_p['regionfile'], 'Inputdata/Region.xlsx')
        # Import paramaters and initial data
        para, initial, var, regions = readdata(set_p)
        # 将para, initial, var, regions存储为在一个文件中
        para['Tmax'] = int((para['Tend'] - para['Tstart'] + 1) / para['step']) + 1
        # 遍历para initial var regions，将其中的dataframe转化为numpy array
        for key in para.keys():
            if isinstance(para[key], pd.DataFrame):
                para[key] = para[key].values
        for key in initial.keys():
            if isinstance(initial[key], pd.DataFrame):
                initial[key] = initial[key].values
        for key in var.keys():
            if isinstance(var[key], pd.DataFrame):
                var[key] = var[key].values

        var = stepvar(var, para['step'], para['Tmax'])

        with open('Inputdata/InputData.txt', 'wb') as f:
            pickle.dump(para, f)
            pickle.dump(initial, f)
            pickle.dump(var, f)
            pickle.dump(regions, f)

        shutil.copy2('Inputdata/InputData.txt', set_p['inputfile'])

            # 注意Tstart = 2015, Tstart 为第0期

    else:
        with open(set_p['inputfile'], 'rb') as f:
            para = pickle.load(f)
            initial = pickle.load(f)
            var = pickle.load(f)
            regions = pickle.load(f)

        # para.update(set_p)
    return para, initial, var, regions


def readdata(para):
    cr = pd.read_excel('Inputdata/Region.xlsx', sheet_name='Sheet1')
    weight = pd.read_excel('Inputdata/InitialData.xlsx', sheet_name='L', header=0)
    weight = common_year(weight, exyear=para['Tstart'])
    # regions = cr['Region'].unique()
    regions = pd.read_excel('Inputdata/Region.xlsx', sheet_name='ParentRegion', header=0)
    para.update({'regions': regions, 'I': len(regions)})
    # Input Temperature
    if para['new_data']:
        temperature = pd.read_stata('Inputdata/TmeanandTdiff.dta')  # 创建新的 dataframe，只包含 Tmean 相关的列
        temperature.columns = temperature.columns.str.replace('_', ' ')  # 将列名中的下划线替换为空格
        temperature = temperature.set_index('year')
        tmean = temperature.filter(regex='Tmean')  # 重命名列名，将 'Tmean' 替换为 'mean'
        tmean = tmean.rename(columns=lambda x: x.replace('Tmean', ''))  # 创建新的 dataframe，只包含 Tdiff 相关的列
        tmean = tmean.iloc[:, :-1]
        tdiff = temperature.filter(regex='Tdiff')  # 重命名列名，将 'Tdiff' 替换为 'diff'
        tdiff = tdiff.rename(columns=lambda x: x.replace('Tdiff', ''))
        tmin = temperature.filter(regex='Tmin')
        tmin = tmin.rename(columns=lambda x: x.replace('Tmin', ''))
        tmax = temperature.filter(regex='Tmax')
        tmax = tmax.rename(columns=lambda x: x.replace('Tmax', ''))
        tmin_var = temperature.filter(regex='Tvarmin')
        tmin_var = tmin_var.rename(columns=lambda x: x.replace('Tvarmin', ''))
        tmax_var = temperature.filter(regex='Tvarmax')
        tmax_var = tmax_var.rename(columns=lambda x: x.replace('Tvarmax', ''))
        avert = temperature.filter(regex='world_Tmean')
        with open('Inputdata/temperature.txt', 'wb') as f:
            pickle.dump(tmean, f)
            pickle.dump(tdiff, f)
            pickle.dump(avert, f)
            pickle.dump(tmin, f)
            pickle.dump(tmax, f)
            pickle.dump(tmin_var, f)
            pickle.dump(tmax_var, f)
    else:
        with open('Inputdata/temperature.txt', 'rb') as f:
            tmean = pickle.load(f)
            tdiff = pickle.load(f)
            avert = pickle.load(f)
            tmin = pickle.load(f)
            tmax = pickle.load(f)
            tmin_var = pickle.load(f)
            tmax_var = pickle.load(f)

    # Input Energy Price
    # Energy price 这里用的好像是最新的，可以考虑换成2015的
    if para['new_data']:
        price = pd.DataFrame()
        energy_price = pd.read_excel('Inputdata/EnergyPrice.xlsx', sheet_name='Trillion US.dollar per pj')
        energy_use = pd.read_csv(
            'Inputdata/VariousEnergyConsumptionBySector.csv')  # Read energy consumption data from CSV file
        energy_use = energy_use.loc[energy_use['Sector'] == 'Residential']  # 保留 energy_use中'Sector' 列为 'Residential' 的行
        energy_use = energy_use.loc[(energy_use['Year'] == 2019)]
        energy_use.iloc[:, 4:8] = energy_use.iloc[:, 4:8].fillna(0)  # 将energy_use中值为nan的元素替换为0
        # 将energy_use energy_price 和 cr 进行合并
        energy = pd.merge(energy_price, energy_use, on='CountryCode', how='outer')  # on 值可能不一定是 CountryCode
        energy = pd.merge(energy, cr, on='CountryCode', how='inner')
        energy.dropna(inplace=True)  # 删除 energy 中值为nan的行
        # 分组对energy做加权平均，权重为energy中对应energy_use的值，数据为energy中对应energy_price的值

        custom_order = regions['Region'].unique().tolist()
        energy = customsort(energy, custom_order, 'Region')
        price['EnergyPriceIndex'] = energy.groupby('Region').apply(
            lambda x: pd.Series(np.sum(x.iloc[:, 1:5].values.flatten() * x.iloc[:, 8:12].values.flatten(), axis=0) /
                                np.sum(x.iloc[:, 8:12].values.flatten())))
        # 电价格等于能源价格中的‘CountryCode’列和‘ElcPrice’列；电消耗等于energy_use中的‘CountryCode’列和‘Elec_TJ’列
        electricity_price = energy_price[['CountryCode', 'ElcPrice']]
        electricity_use = energy_use[['CountryCode', 'Elecrt_TJ']]

        electricity = pd.merge(electricity_price, electricity_use, on='CountryCode', how='outer')
        electricity = pd.merge(electricity, cr, on='CountryCode', how='inner')
        electricity.dropna(inplace=True)
        electricity = customsort(electricity, custom_order, 'Region')
        price['ElectricityPriceIndex'] = electricity.groupby('Region').apply(
            lambda x: pd.Series(np.sum(x.iloc[:, 1].values.flatten()
                                       * x.iloc[:, 2].values.flatten(), axis=0) / np.sum(
                x.iloc[:, 2].values.flatten())))

        with open('Inputdata/price.txt', 'wb') as f:
            pickle.dump(price, f)
    else:
        with open('Inputdata/price.txt', 'rb') as f:
            price = pickle.load(f)

    para['tmean'] = tmean.T[para['Tstart'] + 1].reindex(regions['Region'].unique().tolist())
    para['tdel'] = tdiff.T[para['Tstart'] + 1].reindex(regions['Region'].unique().tolist())
    para['temmin'] = tmin.T[para['Tstart'] + 1].reindex(regions['Region'].unique().tolist())
    para['temmax'] = tmax.T[para['Tstart'] + 1].reindex(regions['Region'].unique().tolist())
    para['AverT'] = avert.T[para['Tstart'] + 1].reindex(regions['Region'].unique().tolist())
    para['tmin_var'] = tmin_var.T[para['Tstart'] + 1].reindex(regions['Region'].unique().tolist())
    para['tmax_var'] = tmax_var.T[para['Tstart'] + 1].reindex(regions['Region'].unique().tolist())
    para['price'] = price

    # Input Parameters
    df = pd.read_excel('Inputdata/Parameters.xlsx', sheet_name=None, header=None)
    metadata = pd.read_excel('Inputdata/Parameters.xlsx', sheet_name='Metadata')
    temp_para = {}
    for i in range(metadata.shape[0]):
        name = metadata.loc[i, 'Parameter']
        ismatric = metadata.loc[i, 'metric']
        if ismatric == 1:
            temp_para[name] = df[name]
            if isinstance(temp_para[name].iloc[0, 0], str):
                temp_para[name].columns = temp_para[name].iloc[0]
                temp_para[name] = temp_para[name].iloc[1:]
                temp_para[name] = temp_para[name].reset_index(drop=True)
        else:
            temp_para[name] = metadata.loc[i, 'value']
    para.update(temp_para)

    # Input Initial Data
    metadata = pd.read_excel('Inputdata/InitialData.xlsx', sheet_name='Metadata')
    df = pd.read_excel('Inputdata/InitialData.xlsx', sheet_name=None, header=None)  # header=None表示第一行不是列名
    initial = {}
    for i in range(metadata.shape[0]):
        name = metadata.loc[i, 'Data']
        ismatric = metadata.loc[i, 'metric']
        isvalue = metadata.loc[i, 'value']
        if ismatric == 1:
            initial[name] = df[name]
            first_row_type = type(initial[name].iloc[0, 0])
            if not (first_row_type == initial[name].dtypes).all():
                initial[name].columns = initial[name].iloc[0]
                initial[name] = initial[name].drop(initial[name].index[0])
                initial[name] = initial[name].reset_index(drop=True)
        # 如果isvalue不是空值，那么就把isvalue赋值给initial[name]
        if not math.isnan(isvalue):
            initial[name] = isvalue
    initial['Ltol'] = pd.merge(initial['L'], initial['WPP2022_medium'], how='inner', on='CountryCode')
    initial['CO2'] = initial['CO2'].T
    initial['CO2pre'] = initial['CO2pre'].T

    # Calculate initial data
    #  获取initial的起始年份
    initial = {k: v for k, v in initial.items() if not (isinstance(v, float) and math.isnan(v))}
    initial_dataframe = {k: v for k, v in initial.items() if isinstance(v, pd.DataFrame)}
    tstart = max(para['Tstart'], get_common_start_year(initial_dataframe))

    period = [p for p in range(para['Tstart'] - para['TA'] + 1, para['Tstart'] + 1)]  # 用于估计参数的时间段

    # 合并地区部分数据用于计算全要素生产率a
    temp_l = common_year(initial['L'], exyear=period)  # 临时数据用于计算l
    l_for_a = merge_region(temp_l, cr, option='total')
    temp_k, temp_w = common_year(initial['K'], initial['L'], period)  # 临时数据用于计算k
    k_for_a = merge_region(temp_k, cr, weight=temp_w,
                           option='percapital') * l_for_a  # index is Region and column is year
    temp_q = common_year(initial['Q'], exyear=period)  # 临时数据用于计算q
    q_for_a = merge_region(temp_q, cr, weight=temp_w, option='percapital') * l_for_a
    temp_a = q_for_a / (l_for_a ** (1 - para['alpha']) * k_for_a ** para['alpha'])  # 这个计算出来有点奇怪
    initial['A'] = temp_a.iloc[:, -1]
    temp_e0 = common_year(initial['E0'], exyear=period)  # 临时数据用于计算e0
    e0_for_sigma = merge_region(temp_e0, cr, weight=temp_w, option='percapital') * l_for_a
    temp_sigma = e0_for_sigma / q_for_a

    # Merge data with regions
    # 读取matadata,如果'type'是'total',那么就在运行merge_region时指定option='total'
    # 如果'type'是'percapital',那么就在运行merge_region时指定option='percapital'
    weight = initial['L'].copy()  # 可能要修改merge_region以满足当weight的年份小于data单足够长时，以weight的年份为准，或者在无weight数据的年份用最一年的weight
    # 这里可能需要用common_year处理一下
    weight = common_year(weight, exyear=tstart)
    for i in np.arange(0, metadata.shape[1 - 1]).reshape(-1):
        if metadata.loc[i, 'type'] == 'total':
            initial[metadata.loc[i, 'Data']] = common_year(initial[metadata.loc[i, 'Data']], exyear=tstart)
            initial[metadata.loc[i, 'Data']] = merge_region(initial[metadata.loc[i, 'Data']], cr, option='total')
        elif metadata.loc[i, 'type'] == 'percapital':
            initial[metadata.loc[i, 'Data']] = common_year(initial[metadata.loc[i, 'Data']], exyear=tstart)
            initial[metadata.loc[i, 'Data']] = merge_region(initial[metadata.loc[i, 'Data']], cr, weight=weight,
                                                            option='percapital')

    # 人均资本存量→资本存量
    # 注意这里的处理，initial['K']可能是储存在Dataframe中而不是np array中
    initial['Kpc'] = initial['K'] * 10 ** 6  # trillion→million
    initial['K'] = initial['K'] * initial['L']

    # 人均GDP→GDP
    initial['Qpc'] = initial['Q'] * 10 ** 6
    initial['Q'] = initial['Q'] * initial['L']
    # 人均排放→总排放
    initial['E0'] = initial['E0'] * 10 ** 3  # ktons CO2 per capital 这里就是10^3
    initial['E0'] = initial['E0'] * initial['L']  # GtC
    # 上述L单位为million，即求得的q单位为 e+18美元，结果*e+6单位为万亿
    # Carbon intensity
    initial['sigma'] = initial['E0'] / initial['Q']  # GtC per million USD
    # Generate parameters
    # Population decline rate

    if para['Tpop'] != 0:
        pop = pd.read_excel('Inputdata/InitialData.xlsx', sheet_name='L', header=0)
        pop = common_year(pop, exyear=period)
        pop = merge_region(pop, cr, weight=weight, mergekey='CountryCode', regionkey='Region', option='total')
        para['gpop'], para['delta_pop'] = calibrate(pop, [(1e-07, 1), (1e-07, 1)], growthfunction=growth_tfp)

    # 这里发现矩阵的方向可能不对。在matlab中index表示年份，column表示地区，而在python中index表示地区，column表示年份。
    # 可能是在merge_region时出现了问题
    # TFP decline rate
    # 如果para存在且para['gA']存在且为np array 或者pd DataFrame
    chn_start = regions.ParentRegion.isna().sum()
    # del para['gA'], para['deltaA'], para['gsigma'], para['deltasigma']
    if 'gA' in para and isinstance(para['gA'], (int, float, np.ndarray, pd.DataFrame)):
        if len(para['gA'].shape)>1:
            if isinstance(para['gA'], pd.DataFrame):
                para['gA'] = para['gA'].iloc[:, 0].values
            elif isinstance(para['gA'], np.ndarray):
                para['gA'] = para['gA'][:, 0]
        else:
            pass
    # 如果gA的二维且第二个维度长度为1，那么就将gA的第二个维度删除
    else:
        # 将temp_a[10:,:]加权平均，权重为l_for_a.iloc[10:,:]，然后再校准
        temp_cal = temp_a.copy()
        # temp_cal.iloc[chn_start:,:] = np.average(temp_a.iloc[chn_start:, :], weights=l_for_a.iloc[chn_start:, :], axis=0)
        para['gA'], para['deltaA'] = calibrate(temp_cal, 1, 0.001, growthfunction=growth_tfp, chn_start=chn_start)
        para['gA'] = np.append(para['gA'], np.tile(para['gA'][-1], para['I'] - len(para['gA'])))
        # para['gA'][chn_start:] = np.mean(para['gA'][chn_start:], axis=0)
    initial['TempA'] = temp_a
    adeltaA = para['deltaA']
    agA = para['gA']
    # 校准sigma
    if 'gsigma' in para and isinstance(para['gsigma'], (int, float, np.ndarray, pd.DataFrame)):
        if len(para['gsigma'].shape)>1:
            if isinstance(para['gsigma'], pd.DataFrame):
                para['gsigma'] = para['gsigma'].iloc[:, 0].values
            elif isinstance(para['gsigma'], np.ndarray):
                para['gsigma'] = para['gsigma'][:, 0]
        else:
            pass
    else:
        # 将temp_sigma[10:,:]加权平均，权重为l_for_a.iloc[10:,:]，然后再校准
        temp_cal = temp_sigma.copy()
        # temp_cal.iloc[chn_start:,:] = np.average(temp_sigma.iloc[chn_start:,:], weights=l_for_a.iloc[chn_start:,:], axis=0)
        para['gsigma'], para['deltasigma'] = calibrate(temp_cal, -1, -0.001, growthfunction=growth_sigma, chn_start=chn_start)
        para['gsigma'] = np.append(para['gsigma'], np.tile(para['gsigma'][-1], para['I'] - len(para['gsigma'])))
        # para['gsigma'][chn_start:] = np.mean(para['gsigma'][chn_start:,], axis=0)
    agsigma = para['gsigma']
    adeltasigma = para['deltasigma']
    # q: GDP distribution
    para['q'] = merge_region(para['q'], cr, weight=weight, option='percapital')
    # sigma: carbon intensity
    para['sigma'] = initial['E0'] / initial['Q']
    # 碳转换矩阵,参考DICE2016数据
    # 五年转换的比例和一年转换的比例应该是一致的
    phi = np.zeros((3, 3))
    phi[0, 1] = para['phi'].loc[0, 1]
    # phi[0, 1] = (1 + phi[0, 1])**(1/5) - 1
    phi[1, 2] = para['phi'].loc[1, 2]
    # phi[1, 2] = (1 + phi[1, 2])**(1/5) - 1
    phi[0, 0] = 1 - phi[0, 1]
    phi[1, 0] = phi[0, 1] * initial['CO2pre'].loc['AT', 0] / initial['CO2pre'].loc['UP', 0]
    phi[1, 1] = 1 - phi[1, 0] - phi[1, 2]
    phi[2, 1] = phi[1, 2] * initial['CO2pre'].loc['UP', 0] / initial['CO2pre'].loc['LO', 0]
    phi[2, 2] = 1 - phi[2, 1]
    # para['phi'] = phi
    # 修正TemData
    para['t'] = para['tdel'] - (initial['T'] + initial['Tpre']) + para['AverT']
    # CO2 increasing
    # initial['CO2'] = initial['CO2'] - initial['CO2pre'];
    # para.update(temp_para)
    # return para, initial
    var={}
    var['A'] =growth_tfp(initial['A'], para['gA'], para['deltaA'], para['Tmax'])
    var['A'].columns = [p for p in range(para['Tstart'], para['Tend'] + 1)]  # 从t[0]=2015开始
    var['sigma'] = growth_sigma(initial['sigma'], para['gsigma'], para['deltasigma'], para['Tmax'])  #
    basic_consumption = np.zeros((para['I'], para['Tmax'], para['N']))

    # 获取外生的人口或者根据既往趋势估计人口
    if para['Tpop'] == 0:
        # var['Ltol'] = pd.merge(temp_l, initial['WPP2022_medium'], how='inner', on='CountryCode')
        period = [p for p in range(para['Tstart'], para['Tend'] + 1)]  # 用于估计参数的时间段
        var['Ltol'] = common_year(initial['Ltol'], exyear=period)
        var['Ltol'] = merge_region(var['Ltol'], cr, weight=weight, mergekey='CountryCode', regionkey='Region',
                                   option='total')
        # 生成一个三维的矩阵，将var['Ltol']分成五个部分，每个部分的维度为（para['Tmax']（年份呢）, para['N']（人口组）, para['I']（区域））
        # 将var['Ltol']除以5再复制五次，然后生成一个三维的矩阵，每个部分的维度为（para['Tmax']（年份呢）, para['N']（人口组）, para['I']（区域））
    else:
        var['Ltol'] = growth_tfp(initial['L'], para['gpop'], para['delta_pop'], para['Tmax'] + 1)

    var['L'] = var['Ltol'].to_numpy() / 5
    var['L'] = np.repeat(var['L'][:, :, np.newaxis], 5, axis=2)

    # CO2 emissions from land-use changes
    var['El'] = initial['CO2Land']
    # 获取var['El']中'year'为para['Tstart']、列名为'mean'的数据，并加总
    el0 = var['El'][var['El']['year'] == para['Tstart']]['mean'].sum()
    var['El'] = (el0 * (1 - para['delL']) ** [t for t in range(0, para['Tmax'])])

    # 根据第0期和最后一期的O设定每一年的O，这里O是一个T*1的数组，假定O逐年平均增长
    var['O'] = np.linspace(initial['O'].iloc[0, 1], initial['O'].iloc[1, 1], para['Tmax'] + 1)
    var['O'] = var['O'][1:]

    # calculate backstop price
    p0 = para['RL'].iloc[:, 1] * initial['pw']
    pb = np.zeros((para['I'], para['Tmax']))
    pb[:, 0] = p0
    for i in range(para['I']):
        for t in range(1, para['Tmax']):
            pb[i, t] = para['Th'] * p0[i] + (1 - para['du']) ** para['step'] * (pb[i, t - 1] - para['Th'] * p0[i])
        if para['period'] - para['Tstart'] + 1 < para['Tmax']:
            for t in range(int(para['period'] - para['Tstart'] + 2), para['Tmax']):
                pb[i, t] = pb[i, t - 1] * ((1 - para['dd']) ** para['step'])
    var['pb'] = pb
    var['cb'] = np.zeros((para['I'], para['Tmax'], para['N']))
    var['theta1'] = pb * var['sigma'] / para['theta'].iloc[1, 0]
    var['tdel'] = np.repeat(para['tdel'].to_numpy().reshape(-1, 1), para['Tmax'], axis=1)
    var['tmean'] = np.repeat(para['tmean'].to_numpy().reshape(-1, 1), para['Tmax'], axis=1)
    var['temmin'] = np.repeat(para['temmin'].to_numpy().reshape(-1, 1), para['Tmax'], axis=1)
    var['temmax'] = np.repeat(para['temmax'].to_numpy().reshape(-1, 1), para['Tmax'], axis=1)
    var['tmin_var'] = np.repeat(para['tmin_var'].to_numpy().reshape(-1, 1), para['Tmax'], axis=1)
    var['tmax_var'] = np.repeat(para['tmax_var'].to_numpy().reshape(-1, 1), para['Tmax'], axis=1)

    return para, initial, var, regions


def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def closest_string(input_string, string_list):
    min_distance = float('inf')  # 初始设定一个无穷大的距离
    closest_str = None

    for string in string_list:
        distance = edit_distance(input_string, string)
        if distance < min_distance:
            min_distance = distance
            closest_str = string

    return closest_str

def edit_distance(str1, str2):
    len_str1 = len(str1)
    len_str2 = len(str2)

    # 创建一个二维数组来存储编辑距离
    dp = [[0] * (len_str2 + 1) for _ in range(len_str1 + 1)]

    # 初始化第一行和第一列
    for i in range(len_str1 + 1):
        dp[i][0] = i
    for j in range(len_str2 + 1):
        dp[0][j] = j

    # 计算编辑距离
    for i in range(1, len_str1 + 1):
        for j in range(1, len_str2 + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j - 1], dp[i][j - 1], dp[i - 1][j])

    return dp[len_str1][len_str2]
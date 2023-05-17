import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline, PchipInterpolator, CubicHermiteSpline
from sklearn.model_selection import train_test_split

def get_timediff(col, data, lesseq = 10):
    #drop missing values and create 1 column df
    new_data = pd.DataFrame(data[col].dropna()) 
    #add col showing time difference in minutes
    new_data['timediff_in_min'] = np.concatenate( [ (new_data.index[1:] - new_data.index[:-1])/np.timedelta64(1, 'm'), [np.nan] ] )
    #add boolean column where value is False when the time difference is greater than lesseq minutes
    new_data['timediff_lesseq_10'] = new_data.loc[:, 'timediff_in_min'] <= lesseq
    #initialize column which will be used to store the indexes where new_data['timediff_lesseq_10'] is False
    new_data['ind_where_False'] = np.arange(new_data.shape[0])
    #make values np.nan where new_data['timediff_lesseq_10'] <= 10
    new_data.iloc[[ind for ind, truth_val in enumerate(new_data.timediff_lesseq_10) if truth_val], -1] = np.nan
    return new_data

def num_two_obs_per_minute(data, col):
    new_data = pd.DataFrame(data[col].dropna())
    return (new_data.groupby(by = [new_data.index.year, new_data.index.month, new_data.index.day, new_data.index.hour, new_data.index.minute]).size() > 1).sum()

def get_subset(data, end_ind, start_ind = None):
    #Function designed to get a subset of a 1 column df
    if start_ind == None: #if start_ind is not given, assume startig index is zero
        #Take the mean for each minute window
        pm_subset = data.iloc[:end_ind, :1].groupby(by = [data.iloc[:end_ind, 0].index.year, data.iloc[:end_ind, 0].index.month, 
                                                          data.iloc[:end_ind, 0].index.day, data.iloc[:end_ind, 0].index.hour, 
                                                          data.iloc[:end_ind, 0].index.minute]).mean()
    else: #if start_ind is given
        pm_subset = data.iloc[start_ind:end_ind, :1].groupby(by = [data.iloc[start_ind:end_ind, 0].index.year, data.iloc[start_ind:end_ind, 0].index.month,
                                                                   data.iloc[start_ind:end_ind, 0].index.day, data.iloc[start_ind:end_ind, 0].index.hour,
                                                                   data.iloc[start_ind:end_ind, 0].index.minute]).mean()
    #Here we assume that data has dtype=datetime64[ns] and follows the said format = '%Y%m%d%H%M'
    pm_subset.index = pd.to_datetime(pm_subset.index.get_level_values(0).to_series().apply(str).reset_index(drop =True) + \
    np.array(['0'+n if len(n) == 1 else n for n in pm_subset.index.get_level_values(1).to_series().apply(str).reset_index(drop = True)]) +\
    np.array(['0'+n if len(n) == 1 else n for n in pm_subset.index.get_level_values(2).to_series().apply(str).reset_index(drop = True)]) +\
    np.array(['0'+n if len(n) == 1 else n for n in pm_subset.index.get_level_values(3).to_series().apply(str).reset_index(drop = True)]) + \
    np.array(['0'+n if len(n) == 1 else n for n in pm_subset.index.get_level_values(4).to_series().apply(str).reset_index(drop = True)]), \
    format = '%Y%m%d%H%M')
    return pm_subset

def sep_by_ind(data, start_to_first_ind = False): 
    #Function to automate getting subsets of df and appending to a list
    #data must contain the indexes in its last column
    subsets_list = []
    ind_list = data.iloc[:,-1].dropna().apply(int)
    if start_to_first_ind == True: #if we want to include ind_list[0] as end_ind and start_ind as the zeroeth index
        subsets_list.append(get_subset(data = data, end_ind = ind_list[0]+1, start_ind = None) )
    else:
        pass
    for start, end in zip(ind_list[:-1], ind_list[1:]): #we add 1 to the ind_list indexes because of the way they were created and how indexing works
        subsets_list.append(get_subset(data = data, start_ind = start+1, end_ind = end+1))
    return subsets_list

def include_missing_dates(data, freq = 'T'):
    index = pd.date_range(data.index[0], data.index[-1], freq = freq) #frequency is in minutes
    new_data = pd.DataFrame()
    new_data.index = index
    for col in data.columns:
        new_data[col] = data[col]
    new_data['ind'] = np.arange(new_data.shape[0])
    return new_data

def prep_for_interp(data, col, lesseq = 10, freq = 'T'):
    new_data = get_timediff(col, data, lesseq = lesseq)
    subsets_list = sep_by_ind(new_data, start_to_first_ind = True)
    with_missing_val = [include_missing_dates(subset, freq= freq) for subset in subsets_list]
    return with_missing_val

def subsets_index_with_missingval(subsets): #get index of subsets with missing values
    for ind, data in enumerate(subsets):
        if data.isnull().sum().sum() != 0:
            print('index:', ind, ', num of missing:', data.isnull().sum().sum())
def index_misisngval_pair(subsets):
    store_here = []
    for ind, data in enumerate(subsets):
        store_here.append(data.isnull().sum().sum() )
    new_df = pd.DataFrame()
    new_df['missing_val'] = store_here
    return new_df.iloc[:,0]

def max_gap_per_subset(subsets): #returns the max time gap per subset
    print('\n')
    for ind, data in enumerate(subsets):
        print('index:', ind, ', max time gap: ', ((data.iloc[:,0].dropna().index[1:] - data.iloc[:,0].dropna().index[:-1])/np.timedelta64(1, 'm') ).max() )

def ind_maxgap_pair(subsets):
    store_here = []
    for ind, data in enumerate(subsets):
        store_here.append( ((data.iloc[:,0].dropna().index[1:] - data.iloc[:,0].dropna().index[:-1])/np.timedelta64(1, 'm') ).max() )
    new_df = pd.DataFrame()
    new_df['max_gap'] = np.array(store_here)
    return new_df.iloc[:,0]
def interp(data, interp_func, specify = None):
    new_data = data.copy()
    if new_data.shape[0] == 1:
        return new_data
    else:
        for col in data.iloc[:1].columns:
            if specify is None:
                f = interp_func(data[data[col].notnull()].ind, data[col].dropna())
            else:
                f = interp_func(data[data[col].notnull()].ind, data[col].dropna(), kind = specify)
            interp_data = f(data.ind)
            new_data[col] = interp_data
        return new_data

def interpolate_subsets(data, col, interp_func, specify = None, lesseq = 10, freq = 'T'):
    subset_list = prep_for_interp(data, col, lesseq = lesseq, freq = freq)
    # subsets_index_with_missingval(subset_list) 
    # max_gap_per_subset(subset_list)
    #max time gap per subset
    return [ interp(data, interp_func, specify = specify) for data in subset_list]

#functions for choosing the interpolation method to use

def add_index_col(data):
    new_data = data.copy()
    new_data['ind'] = np.arange(new_data.shape[0])
    return new_data

def split(data, col, train_ratio = 0.6):
    new_data = get_timediff(col, data)
    subsets_list = sep_by_ind(new_data, start_to_first_ind = True)
    subsets_list = [add_index_col(subset) for subset in subsets_list]
    subsets_index_with_missingval(subsets_list) 
    max_gap_per_subset(subsets_list)
    rand_state = 1
    print('random state', rand_state)
    train_list = []
    test_list = []
    for n in range(len(subsets_list)):
        x_train, x_remaining, y_train, y_remaining = train_test_split(subsets_list[n], subsets_list[n], train_size = train_ratio, shuffle = True, random_state= rand_state)
        while  not( (0 in x_train.ind.to_numpy() ) and (subsets_list[n].ind[-1] in x_train.ind.to_numpy() ) ):
            rand_state +=1
            print('random state:', rand_state)
            x_train, x_remaining, y_train, y_remaining = train_test_split(subsets_list[n], subsets_list[n], train_size = train_ratio, shuffle = True, random_state= rand_state)
        train_list.append(x_train.sort_values('ind'))
        test_list.append(x_remaining.sort_values('ind'))
    return train_list, test_list

def interp_test_set(train_list, test_list, interp_func):
    interp_list = []
    for train_subset, test_subset in zip(train_list, test_list):
        f = interp_func(train_subset.ind, train_subset.iloc[:,0])
        interp_data = f(test_subset.ind)
        interp_list.append(interp_data)
    return interp_list


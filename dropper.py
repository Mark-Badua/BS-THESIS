import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

def plot_normalhist(data, col, bins = 50):
    data = data[col].dropna() #drop missing values
    data.hist(bins = bins) #plot a histogram
    print('min: ', data.min()) #print min value
    print('max: ', data.max()) #print max value
    plt.xlabel(col) #x axis label
    plt.ylabel('count') #y axis label
    mean = np.mean(data) #mean
    std = np.std(data) #population standard deviation
    print('Shapiro Wilk Test: \n', stats.shapiro(data.to_numpy() )) #Testing for normality
    print('3* std : ', std*3)
    print('mean : ',  mean)
    print('u + 3sigma:', mean + (3*std))
    print('u - 3sigma', mean - (3*std))
    return mean - (3*std), mean + (3*std) 

def compute_3_sigma_int(data, col, bins = 50): #compute where 
    ''' Return values that have distances of three standard deviations from the mean'''
    data = data[col].dropna() #drop missing values
    print('min: ', data.min())#min value
    print('max: ', data.max())#max
    mean = np.mean(data) # mean
    std = np.std(data)# population standard dev
    print('Shapiro Wilk Test: \n', stats.shapiro(data.to_numpy() )) #Normality Test
    print('3* std : ', std*3)
    print('mean : ',  mean)
    print('u + 3sigma:', mean + (3*std))
    print('u - 3sigma', mean - (3*std))
    return mean - (3*std), mean + (3*std) 

def omit_dates(data, date_list):  #date must be of the form 'yyyy-mm-dd hh:mm:ss'
    dataframe = data.iloc[:,:]
    # print(dataframe.shape)
    for d in date_list:
        date = d.split(' ')[0].split('-')
        time = d.split(' ')[1].split(':')
        dataframe = dataframe[(dataframe.index.year != int(date[0])) | (dataframe.index.month != int(date[1])) 
                              | (dataframe.index.day != int(date[-1])) | (dataframe.index.hour != int(time[0]) )
                              | ( dataframe.index.minute != int(time[1]) ) | ( dataframe.index.second != int(time[-1]) )
                             ]
        # print(dataframe.shape)
    dataframe.sort_index(inplace = True)
    return dataframe

def exclude_dates(data, date_list):  #date must be of the form 'yyyy-mm-dd hh:mm:ss'
    '''Exclude rows with dates in date_list'''
    dataframe = data.iloc[:,:] #copy data
    dataframe['ind'] = np.arange(0, data.shape[0]) #create column of index
    ind = [dataframe[date:date].ind.to_numpy()[0] for date in date_list] #index of rows with dates in date_list
    bool_val = dataframe.ind.apply(lambda x: x in ind) #condition to get rows with dates in date_list
    return dataframe[~bool_val].sort_index() #get subset of data with no dates in date_list

def get_forward_slopes(data, col):
    new_data = pd.DataFrame(data[col].dropna()) #drop missing values
    #next point in time
    new_data['x_n_1'] = np.concatenate( (new_data[col][1:].to_numpy(), [np.nan]) )
    #subtracting next point in time to current point in time
    new_data['timediff_in_min'] = np.concatenate( (np.array( (new_data.index[1:] - new_data.index[:-1])/np.timedelta64(1,'m') ), [np.nan]
                                                  ) 
                                                )
    #compute slope
    new_data['forward_slope'] = ( new_data.x_n_1[:-1] - new_data[col][:-1] )/( new_data.timediff_in_min[:-1])
    #next slope in time
    new_data['fwd_slope_1'] = np.concatenate( (new_data['forward_slope'][1:], [np.nan]
                                              ) 
                                            ) 
    new_data['ind'] = np.arange(0, new_data.shape[0]) #column of index
    return new_data

def get_anomalies(data, col, start_date = '2019-12-31 12', end_date ='2020-01-01 06'):
    new_data = get_forward_slopes(data, col) #data with slopes and time diff
    data_subset = new_data[new_data.timediff_in_min <= 10] #get subset containing timediff <= 10
    # lim = compute_3_sigma_int(data_subset, 'forward_slope') #from the subset, compute for x-3sigma and x+3sgima of the slopes
    # up_lim = lim[1]#upper bound of 3 sigma interval
    # down_lim = lim[0] #lower bound
    
    up_lim = data_subset[start_date: end_date].forward_slope.max() * 5
    down_lim = data_subset[start_date: end_date].forward_slope.min() * 5
    data_subset_1 = data_subset[(data_subset.forward_slope * data_subset.fwd_slope_1) <0] #limit subset to slopes that are alternating positive and negative
    # limit again to points whose slopes are outside the arbitrary boundary set
    data_subset_2 = data_subset_1[(data_subset_1.forward_slope > up_lim) | (data_subset_1.forward_slope < down_lim)]
    #limit to points with only positive slopes 
    data_subset_3 = data_subset_2[data_subset_2.forward_slope >= 0]
    data_subset_4 = new_data[new_data.ind.apply(lambda x: x in (data_subset_3.ind.to_numpy()+1))] #POIs are the next point in time
    date_list = list(data_subset_4.index.astype(str)) #get list of dates of POIs
    #exclude dates in date_list
    final_data = exclude_dates(new_data, date_list).iloc[:, 0]
    return final_data

def get_omitted_dates(data, col, start_date = '2019-12-31 12', end_date ='2020-01-01 06'):
    new_data = get_forward_slopes(data, col) #data with slopes and time diff
    data_subset = new_data[new_data.timediff_in_min <= 10] #get subset containing timediff <= 10
    # lim = compute_3_sigma_int(data_subset, 'forward_slope') #from the subset, compute for x-3sigma and x+3sgima of the slopes
    # up_lim = lim[1]#upper bound of 3 sigma interval
    # down_lim = lim[0] #lower bound
    
    up_lim = data_subset[start_date: end_date].forward_slope.max() * 5
    down_lim = data_subset[start_date: end_date].forward_slope.min() * 5
    data_subset_1 = data_subset[(data_subset.forward_slope * data_subset.fwd_slope_1) <0] #limit subset to slopes that are alternating positive and negative
    # limit again to points whose slopes are outside the arbitrary boundary set
    data_subset_2 = data_subset_1[(data_subset_1.forward_slope > up_lim) | (data_subset_1.forward_slope < down_lim)]
    #limit to points with only positive slopes 
    data_subset_3 = data_subset_2[data_subset_2.forward_slope >= 0]
    data_subset_4 = new_data[new_data.ind.apply(lambda x: x in (data_subset_3.ind.to_numpy()+1))] #POIs are the next point in time
    date_list = list(data_subset_4.index.astype(str)) #get list of dates of POIs
    return data_subset_4

def derivative_hist(data, col, start_date = '2019-12-31 12', end_date ='2020-01-01 06'):
    new_data = get_forward_slopes(data,col)
    data_subset = new_data[new_data.timediff_in_min <= 10]
    # lim = compute_3_sigma_int(data_subset, 'forward_slope')
    # up_lim = lim[1]
    # down_lim = lim[0]
    up_lim = data_subset[start_date: end_date].forward_slope.max()
    down_lim = data_subset[start_date: end_date].forward_slope.min()
    print("minimum rate of change: ",down_lim)
    print("maximum rate of change: ",up_lim)
    return data_subset
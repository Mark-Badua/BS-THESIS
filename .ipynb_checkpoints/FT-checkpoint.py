import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
def create_fourier_df(data, substract_mean = True, freq_converter = None):
    
    new = pd.DataFrame()
    #default freq units is cyles per default unit of time of data
    #if measurements exists per minute => freq = 1 sample per minute
    if freq_converter == None: #number to multiply to obtain frequencies in the desired units
        if substract_mean == True: #subtract DC component, take FT, convert freq units
            for col in data.columns[:1]:
                new['freq'] = np.fft.rfftfreq(
                    n = data[col].shape[0], d= int( (data.index[1] - data.index[0])/np.timedelta64(1, 'm') )
                                             )
                new['abs_fourier_coeff_square'] = np.abs(
                    np.fft.rfft(data[col] - data[col].mean())
                                   )**2
                new['period (in days)'] = (1/new['freq'])/(60*24) #default period is in days
        else: #take FT, convert freq units
            
            for col in data.columns[:1]:
                new['freq'] = np.fft.rfftfreq(n = data[col].shape[0], d=int((data.index[-1] - data.index[0])/np.timedelta64(1, 'm'))/data.shape[0])
                new['amp'] = np.abs(np.fft.rfft(data[col]))**2
                new['period'] = (1/new['freq'])/(60*24) #default period in days
                
    else: #Default frequency is cycles per minute
        if substract_mean == True: #if DC component is not zero, we subtract the mean before FT
            for col in data.columns[:1]:
                new['freq'] = np.fft.rfftfreq(n = data[col].shape[0], d=int((data.index[-1] - data.index[0])/np.timedelta64(1, 'm'))/data.shape[0])
                new['amp'] = np.abs(np.fft.rfft(data[col] - data[col].mean()) )**2
                new['period'] = (1/new['freq'])/(60*24) #default period in days
                new['freq'] = new['freq'] * freq_converter
        else:  #take FT without subtracting DC component
            for col in data.columns[:1]:
                new['freq'] = np.fft.rfftfreq(n = data[col].shape[0], d=int((data.index[-1] - data.index[0])/np.timedelta64(1, 'm'))/data.shape[0])
                new['amp'] = np.abs(np.fft.rfft(data[col]))**2
                new['period'] = (1/new['freq'])/(60*24) #default period in days
                new['freq'] = new['freq'] * freq_converter
    return new

def get_next_power2(num):
    return pow(2, int(np.log2(num)) + 1)

def normalize(data, col):
    sum = data[col].sum()
    new_col = data[col].apply(lambda x: x/sum)
    new_data = data.copy()
    new_data[col] = new_col
    return new_data

def FT_with_zp(data, substract_mean = True, freq_converter = None, complex_FT = False):
    new_len = get_next_power2(data.shape[0]) 
    orig_duration = ((data.index[-1] - data.index[0])/np.timedelta64(1, 'm'))
    f_s = data.shape[0] / orig_duration #samples/total duration in minutes
    f_zp = np.arange(0, f_s/2, f_s/new_len)
    n_diff = new_len - data.shape[0]
    #zero pad until length of samples is the next higher power of 2 
    new = pd.DataFrame()
    #default freq units is cyles per minutes
    if freq_converter == None: #number to multiply to obtain frequencies in the desired units
        if substract_mean == True: #subtract DC component, take FT, convert freq units
            for col in data.columns[:1]:
                new['freq'] = np.fft.rfftfreq(new_len, 1/f_s)
                new['mag_squared'] = np.abs(np.fft.rfft(data[col] - data[col].mean(), n = new_len))**2
                new['period'] = (1/new['freq'])/(60*24) #default period is in days
        else: #take FT, convert freq units
            
            for col in data.columns[:1]:
                new['freq'] = np.fft.rfftfreq(new_len, 1/f_s)
                new['mag_squared'] = np.abs(np.fft.rfft(data[col], n = new_len))**2
                new['period'] = (1/new['freq'])/(60*24) #default period in days
    
    elif complex_FT == True:
        return np.fft.rfft(data[col] - data[col].mean(), n = new_len)
    else: #Default frequency is cycles per minute
        if substract_mean == True: #if DC component is not zero, we subtract the mean before FT
            for col in data.columns[:1]:
                new['freq'] = np.fft.rfftfreq(new_len, 1/f_s)
                new['mag_squared'] = np.abs(np.fft.rfft(data[col] - data[col].mean(), n = new_len))**2
                new['period'] = (1/new['freq'])/(60*24) #default period in days
                new['freq'] = new['freq'] * freq_converter
        else:  #take FT without subtracting DC component
            for col in data.columns[:1]:
                new['freq'] = np.fft.rfftfreq(new_len, 1/f_s)
                new['mag_squared'] = np.abs(np.fft.rfft(data[col], n = new_len))**2
                new['period'] = (1/new['freq'])/(60*24) #default period in days
                new['freq'] = new['freq'] * freq_converter
    return new

def normed_zeropadded_FT(data, substract_mean = True, freq_converter = None, complex_FT = False):
    new_data = FT_with_zp(data, substract_mean = substract_mean, freq_converter = freq_converter, complex_FT = complex_FT)
    return normalize(new_data, 'mag_squared')

def normed_FT(data, substract_mean = True, freq_converter = None):
    new_data = create_fourier_df(data, substract_mean = substract_mean, freq_converter = freq_converter)
    return normalize(new_data, 'abs_fourier_coeff_square')
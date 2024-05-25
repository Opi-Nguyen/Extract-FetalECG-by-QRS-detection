import numpy as np
import kmeans as km
import pandas as pd
import filters as fl
import svd_analysis as svd
import plot_data_functions as pld

def detect_MQRS(data, len_of_data, win_high, win_low, cutoff_low, cutoff_high, filter_order, sampling_rate, start_index, mqrs, MARGIN_FP, MARGIN_FN, window_width, increment_width):
    st_index = start_index
    MQRS = mqrs.copy()
    samples = np.arange(0, len_of_data)
    while(True):
        data_in_window = np.asarray(data[win_low:win_high])
        # Prefilter the data with notch filter to remove power supply noise
        notch_filter_data = fl.Implement_Notch_Filter(sampling_rate, 20, 50, 10, 2, 'butter', data_in_window)
        notch_filter_data = fl.Implement_Notch_Filter(sampling_rate, 5, 150, 10, 2, 'butter', notch_filter_data)
        #filter data and square it to increase relative magnitude of MQRS peaks
        data_MQRS = fl.butter_lowpass_filter(notch_filter_data, cutoff_frequency=cutoff_low, sampling_rate=sampling_rate, order=filter_order)
        data_MQRS_1 = fl.butter_highpass_filter(data_MQRS, cutoff_frequency=cutoff_high, sampling_rate=sampling_rate, order=filter_order)
        data_MQRS = np.square(data_MQRS_1)
        # Apply the K means algorithm with k = 2 to detect the MQRS peaks
        maximums_sample_location = km.get_max_points(data_MQRS, win_low)
        minimums_sample_location = km.get_min_points(data_MQRS, win_low)
        maximums_sample_location = km.filter_max_points(data_MQRS,maximums_sample_location,width_of_filter=75)
        max_min_pairs = km.kmeans_2(minimums_sample_location, maximums_sample_location, data_MQRS, win_low)
        # seperate the MQRS peaks
        mqrs_group = [2]
        MQRS_set = max_min_pairs[max_min_pairs.minimum.isin(mqrs_group)]
        MQRS = pd.concat([MQRS, MQRS_set]).drop_duplicates().reset_index(drop=True)
        # MQRS correction
        MQRS = km.MQRS_correction(MQRS, st_index, data_MQRS, MARGIN_FP, MARGIN_FN, 3, 'g')
        st_index = (MQRS.shape[0] - MQRS_set.shape[0])
        if(win_high == len_of_data):
            break
        if(win_high + increment_width > len_of_data):
            win_high = len_of_data
            win_low = len_of_data - window_width
        else:
            win_high += increment_width
            win_low += increment_width
    return MQRS 


#MQRS template subtraction
def template_subtraction(data, MQRS, start_index_sub, cycle_width, P_Q_duration, mecg_cycles, pc_num):
    start_index = start_index_sub
    while(MQRS.shape[0] - start_index > 20):
        # check if there is enough data in front of first MQRS point for PCA analysis
        current_MQRS = MQRS.iloc[0, 0]
        if (current_MQRS < cycle_width * P_Q_duration):
            start_index += 1
        # get the array of data for performing PCA algorithm
        PCA_array, new_index = km.get_PCA_array(MQRS, data, start_index, P_Q_duration, mecg_cycles, cycle_width)
        # SVD analysis, returns the U, sigma and VT array (VT is V transpose)
        U, sigma, VT = svd.get_SVD_arrays(PCA_array)
        # reconstruct the n principal components as an array
        test_mecg_array = svd.get_MQRS_array(U, sigma, VT, pc_num)
        # perform template subtraction to get the residual data containing noise and FECG
        data = svd.subtract_MECG_template(MQRS, test_mecg_array, data, new_index, P_Q_duration, mecg_cycles, cycle_width)
        current_index = start_index + new_index
        start_index = new_index + mecg_cycles
        end_data = int(MQRS.iloc[start_index,0] - cycle_width*P_Q_duration)
    return data, end_data, current_index

#FQRS detection function
def detect_FQRS(data,length_of_data,window_high,window_low,cutoff_low,cutoff_high,filter_ord,fs,start_index_detection,MARGIN_FP_f,MARGIN_FN_f):
    samples = np.arange(0, length_of_data)
    notch_filter_data = fl.Implement_Notch_Filter(fs, 20, 50, 10, 2, 'butter', data)
    notch_filter_data = fl.Implement_Notch_Filter(fs, 5, 150, 10, 2, 'butter', notch_filter_data)
    data = fl.butter_lowpass_filter(data, cutoff_frequency=cutoff_low, sampling_rate=fs,
                                                 order=filter_ord)
    data = fl.butter_highpass_filter(data, cutoff_frequency=cutoff_high, sampling_rate=fs,
                                                  order=filter_ord)
    maximums_t_fetal = km.get_max_points(data, window_low)
    minimums_t_fetal = km.get_min_points(data, window_low)
    # pld.plot_max_min(data_FQRS_temp, direct_fetal_data, samples, maximums_t_fetal, minimums_t_fetal, window_high,window_low, 100, "Unfiltered Squared data")
    # maximums_t_fetal = km.filter_max_points2(data_FQRS, maximums_t_fetal, width_of_filter=100)

    max_min_pairs = km.kmeans_2(minimums_t_fetal, maximums_t_fetal, data, window_low)
    # seperate the MQRS peaks
    fqrs_group = [2]
    FQRS_set = max_min_pairs[max_min_pairs.minimum.isin(fqrs_group)]
    # MQRS correction
    FQRS = km.MQRS_correction(FQRS_set, start_index_detection, data, MARGIN_FP_f, MARGIN_FN_f, 3, 'g')

    return FQRS, data

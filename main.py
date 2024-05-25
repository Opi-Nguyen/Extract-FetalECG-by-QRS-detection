import numpy as np
import kmeans as km
import pandas as pd
import filters as fl
import plot_data_functions as pld
import processing_functions as pf
import mne

###############################

# Converting to a CSV File
# If you dont have the CSV data, then uncomment this block of code below
edf = mne.io.read_raw_edf('ABDFECG/r04.edf')
header = ','.join(edf.ch_names)
np.savetxt('csv_data/r10.csv', edf.get_data().T, delimiter=',', header=header)
# Main filter parameters ( using 10 and 15)

fs = 1000
cutoff_high = 10
cutoff_low = 15
cutoff_low_direct_data = 20
filter_order = 2
notch_freq = 50
quality = 30

# -----------------------------------------------------------------
# Loading data and all 5 columns in a, b, c, d, e arrays
a,b,c,d,e = np.loadtxt('csv_data/r04.csv', delimiter=',', unpack=True).tolist()

data_raw = np.asarray(b)

# Apply the filter to the direct fetal data to remove baseline and high freq noise (For dataset R)
temp_direct_fetal_data = fl.butter_lowpass_filter(a, cutoff_frequency=cutoff_low_direct_data, sampling_rate=fs, order=filter_order)
direct_fetal_data = fl.butter_highpass_filter(temp_direct_fetal_data, cutoff_frequency=cutoff_high, sampling_rate=fs, order=filter_order)

# Creating an array for sample locations since some data arrays only have voltage signal
length_of_data = len(data_raw)

#Filter param
cutoff_high_MQRS = 2
cutoff_low_MQRS = 15

# Parameters for MQRS detection
window_width = 15000
window_high = window_width
window_low = 0
increment_width = 5000
start_index_detection = 0
MARGIN_FP = 0.8
MARGIN_FN = 1.5
# this is the dataframe that will hold the MQRS points for all the data.
MQRS = pd.DataFrame()


# Parameters for MQRS subtraction
P_Q_duration = .25
PC_num = 1
CYCLE_WIDTH = 700
MECG_CYCLES = 20
start_index_subtraction = 0 # this index keeps track of the current MQRS point

#MQRS detection
#################################################################################
MQRS = pf.detect_MQRS(
    data_raw,
    length_of_data,
    window_high,
    window_low,
    cutoff_low_MQRS,
    cutoff_high_MQRS,
    filter_order,
    fs,
    start_index_detection,
    MQRS,
    MARGIN_FP,
    MARGIN_FN,
    window_width,
    increment_width
    )


print(MQRS['sample_location'])
# Template subtraction up to current end point
data, end_data, start_index_subtraction = pf.template_subtraction(data_raw,MQRS,start_index_subtraction,CYCLE_WIDTH,P_Q_duration,MECG_CYCLES, PC_num)

############################################################################################################################################
# # from here is the code for detecting the FQRS peaks from the FECG residual
# # Find the max and min points of the data within the window
length_of_data_f = end_data
samples_f =np.arange(0, length_of_data)

# Parameters for subtraction
CYCLE_WIDTH = 300
FECG_CYCLES = 120
P_Q_duration = .3
PC_num = 2

# #Filter data for FQRS detection
# #################################################################################
#Filter param for FQRS detection ( using 8 and 100)
cutoff_high_FQRS = 8
cutoff_low_FQRS = 100

# Parameters for FQRS detection
window_width_f = 15000
window_high_f = end_data
window_low_f = 0
increment_width_f = 5000
start_index_detection_f = 0
MARGIN_FP_f = 0.8
MARGIN_FN_f = 1.5
# this is the dataframe that will hold the MQRS points for all the data.

data_FQRS = data[window_low_f:window_high_f]

FQRS,filtered_data = pf.detect_FQRS(data_FQRS,length_of_data_f,window_high_f,window_low_f,cutoff_low_FQRS,cutoff_high_FQRS,filter_order,fs,start_index_detection_f,MARGIN_FP_f,MARGIN_FN_f)

samples = np.arange(0, length_of_data)
# pld.print_data_2(filtered_data, direct_fetal_data, FQRS, samples_f, length_of_data_f, 0, "Plot of the detected FQRS points")
pld.print_data_2(data_raw, direct_fetal_data, MQRS, samples_f, length_of_data_f, 0, "Plot of the detected FQRS points")

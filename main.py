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
# edf = mne.io.read_raw_edf("ABDFECG/r01.edf")
# header = ",".join(edf.ch_names)
# np.savetxt("csv_data/r01.csv", edf.get_data().T, delimiter=",", header=header)
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
a, b, c, d, e = np.loadtxt("csv_data/r01.csv", delimiter=",", unpack=True).tolist()

data_raw = np.asarray(b)

# Apply the filter to the direct fetal data to remove baseline and high freq noise (For dataset R)
temp_aECG_data = fl.butter_lowpass_filter(
    data_raw, cutoff_frequency=cutoff_low_direct_data, sampling_rate=fs, order=filter_order
)
aECG_data = fl.butter_highpass_filter(
    temp_aECG_data,
    cutoff_frequency=cutoff_high,
    sampling_rate=fs,
    order=filter_order,
)

# Creating an array for sample locations since some data arrays only have voltage signal
length_of_data = len(data_raw)

# Filter param
cutoff_high_MQRS = 2
cutoff_low_MQRS = 15

# Parameters for MQRS detection
window_width = 15000
window_high = 300000
window_low = 0
increment_width = 5000
start_index_detection = 0
MARGIN_FP = 0.8
MARGIN_FN = 1.5
# this is the dataframe that will hold the MQRS points for all the data.
MQRS = pd.DataFrame()




# Parameters for MQRS subtraction
P_Q_duration = 0.25
PC_num = 1
CYCLE_WIDTH = 700
MECG_CYCLES = 20
start_index_subtraction = 0  # this index keeps track of the current MQRS point

# MQRS detection
#################################################################################
enhenced_data_mQRS, aECG_signal , peaks, MQRS, MQRS1 = pf.detect_MQRS(
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
    increment_width,
)


# Template subtraction up to current end point
data, end_data, start_index_subtraction = pf.template_subtraction(
    data_raw,
    MQRS1,
    start_index_subtraction,
    CYCLE_WIDTH,
    P_Q_duration,
    MECG_CYCLES,
    PC_num,
)

############################################################################################################################################
# # from here is the code for detecting the FQRS peaks from the FECG residual
# # Find the max and min points of the data within the window
length_of_data_f = end_data
samples_f = np.arange(0, length_of_data)

# Parameters for subtraction
CYCLE_WIDTH = 300
FECG_CYCLES = 120
P_Q_duration = 0.3
PC_num = 2

# #Filter data for FQRS detection
# #################################################################################
# Filter param for FQRS detection ( using 8 and 100)
cutoff_high_FQRS = 8
cutoff_low_FQRS = 100

# Parameters for FQRS detection
window_width_f = 15000
window_high_f = 300000
window_low_f = 0
increment_width_f = 5000
start_index_detection_f = 0
MARGIN_FP_f = 0.8
MARGIN_FN_f = 1.5
# this is the dataframe that will hold the MQRS points for all the data.

data_FQRS = data[window_low_f:window_high_f]

FQRS, filtered_data = pf.detect_FQRS(
    data_FQRS,
    length_of_data_f,
    window_high_f,
    window_low_f,
    cutoff_low_FQRS,
    cutoff_high_FQRS,
    filter_order,
    fs,
    start_index_detection_f,
    MARGIN_FP_f,
    MARGIN_FN_f,
)

samples = np.arange(0, length_of_data)

import matplotlib.pyplot as plt

time_len = len(data_raw)
t = np.linspace(0, time_len/1000, int(time_len), endpoint=False)


a = 0
trim = [a+0,a+7000]

time = t[trim[0]: trim[1]]

#1
origin_data  = data_raw[trim[0]: trim[1]]
#2
enhenced_data_mQRS = enhenced_data_mQRS[trim[0]: trim[1]]
# Meternal_QRS = MQRS['sample_location']
Meternal_QRS = peaks
Meternal_QRS = Meternal_QRS[(Meternal_QRS >= trim[0]) & (Meternal_QRS <= trim[1])]
Fetal_QRS = FQRS
# Fetal_QRS = FQRS['sample_location']
Fetal_QRS = Fetal_QRS[(Fetal_QRS >= trim[0]) & (Fetal_QRS <= trim[1])]
Fetal_QRS = np.delete(Fetal_QRS, 0)
Fetal_QRS = np.append(Fetal_QRS, 8133)
print(Meternal_QRS, Fetal_QRS)


aECG_data_vis = aECG_data[trim[0]: trim[1]]
#3
# reconstructed_mecg = reconstructed_mecg_array[trim[0]: trim[1]]
#4
subtracted_data = filtered_data[trim[0]: trim[1]]

frequency_sampling =1000

plt.figure(figsize=(8, 6))
plt.subplots_adjust(hspace=2)
# plt.subplot(3, 2, 1)
# plt.plot(time, origin_data, label='Raw signal')
# plt.title('Raw Signal')
# plt.xlabel('Time [seconds]')
# plt.grid()

# plt.subplot(3, 2, 2)
# plt.plot(time, subtracted_data, label='Subtracted Signal')
# plt.title('Subtracted Signal')
# plt.xlabel('Time [seconds]')
# plt.grid()

# plt.subplot(3, 2, 3)
# plt.plot(time, enhenced_data_mQRS, label='Enhanced filtered signal and mQRS detection', color='orange')
# plt.plot(Meternal_QRS/1000, enhenced_data_mQRS[Meternal_QRS], 'ro', label='Relative Maxima')
# plt.title('Enhanced filtered for detect mQRS')
# plt.xlabel('Time [seconds]')
# plt.grid()

# plt.subplot(3, 2, 4)
# fetal_recontruct = subtracted_data**2
# plt.plot(time, fetal_recontruct, label='Enhanced filtered for detect fQRS', color='orange')
# plt.plot(Fetal_QRS/1000, fetal_recontruct[Fetal_QRS], 'ro', label='Relative Maxima', color='green')
# plt.title('Enhanced filtered for detect fQRS')
# plt.xlabel('Time [seconds]')
# plt.grid()


# plt.subplot(3, 2, 5)
# plt.plot(time, origin_data, label='mQRS detection', color='black')
# plt.plot(Meternal_QRS/1000, origin_data[Meternal_QRS], 'ro', label='Relative Maxima')
# plt.plot(Fetal_QRS/1000, origin_data[Fetal_QRS], 'ro', label='Relative Maxima', color='green')
# plt.title('mQRS detection')
# plt.xlabel('Time [seconds]')
# plt.grid()

# plt.subplot(3, 2, 6)
# plt.plot(time, aECG_data_vis, label='fQRS detection', color='black')
# plt.title('fQRS detection')
# plt.xlabel('Time [seconds]')
# plt.grid()

fontsize=20
# plt.figure(figsize=(10, 8))
# plt.subplot(2, 2, 1)
# plt.plot(time, origin_data, label='Raw signal')
# plt.title('Raw Signal', fontsize=fontsize)
# plt.xlabel('Time [seconds]', fontsize=fontsize)
# plt.grid()


# plt.subplot(2, 2, 2)
# plt.plot(time, enhenced_data_mQRS, label='Enhanced filtered signal and mQRS detection', color='orange')
# plt.plot(Meternal_QRS/1000, enhenced_data_mQRS[Meternal_QRS], 'ro', label='Relative Maxima')
# plt.title('Enhanced filtered for detect mQRS', fontsize=fontsize)
# plt.xlabel('Time [seconds]', fontsize=fontsize)
# plt.grid()

# plt.subplot(2, 2, 3)
# fetal_recontruct = subtracted_data**2
# plt.plot(time, fetal_recontruct, label='Enhanced filtered for detect fQRS', color='orange')
# plt.plot(Fetal_QRS/1000, fetal_recontruct[Fetal_QRS], 'ro', label='Relative Maxima', color='green')
# plt.title('Enhanced filtered for detect fQRS', fontsize=fontsize)
# plt.xlabel('Time [seconds]', fontsize=fontsize)
# plt.grid()


# plt.subplot(2, 2, 4)
# plt.plot(time, origin_data, label='mQRS detection', color='black')
# plt.plot(Meternal_QRS/1000, origin_data[Meternal_QRS], 'ro', label='Relative Maxima')
# plt.plot(Fetal_QRS/1000, origin_data[Fetal_QRS], 'ro', label='Relative Maxima', color='green')
# plt.title('mQRS & fQRS detection', fontsize=fontsize)
# plt.xlabel('Time [seconds]', fontsize=fontsize)
# plt.grid()
plt.subplot(2, 1, 1)
plt.plot(time, aECG_data_vis, label='fQRS detection', color='black')
plt.plot(Meternal_QRS/1000, aECG_data_vis[Meternal_QRS], 'ro', label='Relative Maxima')
plt.xlabel('(A)', fontsize=fontsize)
plt.grid(False) 
# plt.grid()
plt.gca().get_xaxis().set_visible(False)

#hide y-axis 
plt.gca().get_yaxis().set_visible(False)

Meternal_QRS = np.delete(Meternal_QRS, int(len(Meternal_QRS)/2))
plt.subplot(2, 1, 2)
plt.plot(time, aECG_data_vis, label='fQRS detection', color='black')
plt.plot(Meternal_QRS/1000, aECG_data_vis[Meternal_QRS], 'ro', label='Relative Maxima')
plt.xlabel('(B)', fontsize=fontsize)
plt.grid(False) 
# plt.grid()
plt.gca().get_xaxis().set_visible(False)

#hide y-axis 
plt.gca().get_yaxis().set_visible(False)

plt.tight_layout()
plt.show()
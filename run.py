from modules.filterbank import FilterBank
from modules.detectFQRS import DetectFQRS
import numpy as np
import matplotlib.pyplot as plt

#get data
raw_direct_fecg_data, raw_abdecg_data_c1, _, _, _ = np.loadtxt('csv_data/r04.csv', delimiter=',', unpack=True).tolist()
raw_abdecg_data_c1 = np.asarray(raw_abdecg_data_c1)
crop_data = raw_abdecg_data_c1

#config params, attr
frequency_sampling = 1000
FQRS_detector = DetectFQRS(frequency_sampling=frequency_sampling, data=crop_data)

#1 preprocess
preprocessed_MECG = FQRS_detector.preprocess_AECG(data=crop_data, cutoff_low=15, cutoff_high=2, filter_order=2)

#2 detectQRS
Meternal_QRS = FQRS_detector.detect_QRS(data=preprocessed_MECG, fs=200)

##MQRS template subtraction
cycle_width = 700
P_Q_duration = 0.25
start_idx = 0
MECG_CYCLES = 20
pc_num = 1

while len(Meternal_QRS)-start_idx > 20:
    # check if there is enough data in front of first MQRS point for PCA analysis
    current_Meternal_QRS = Meternal_QRS[0]
    if (current_Meternal_QRS < cycle_width * P_Q_duration):
        start_idx +=1
    # get the array of data for performing PCA algorithm
    # PCA
    PCA_QRS_array, new_idx = FQRS_detector.PCA_ananlysis(
        data=temp_data,
        QRS_data=Meternal_QRS,
        start_index=start_idx,
        P_Q_duration=P_Q_duration,
        total_cycles=MECG_CYCLES,
        cycle_width=cycle_width
        )
    # SVD analysis, returns the U, sigma and VT array (VT is V transpose)
    U, sigma, VT = FQRS_detector.SVD_ananlysis(PCA_QRS_array)
    # reconstruct the n principal components as an array
    reconstructed_mecg_array = FQRS_detector.recontruct_MECG(U, sigma, VT, pc_num)
    # # perform template subtraction to get the residual data containing noise and FECG
    temp_data = FQRS_detector.subtract_template(Meternal_QRS, reconstructed_mecg_array, temp_data, new_idx, P_Q_duration, MECG_CYCLES, cycle_width)
    current_index = start_idx + new_idx
    start_idx = new_idx + MECG_CYCLES
    end_data = int(Meternal_QRS[start_idx] - cycle_width*P_Q_duration)
    # data, end_data, current_index



# preprocess FECG 
preprocessed_FECG = FQRS_detector.preprocess_FECG(data=temp_data, cutoff_low=100, cutoff_high=8, filter_order=2)










time_len = len(crop_data)
t = np.linspace(0, time_len/1000, int(time_len), endpoint=False)


trim = [0,10000]

time = t[trim[0]: trim[1]]

#1
origin_data  = crop_data[trim[0]: trim[1]]
#2
preprocessed_QRS = preprocessed_MECG[trim[0]: trim[1]]
Meternal_QRS = Meternal_QRS[(Meternal_QRS >= trim[0]) & (Meternal_QRS <= trim[1])]
#3
# print(reconstructed_mecg_array)
# reconstructed_mecg = reconstructed_mecg_array[trim[0]: trim[1]]
#4
subtracted_data = preprocessed_FECG[trim[0]: trim[1]]


trim = [0,10000]

time = t[trim[0]: trim[1]]

#1
origin_data  = crop_data[trim[0]: trim[1]]
#2
preprocessed_QRS = preprocessed_MECG[trim[0]: trim[1]]
Meternal_QRS = Meternal_QRS[(Meternal_QRS >= trim[0]) & (Meternal_QRS <= trim[1])]
#3
# print(reconstructed_mecg_array)
# reconstructed_mecg = reconstructed_mecg_array[trim[0]: trim[1]]
#4
subtracted_data = preprocessed_FECG[trim[0]: trim[1]]


plt.figure(figsize=(10, 8))
plt.subplot(3, 1, 1)
plt.plot(time, origin_data, label='Original signal')
plt.title('Original Signal')
plt.xlabel('Time [seconds]')
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(time, preprocessed_QRS, label='Filtered signal (high-pass)', color='orange')
plt.plot(Meternal_QRS/frequency_sampling, preprocessed_MECG[Meternal_QRS], 'ro', label='Relative Maxima')
plt.title('Filtered Signal')
plt.xlabel('Time [seconds]')
plt.grid()

plt.subplot(3, 1, 3)
plt.plot(time, subtracted_data, label='Reconstructed MECG', color='red')
# plt.plot(Meternal_QRS/frequency_sampling, preprocessed_data[Meternal_QRS], 'ro', label='Relative Maxima')
plt.title('Filtered Signal')
plt.xlabel('Time [seconds]')
plt.grid()

# plt.subplot(4, 2, 2)
# plt.plot(time, subtracted_data, label='Subtracted FECG', color='red')
# # plt.plot(Meternal_QRS/frequency_sampling, preprocessed_data[Meternal_QRS], 'ro', label='Relative Maxima')
# plt.title('Filtered Signal')
# plt.xlabel('Time [seconds]')
# plt.grid()

plt.tight_layout()
plt.show()
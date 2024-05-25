import numpy as np
import matplotlib.pyplot as plt

from modules.filterbank import FilterBank
from modules.detectFQRS import DetectFQRS
import numpy as np
import mne

edf = mne.io.read_raw_edf("ABDFECG/r01.edf")
header = ",".join(edf.ch_names)
np.savetxt("csv_data/r01.csv", edf.get_data().T, delimiter=",", header=header)

#get data
raw_direct_fecg_data, raw_abdecg_data_c1, a2, a3, a4 = np.loadtxt('csv_data/r01.csv', delimiter=',', unpack=True).tolist()
ecg_signal = np.asarray(raw_direct_fecg_data)

# Sampling frequency
Fs = 1000  # Hz

# Perform FFT
n = len(ecg_signal)
T = 1.0 / Fs
f = np.fft.fftfreq(n, T)
ecg_fft = np.fft.fft(ecg_signal)

# Calculate the magnitude
magnitude = np.abs(ecg_fft)

# Select the frequency range from 0 to 200 Hz
freq_range = (f >= 0) & (f <= 200)

# Plot the frequency spectrum
plt.figure(figsize=(12, 6))
plt.plot(f[freq_range], magnitude[freq_range])
plt.title('ECG Frequency Spectrum (0 - 200 Hz)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid()
plt.show()
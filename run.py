from modules import FilterBank, DetectFQRS
import numpy as np

raw_direct_fecg_data, raw_abdecg_data_c1, _, _, _ = np.loadtxt('r10.csv', delimiter=',', unpack=True).tolist()

data = np.asarray(raw_abdecg_data_c1)



import numpy as np
from scipy.linalg import svd
import kmeans as km

def get_SVD_arrays(PCA_array):
    U, sigma, VT = svd(PCA_array)
    return U, sigma, VT


def get_MQRS_array(U, sigma, VT, num_of_components):
    matrix_MECG = np.matrix(U[:, :num_of_components]) * np.diag(sigma[:num_of_components]) * np.matrix(VT[:num_of_components, :])
    array_MECG = np.squeeze(np.asarray(matrix_MECG))
    return array_MECG


# this function subtracts the template of MQRS signals from the array data
def subtract_MECG_template(mqrs_set, MECG_array, data, start_index, P_Q_duration, mecg_cycles, cycle_width):
    current_index = start_index
    data_new = data.copy()
    #copy the samples to the array
    for i in range(0,mecg_cycles):
        current_MQRS = mqrs_set.iloc[current_index, 0]
        data_index = int(current_MQRS - (cycle_width*P_Q_duration))
        for j in range(0, cycle_width):
            data_new[data_index] = data_new[data_index] - MECG_array[i][j]
            data_index = data_index + 1
        current_index = current_index + 1
    return data_new

# this function adds the template of MQRS signals and an array data
def add_MECG_template(mqrs_set, MECG_array, data, start_index, P_Q_duration, mecg_cycles, cycle_width):
    end = min(len(mqrs_set.index), len(MECG_array))
    current_index = start_index
    data_new = data
    #copy the samples to the array
    for i in range(start_index,end):
        current_MQRS = mqrs_set.iloc[current_index, 0]
        data_index = int(current_MQRS - (cycle_width*P_Q_duration))
        for j in range(0, cycle_width):
            data_new[data_index] = data_new[data_index] + MECG_array[i][j]
            data_index = data_index + 1
        current_index = current_index + 1
    return data_new


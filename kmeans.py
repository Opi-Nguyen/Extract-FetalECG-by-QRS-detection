import numpy as np
import pandas as pd
import scipy.signal
from numpy import linalg as la


# This function will return the indexes from the local maximums in the data array. The window start is the start index in the original data.
def get_max_points(data, window_start):
    maximums_t = scipy.signal.argrelextrema(data, np.greater)
    maximums_t = np.array(maximums_t)
    return maximums_t

# This function will filter the max points taking the largest magnitude in a window
def filter_max_points(data, maximums, width_of_filter):
    print(maximums)
    i = 0
    while i < len(maximums[0]) - 1:
        if maximums[0][i+1] - maximums[0][i] <= width_of_filter:
            if data[maximums[0][i]] >= data[maximums[0][i+1]]:
                maximums = np.delete(maximums, [i+1], 1)
                print("del")
            else:
                maximums = np.delete(maximums, [i], 1)
                continue
        i += 1
    return maximums


def filter_max_points2(data, maximums, width_of_filter):
    i = 0
    while (i < len(maximums[0]) - 1):
        max = 0;
        index = 0;
        j = 0
        while (maximums[0][j] - maximums[0][i] < width_of_filter):
            print(data[maximums[0][j]])
            if (data[maximums[0][j]] >= max):
                print('got here')
                max = data[maximums[0][j]]
                index = j
            if i < len(maximums[0]) - 2:
                j = j + 1
            else:
                j = 0
                break
        if (j != 0):
            for k in range(i, j + 1):
                if k != index:
                    maximums = np.delete(maximums,[k],1)
        i = i + j
    return maximums


# This function will return the indexes of the local maximums in the data array. The window start is the start index in the original data.
def get_min_points(data, window_start):
    minimums_t = scipy.signal.argrelextrema(data, np.less)
    minimums_t = np.array(minimums_t)
    return minimums_t


# define set values for k mean centers. This will ensure that the algorithm consistently returns the same grouping.
def init_centers(max_min_pairs, centers):
    centers[1][0] = 0
    centers[2][0] = 0
    centers[3][0] = 0
    centers[1][1] = min(max_min_pairs['y'])
    centers[2][1] = (max(max_min_pairs['y'] - min(max_min_pairs['y']))/5)
    centers[3][1] = max(max_min_pairs['y'])
    return centers

# define set values for K-man centers for k = 2.
def init_centers_2(max_min_pairs, centers):
    centers[1][0] = 0
    centers[2][0] = 0
    centers[1][1] = min(max_min_pairs['y'])
    centers[2][1] = max(max_min_pairs['y'])
    return centers


# This function will calculate the dx*dy feature of the data to generate a dataframe used in K-means algorithm.
# The minumums and maximums parameter are the arrays with index of local min and local max. The data parameter
# is the data array from the signal.
def get_x_y_pairs_1(minimums, maximums, data, window_low):
    start_index_min = 0  # this is the index value into the array of signal data for min array
    start_index_max = 0  # this is the index value into the array of signal data for max array
    # ensure that the first value is a maximum
    if (minimums[0][start_index_min] < maximums[0][start_index_max]):
        start_index_min = start_index_min + 1

    sample_location = np.array([])  # array to store the sample location for dx*dy value
    delta_y_array = np.array([])
    delta_x_array = np.array([])

    for i in range(start_index_max, min(len(minimums[0]) - start_index_min, len(maximums[0]))):
        if(i + start_index_min > 0):
            delta_x = abs(minimums[0][i + start_index_min] - minimums[0][i + start_index_min - 1])
        else:
            delta_x = abs(minimums[0][i + start_index_min] - maximums[0][i])  # delta x is the sample # difference
        delta_y = (abs(data[minimums[0][i + start_index_min]] - data[
             maximums[0][i]])) # delta y is the y difference between max and min
        # delta_y = data[maximums[0][i]]
        delta_y_array = np.append(delta_y_array, delta_y)
        # delta_y_array = np.append(delta_y_array, delta_y*delta_x)
        delta_x_array = np.append(delta_x_array, 0)

        sample_location = np.append(sample_location, maximums[0][i] + window_low)

    max_min_pairs = pd.DataFrame()
    max_min_pairs['sample_location'] = sample_location.T
    max_min_pairs.sample_location = max_min_pairs.sample_location.astype(int)
    max_min_pairs["x"] = delta_x_array.T
    max_min_pairs["y"] = delta_y_array.T

    return max_min_pairs


# takes the dataframe and calculates the distance from each point to each of the centers. Leaves a column "minimum"
# that contains the center that is closest to the point and returns a dataframe
def assign_points(max_min_pairs, centers, color_map):
    for i in centers.keys():
        max_min_pairs['distance_from_{}'.format(i)] = ((np.sqrt((max_min_pairs['x'] -
                      centers[i][0])**2 + (max_min_pairs['y'] - centers[i][1])**2))) # this is to calculate cartesian distance
    centers_dist_col = ['distance_from_{}'.format(i) for i in centers.keys()]
    max_min_pairs['minimum'] = max_min_pairs.loc[:,centers_dist_col].idxmin(axis=1)
    max_min_pairs['minimum'] = max_min_pairs['minimum'].map(lambda x: int(x.lstrip('distance_from_')))
    max_min_pairs['color'] = max_min_pairs['minimum'].map(lambda x: color_map[x])
    return max_min_pairs


# Update the centers based on the points that were closest to them
def recalculate_centers(centers, max_min_pairs):
    for i in centers.keys():
        centers[i][0] = np.mean(max_min_pairs[max_min_pairs['minimum'] == i]['x'])
        centers[i][1] = np.mean(max_min_pairs[max_min_pairs['minimum'] == i]['y'])
    return centers



# this function returns the array of values that can be used for PCA analysis
def get_PCA_array(mqrs_set, data, start_index, P_Q_duration, total_cycles, cycle_width):
    # the new index for the MQRS point of the first cycle in the design array
    new_index = start_index
    current_index = start_index
    current_MQRS = mqrs_set.iloc[current_index,0]

    # verify that there are enough samples to copy full QRS complex to array otherwise skip the first cycle
    if(current_MQRS < cycle_width*P_Q_duration):
        current_index = current_index + 1
        new_index = current_index

    QRS_array = [[0 for i in range(cycle_width)] for j in range(total_cycles)]

    #copy the samples to the array
    for i in range(0,total_cycles):
        current_MQRS = mqrs_set.iloc[current_index, 0]
        data_index = int(current_MQRS - (cycle_width*P_Q_duration))
        for j in range(0, cycle_width):
            # print('data_index %d' %data_index)
            QRS_array[i][j] = data[data_index]
            data_index = data_index + 1
        current_index = current_index + 1
    return QRS_array, new_index


# will have to create another method that indexes the data until stability is reached. The two first intervals can then be
# added as an input to this function to ensure the start point is correct
def MQRS_correction(mqrs_set, start_index, data, margin_fp, margin_fn, group, color):
    # will have to figure out how to stabilize the data for real time prediction
    # for now going ahead 2 will work
    mqrs_set1 = mqrs_set.copy()
    pd.set_option('display.max_columns', None)
    current_index = start_index + 2
    # detection will have to use a overlapping window
    while(current_index < mqrs_set1.shape[0] - 2):
        interval_b1 = mqrs_set1.iloc[current_index - 1,0] - mqrs_set1.iloc[current_index - 2,0]
        current_interval = mqrs_set1.iloc[current_index,0] - mqrs_set1.iloc[current_index - 1,0]
        if(current_interval >= margin_fp*interval_b1 and current_interval <= margin_fn*interval_b1):
            current_index = current_index + 1
            continue
        elif(current_interval < margin_fp*interval_b1):
            interval_f1 = mqrs_set1.iloc[current_index + 1, 0] - mqrs_set1.iloc[current_index, 0]
            interval_f2 = mqrs_set1.iloc[current_index + 2, 0] - mqrs_set1.iloc[current_index + 1, 0]
            if(current_interval < margin_fp*interval_f1 or current_interval < margin_fp*interval_f2):
                mqrs_set1 = mqrs_set1.drop(mqrs_set1.index[current_index])
                current_index = current_index - 1
        elif(current_interval > margin_fn*interval_b1):
            interval_f1 = mqrs_set1.iloc[current_index + 1, 0] - mqrs_set1.iloc[current_index, 0]
            interval_f2 = mqrs_set1.iloc[current_index + 2, 0] - mqrs_set1.iloc[current_index + 1, 0]
            if(current_interval > margin_fn*interval_f1 or current_interval > margin_fn*interval_f2):
                start_index = mqrs_set1.iloc[current_index - 1, 0]
                end_index = mqrs_set1.iloc[current_index,0]
                cycles = int(((end_index - start_index + 50))/interval_b1)
                max_index = start_index
                for i in range(1,cycles):
                    max = -1
                    # start = start_index + interval_b1*i - 25
                    # stop = start_index + interval_b1*i + 25
                    start = interval_b1*i - 25
                    stop = interval_b1*i + 25
                    for i in range(start,stop):
                        if(abs(data[i]) > max):
                            max = data[i]
                            max_index = i
                    mqrs_set1 = mqrs_set1.append({'sample_location': max_index, 'x': 0, 'minimum': group, 'color': color},ignore_index=True)
        current_index = current_index + 1
    mqrs_set1 = mqrs_set1.sort_values(by=['sample_location'])
    return mqrs_set1




def kmeans_2(min_sample_loc, max_sample_loc, data, window_low):
    color_map = {1: 'r', 2: 'b'} # Map the points groups to different colors
    # Initialization section
    max_min_pairs = get_x_y_pairs_1(min_sample_loc, max_sample_loc, data, window_low)
    k = 2

    centers = {
        i + 1: [0,np.random.uniform(min(max_min_pairs['y']),max(max_min_pairs['y']))]
        for i in range(k)
    }

    # initialize the centers
    centers = init_centers_2(max_min_pairs, centers)

    # Initial point assignment and center recalculation
    max_min_pairs = assign_points(max_min_pairs, centers, color_map)
    centers = recalculate_centers(centers, max_min_pairs)

    # copy and update, if the old value center assignments equals the new, then the k means process is complete
    while True:
        old_assignments = max_min_pairs['minimum'].copy(deep=True)
        centers = recalculate_centers(centers, max_min_pairs)
        max_min_pairs = assign_points(max_min_pairs,centers, color_map)
        if old_assignments.equals(max_min_pairs['minimum']):
            break
    return max_min_pairs



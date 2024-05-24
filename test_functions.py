# start of windowing algorith
increment = 1000
start_index = window_low + increment
end_index = window_high + increment
while(end_index <= length_of_data):
    data_in_window = b[start_index:end_index]
    # Prefilter the data
    notch_filter_data = fl.Implement_Notch_Filter(fs, 20, 50, 10, 2, 'butter', data_in_window)
    notch_filter_data = fl.Implement_Notch_Filter(fs, 5, 150, 10, 2, 'butter', notch_filter_data)
    low_filter_data = fl.butter_lowpass_filter(notch_filter_data, cutoff_frequency=cutoff_low, sampling_rate=fs,order=filter_order)
    high_filter_data = fl.butter_highpass_filter(low_filter_data, cutoff_frequency=cutoff_high, sampling_rate=fs,order=filter_order)
    # keep the full filtered dataset
    filtered_data_set = np.append(filtered_data_set,high_filter_data[end_index-start_index-increment:end_index-start_index])
    # Find the max and min points of the data within the window
    temp_maximums_sample_location = km.get_max_points(filtered_data_set, end_index-start_index-increment, end_index)
    temp_minimums_sample_location = km.get_min_points(filtered_data_set, end_index-start_index-increment, end_index)
    minimums_sample_location = np.append(minimums_sample_location,temp_minimums_sample_location)
    maximums_sample_location = np.append(maximums_sample_location,temp_maximums_sample_location)
    # Apply the second k means clustering algorithm
    ###############################################
    # Initialization section
    temp_max_min_pairs = km.get_x_y_pairs_1(temp_minimums_sample_location, temp_maximums_sample_location, high_filter_data, start_index,
                                       end_index)
    k = 3

    centers = {
        i + 1: [0, np.random.uniform(min(temp_max_min_pairs['y']), max(temp_max_min_pairs['y']))]
        for i in range(k)
    }

    # initialize the centers
    centers = km.init_centers(temp_max_min_pairs, centers)

    # Initial point assignment and center recalculation
    temp_max_min_pairs = km.assign_points(temp_max_min_pairs, centers, color_map)
    centers = km.recalculate_centers(centers, temp_max_min_pairs)

    # copy and update, if the old value center assignments equals the new, then the k means process is complete
    while True:
        old_assignments = temp_max_min_pairs['minimum'].copy(deep=True)
        centers = km.recalculate_centers(centers, temp_max_min_pairs)
        temp_max_min_pairs = km.assign_points(temp_max_min_pairs, centers, color_map)
        if old_assignments.equals(temp_max_min_pairs['minimum']):
            break

    start_index = start_index + increment
    end_index = end_index + increment

#plot the data showing the min max pairs
pld.plot_max_min(filtered_data_set, direct_fetal_data, samples, maximums_sample_location, minimums_sample_location, 60000, 0,200, "Filtered data")


# # Apply the second k means clustering algorithm
# ###############################################
# # color map so that the cluster data can be plotted
# color_map = {1: 'r', 2: 'b', 3: 'g'} # Map the points groups to different colors
# # Initialization section
# max_min_pairs = km.get_x_y_pairs_1(minimums_sample_location, maximums_sample_location, high_filter_data)
# k = 3
#
# centers = {
#     i + 1: [0,np.random.uniform(min(max_min_pairs['y']),max(max_min_pairs['y']))]
#     for i in range(k)
# }
#
# # initialize the centers
# centers = km.init_centers(max_min_pairs, centers)
#
# # Initial point assignment and center recalculation
# max_min_pairs = km.assign_points(max_min_pairs, centers, color_map)
# centers = km.recalculate_centers(centers, max_min_pairs)
#
# # copy and update, if the old value center assignments equals the new, then the k means process is complete
# while True:
#     old_assignments = max_min_pairs['minimum'].copy(deep=True)
#     centers = km.recalculate_centers(centers, max_min_pairs)
#     max_min_pairs = km.assign_points(max_min_pairs,centers, color_map)
#     if old_assignments.equals(max_min_pairs['minimum']):
#         break
#
# groups = [2, 3]
# mqrs_group = [3]
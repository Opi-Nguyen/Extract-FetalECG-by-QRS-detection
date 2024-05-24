import matplotlib.pyplot as plt



def print_data_2(data_1, data_2, qrs_set, samples, window_high, window_low, label):
    plt.subplots(figsize=(100,10))
    plt.scatter(qrs_set['sample_location'], data_1[qrs_set['sample_location']], color='r', s=600, alpha=1, edgecolors='k')
    plt.plot(samples[window_low:window_high], data_1[window_low:window_high],color='black',linewidth=4)
    # plt.plot(samples[window_low:window_high], data_2[window_low:window_high])
    plt.xlabel('samples')
    plt.ylabel('signal')
    plt.title(label)
    plt.show()



def print_k_means_groupings(grouping, label):
    plt.subplots(figsize=(10,10))
    plt.scatter(grouping['x'], grouping['y'], color =grouping['color'], alpha=0.5, edgecolors='k')
    plt.xlabel('x')
    plt.ylabel('x*y')
    plt.title(label)
    plt.show()



def plot_scatter(array, samples, cycle_width, title):
    plt.subplots(figsize=(10,10))
    for i in range(0,len(array)):
        plt.scatter(samples[0:cycle_width], array[i][0:cycle_width])
    plt.xlabel('samples')
    plt.ylabel('signal')
    plt.title(title)
    plt.show()




def plot_max_min(data1, data2, samples, max, min, window_high, window_low, winsize, title):
    plt.subplots(figsize=(winsize,10))
    plt.plot(samples[window_low:window_high], data1[window_low:window_high],color='black',linewidth=4)
    # plt.scatter(samples[window_low:window_high], data1[window_low:window_high])
    # plt.plot(samples[window_low:window_high], data2[window_low:window_high])
    plt.plot(samples[max], data1[max], 'o', color='r')
    plt.plot(samples[min], data1[min], 'o', color ='g')
    plt.xlabel('Samples')
    plt.ylabel('Signal')
    plt.title(title)
    plt.show()


def plot_unfiltered(data1, samples, window_high, window_low, winsize, title):
    plt.subplots(figsize=(winsize,10))
    plt.plot(samples[window_low:window_high], data1[window_low:window_high],color='black',linewidth=4)
    plt.xlabel('Samples')
    plt.ylabel('Signal')
    plt.title(title)
    plt.show()


def plot_solid(pca_array, pca_samples, cycle_width, title):
    plt.subplots(figsize=(10,10))
    for i in range(0,len(pca_array)):
        plt.plot(pca_samples[0:cycle_width], pca_array[i][0:cycle_width],linewidth=4)
    plt.xlabel('samples')
    plt.ylabel('MECG signal')
    plt.title(title)
    plt.show()



def plot_overlay_3(data_1, data_2, data_3, samples, window_high, window_low, title):
    plt.subplots(figsize=(100,10))
    plt.plot(samples[window_low:window_high], data_1[window_low:window_high])
    plt.plot(samples[window_low:window_high], data_2[window_low:window_high])
    plt.plot(samples[window_low:window_high], data_3[window_low:window_high])
    plt.xlabel('Samples')
    plt.ylabel('Signal')
    plt.title(title)
    plt.show()


def plot_overlay_2(data_1, data_2, samples, window_high, window_low, size, title):
    plt.subplots(figsize=(size,6))
    plt.plot(samples[window_low:window_high], data_1[window_low:window_high],color='black',linewidth=4)
    plt.plot(samples[window_low:window_high], data_2[window_low:window_high])
    plt.xlabel('Samples')
    plt.ylabel('Signal')
    plt.title(title)
    plt.show()


def plot_overlay_1(data_1, samples, window_high, window_low, size, title):
    plt.subplots(figsize=(size,10))
    plt.plot(samples[window_low:window_high], data_1[window_low:window_high],color='black',linewidth=4)
    plt.xlabel('Samples')
    plt.ylabel('Signal')
    plt.title(title)
    plt.show()

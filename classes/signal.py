import numpy as np
from scipy import signal as sp

# split_equal helper function to add horizontal margin
def add_margin(array, margin):
    if array[0] - margin > 0:
        array[0] = array[0] - margin
    else:
        array[0] = 0
    if array[1] + margin < 540: # image width
        array[1] = array[1] + margin
    else:
        array[1] = 540
    return array

# Split view in left and right matrix
# Define peak and width with signal processing
def split(matrix):
    # Cleanup Manual
    matrix = matrix[:, 100:-100] # cut sides

    # Cleanup Vertical (bottom)
    non_zero_bottom_index = np.nonzero(np.count_nonzero(matrix, axis=1))[0][-1]
    matrix = matrix[:non_zero_bottom_index, :]


    # Cleanup Horizontal
    non_zero_column = np.count_nonzero(matrix, axis=0) # count the numbers that are not 0 for each column
    peaks, _ = sp.find_peaks(non_zero_column, height=150, distance=50, width=10)
    # Get two highest peaks
    # sort values by highest value and return top 2 value indexes
    # sort indexes from low to high (left to right)
    ind = np.sort(np.argpartition(non_zero_column[peaks], -2)[-2:])
    peaks = peaks[ind]
    # Get width
    _, _, left_ips, right_ips = sp.peak_widths(non_zero_column, peaks, rel_height=0.80)

    # plt.plot(non_zero_column)
    # plt.scatter(peaks, non_zero_column[peaks], color="yellow")
    # plt.show()

    left_ips = left_ips.astype(int)
    right_ips = right_ips.astype(int)

    margin_horizontal = 20

    left_valley = add_margin([left_ips[0], right_ips[0]], margin_horizontal)
    right_valley = add_margin([left_ips[1], right_ips[1]], margin_horizontal)

    # Remove vertical (top)
    margin_vertical = 200
    top_index = 0 if matrix.shape[0] - margin_vertical < 0 else matrix.shape[0] - margin_vertical
    matrix = matrix[top_index:, :]

    # Remove horizontal by value & split
    left_matrix = matrix[:, left_valley[0]:left_valley[1]]
    right_matrix = matrix[:, right_valley[0]:right_valley[1]]

    return left_matrix, right_matrix, non_zero_column[peaks]
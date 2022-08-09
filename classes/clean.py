import numpy as np
from numba import vectorize, float64, uint16

@vectorize([uint16(uint16, float64)], nopython=True)
def compare(x, median):
    if np.absolute(x - median) <= 60: return 0 # if difference bigger than 5cm, replace with 0
    else: return x

def row_loop(row):
    row_median = np.median(row[row != 0])
    if not np.isnan(row_median):
        return compare(row, row_median)
    else:
        return row

def remove_ground(matrix):
    new_matrix = np.array([row_loop(row) for row in matrix])
    return new_matrix

# remove all values larger than certain value
def remove_background(matrix):
    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_scale = 0.0010000000474974513

    # We will be removing the background of objects more than
    #  clipping_distance_in_meters meters away
    clipping_distance_in_meters = 1.5 #1 meter
    clipping_distance = clipping_distance_in_meters / depth_scale

    grey_color = 0

    # everything above row 150 smaller than 300 is not correct
    matrix_removed_background = np.where((matrix > clipping_distance), grey_color, matrix)

    return matrix_removed_background

# remove noise above certain row
def remove_noise(matrix, distance = 500):
    grey_color = 0
    height = matrix.shape[0]
    height = int(height / 2)
    matrix_rn = np.where((matrix[:height, :] < distance), grey_color, matrix[:height, :])
    matrix = np.vstack([matrix_rn, matrix[height:, :]])
    return matrix
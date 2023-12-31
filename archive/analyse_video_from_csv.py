# Math time
import matplotlib.pyplot as plt
import numpy as np

import csv

selected_margin = 50
path = f"C:\\Users\\tibod\\Documents\\Visual Code\\final-work\\csv\\"

def get_distance(first, second):
    second_distance = []
    first_distance = []

    first_index_list = []
    second_index_list = []

    for i in range(len(first)):
        first_index_list.append(i)
        first_index_list.append(i)
    
    for i in range(len(second)):
        second_index_list.append(i)
        second_index_list.append(i)

    if len(first) == len(second):
        second_index_list = second_index_list[:-1]
        first_index_list = first_index_list[1:]
    else:
        first_index_list = first_index_list[1:-1]

    for i in first_index_list:
        first_distance.append(first[i])
    for i in second_index_list:
        second_distance.append(second[i])
    
    first_distance = np.array(first_distance)
    second_distance = np.array(second_distance)

    distance = first_distance + second_distance

    return distance[1:][::2], distance[0:][::2]

def get_time(a):
    time_mm = []
    for i in range(1, len(a)):
        time_mm.append(round((a[i] - a[i - 1])/30*1000))
    return time_mm

def get_margin_for_valley(peaks, min_values):
    # Will get x values of where the peaks are at the same hight
    idx = np.argwhere(np.diff(np.sign(peaks[0] - peaks[1]))).flatten()
    
    # take average of min_values distance when the peaks overlap
    average_distance = np.average(np.hstack([min_values[0][idx], min_values[1][idx]]))
    return average_distance

# Edited script from https://gist.github.com/endolith/250860
import sys
from numpy import NaN, Inf, arange, isscalar, asarray, array
def peakdet(v, delta, margin = 0, x = None):
    """
    Converted from MATLAB script at http://billauer.co.il/peakdet.html
    
    Returns two arrays
    
    function [maxtab, mintab]=peakdet(v, delta, x)
    %PEAKDET Detect peaks in a vector
    %        [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
    %        maxima and minima ("peaks") in the vector V.
    %        MAXTAB and MINTAB consists of two columns. Column 1
    %        contains indices in V, and column 2 the found values.
    %      
    %        With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
    %        in MAXTAB and MINTAB are replaced with the corresponding
    %        X-values.
    %
    %        A point is considered a maximum peak if it has the maximal
    %        value, and was preceded (to the left) by a value lower by
    %        DELTA.
    
    % Eli Billauer, 3.4.05 (Explicitly not copyrighted).
    % This function is released to the public domain; Any use is allowed.
    
    """
    maxtab = []
    mintab = []
       
    if x is None:
        x = arange(len(v))
    
    v = asarray(v)
    
    if len(v) != len(x):
        sys.exit('Input vectors v and x must have same length')
    
    if not isscalar(delta):
        sys.exit('Input argument delta must be a scalar')
    
    if delta <= 0:
        sys.exit('Input argument delta must be positive')
    
    mn, mx = Inf, -Inf
    mnpos, mxpos = NaN, NaN
    
    lookformax = True
    
    for i in arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]
        
        if lookformax:
            if this < mx-delta and this > margin:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn+delta and this < margin:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True

    return array(maxtab), array(mintab)

def main():
    global selected_margin
    min_values = []
    with open(f"{path}min_{selected_margin}.csv") as csvfile:
        reader = csv.reader(csvfile, delimiter=';', quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
        for row in reader: # each row is a list
            min_values.append(row)

    min_peak = []
    with open(f"{path}min_peak_{selected_margin}.csv") as csvfile:
        reader = csv.reader(csvfile, delimiter=';', quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
        for row in reader: # each row is a list
            min_peak.append(row)

    max_peak = []
    with open(f"{path}max_peak_{selected_margin}.csv") as csvfile:
        reader = csv.reader(csvfile, delimiter=';', quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
        for row in reader: # each row is a list
            max_peak.append(row)

    peak = []
    with open(f"{path}peak_{selected_margin}.csv") as csvfile:
        reader = csv.reader(csvfile, delimiter=';', quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
        for row in reader: # each row is a list
            peak.append(row)

    min_values = np.array(min_values)
    min_peak = np.array(min_peak)
    max_peak = np.array(max_peak)
    peak = np.array(peak)

    margin = get_margin_for_valley(peak, min_values)

    _, valley_right_min = peakdet(min_values[0], 1, margin)
    _, valley_left_min = peakdet(min_values[1], 1, margin)

    valley_right_x = np.array(valley_right_min[:, 0]).astype(int)
    valley_left_x = np.array(valley_left_min[:, 0]).astype(int)

    valley_right_y = np.array(valley_right_min[:, 1])
    valley_left_y = np.array(valley_left_min[:, 1])

    if np.min(valley_left_x) < np.min(valley_right_x): # select the foot that has first IC
        # If this is the left foot
        left_stride_lengths, right_stride_lengths = get_distance(min_values[0][valley_left_x] - valley_left_y, min_values[1][valley_right_x] - valley_right_y)
    else:
        # If this is the right foot
        right_stride_lengths, left_stride_lengths = get_distance(min_values[1][valley_right_x] - valley_right_y, min_values[0][valley_left_x] - valley_left_y)
    
    right_stride_times = get_time(valley_right_x)
    left_stride_times = get_time(valley_left_x)

    # print(f"x-values: {valley_right_x}")
    # print(f"IC right foot: {valley_right_y}")
    # print(f"Presw left foot: {min_values[1][valley_right_x]}")
    print(f"Stride distance left foot: {left_stride_lengths}")
    print(f"Stride duration left foot: {left_stride_times}")
    print(f"---")
    # print(f"x-values: {valley_left_x}")
    # print(f"IC left foot: {valley_left_y}")
    # print(f"Presw right foot: {min_values[0][valley_left_x]}")
    print(f"Stride distance right foot: {right_stride_lengths}")
    print(f"Stride duration right foot: {right_stride_times}")

if __name__ == "__main__":
    main()
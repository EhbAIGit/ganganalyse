# Math time
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sp
import os.path

import csv

min = []
with open("csv/min.csv") as csvfile:
    reader = csv.reader(csvfile, delimiter=';', quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
    for row in reader: # each row is a list
        min.append(row)

min_peak = []
with open("csv/min_peak.csv") as csvfile:
    reader = csv.reader(csvfile, delimiter=';', quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
    for row in reader: # each row is a list
        min_peak.append(row)

max_peak = []
with open("csv/max_peak.csv") as csvfile:
    reader = csv.reader(csvfile, delimiter=';', quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
    for row in reader: # each row is a list
        max_peak.append(row)

peak = []
with open("csv/peak.csv") as csvfile:
    reader = csv.reader(csvfile, delimiter=';', quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
    for row in reader: # each row is a list
        peak.append(row)

min = np.array(min)
min_peak = np.array(min_peak)
max_peak = np.array(max_peak)
peak = np.array(peak)

def color_scatter(axis, x_l, y_l, x_r, y_r):
    color = iter(plt.cm.rainbow(np.linspace(0, 1, len(x_l) + len(x_r))))
    for i in range(len(x_l)):
        c = next(color)
        axis.scatter(x_l[i], y_l[i], color=c)
    for i in range(len(x_r)):
        c = next(color)
        axis.scatter(x_r[i], y_r[i], color=c)


def get_margin_for_valley(peaks, min):
    # Will get x values of where the peaks are at the same hight
    idx = np.argwhere(np.diff(np.sign(peaks[0] - peaks[1]))).flatten()
    
    # take average of min distance when the peaks overlap
    average_distance = np.average(np.hstack([min[0][idx], min[1][idx]]))
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

# peaks_right, _ = sp.find_peaks(min[0])
# peaks_left, _ = sp.find_peaks(min[1])

# _, _, right_left_ips, right_right_ips = sp.peak_widths(min[0], peaks_right)
# _, _, left_left_ips, left_right_ips = sp.peak_widths(min[1], peaks_left)

margin = get_margin_for_valley(peak, min)

# Edit this function to only have peaks under certain value -> this value being the distance of the feet when next to each other
_, valley_right_min = peakdet(min[0], 1, margin)
_, valley_left_min = peakdet(min[1], 1, margin)

valley_right_x = np.array(valley_right_min[:, 0]).astype(int)
valley_left_x = np.array(valley_left_min[:, 0]).astype(int)

figure, axis = plt.subplots(1, 3)

# Minimum
axis[0].plot(min[0], color="red")
axis[0].plot(min[1], color="blue")
color_scatter(axis[0], valley_left_x, valley_left_min[:,1], valley_right_x, valley_right_min[:,1])
# axis[0].scatter(peaks_right, min_right[peaks_right], color="yellow")
# axis[0].scatter(peaks_left, min_left[peaks_left], color="yellow")
axis[0].set_title("Minimum")

# Minimum Peak
axis[1].plot(min_peak[0], color="red")
axis[1].plot(min_peak[1], color="blue")
color_scatter(axis[1], valley_left_x, min_peak[0][valley_left_x], valley_right_x, min_peak[1][valley_right_x])
axis[1].set_title("Minimum Peak")

# Maximum Peak
axis[2].plot(max_peak[0], color="red")
axis[2].plot(max_peak[1], color="blue")
color_scatter(axis[2], valley_left_x, max_peak[0][valley_left_x], valley_right_x, max_peak[1][valley_right_x])
axis[2].set_title("Maximum Peak")

# Peak Min Distance
print("Staplengte")
print("Minimum van piek - minimum van initial contact")
print(valley_left_x)
print(valley_right_x)
print(min[0][valley_left_x])
print(min[1][valley_right_x])
print(min_peak[0][valley_left_x])
print(min_peak[1][valley_right_x])
print(max_peak[0][valley_left_x])
print(max_peak[1][valley_right_x])
# print(min_peak[1][valley_right_x] - valley_right_min[:,1])
# print(min_peak[0][valley_left_x] - valley_left_min[:,1])

# Peak Max Distance
# print("Maximum van piek - minimum van initial contact")
# print(max_peak[1][valley_right_x] - valley_right_min[:,1])
# print(max_peak[0][valley_left_x] - valley_left_min[:,1])

# Combine all the operations and display
plt.show()
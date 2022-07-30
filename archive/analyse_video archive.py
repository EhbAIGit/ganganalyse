# Math time
import matplotlib.pyplot as plt
import numpy as np

import csv

real_values = np.array([[815, 892], [841, 827]])

min_stack = {}
min_peak_stack = {}
max_peak_stack = {}

margins = list(range(40, 210, 10))

def color_scatter(axis, x_l, y_l, x_r, y_r, set_label=False):
    color = iter(plt.cm.rainbow(np.linspace(0, 1, len(x_l) + len(x_r))))
    for i in range(len(x_l)):
        c = next(color)
        if set_label:
            axis.scatter(x_l[i], y_l[i], color=c, label=f"{y_l[i]}")
        else:
            axis.scatter(x_l[i], y_l[i], color=c)
    for i in range(len(x_r)):
        c = next(color)
        if set_label:
            axis.scatter(x_r[i], y_r[i], color=c, label=f"{y_r[i]}")
        else:
            axis.scatter(x_r[i], y_r[i], color=c)

def draw_lines(axis, x_l, y_l, x_r, y_r, v_l, v_r):
    color = iter(plt.cm.rainbow(np.linspace(0, 1, len(x_l) + len(x_r))))
    for i in range(len(x_l)):
        c = next(color)
        axis.plot([x_l[i], x_l[i]], [y_l[i], v_l[i]], color=c)
    for i in range(len(x_r)):
        c = next(color)
        axis.plot([x_r[i], x_r[i]], [y_r[i], v_r[i]], color=c)

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

for selected_margin in margins:
    min_values = []
    with open(f"csv/min_{selected_margin}.csv") as csvfile:
        reader = csv.reader(csvfile, delimiter=';', quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
        for row in reader: # each row is a list
            min_values.append(row)

    min_peak = []
    with open(f"csv/min_peak_{selected_margin}.csv") as csvfile:
        reader = csv.reader(csvfile, delimiter=';', quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
        for row in reader: # each row is a list
            min_peak.append(row)

    max_peak = []
    with open(f"csv/max_peak_{selected_margin}.csv") as csvfile:
        reader = csv.reader(csvfile, delimiter=';', quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
        for row in reader: # each row is a list
            max_peak.append(row)

    peak = []
    with open(f"csv/peak_{selected_margin}.csv") as csvfile:
        reader = csv.reader(csvfile, delimiter=';', quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
        for row in reader: # each row is a list
            peak.append(row)

    min_values = np.array(min_values)
    min_peak = np.array(min_peak)
    max_peak = np.array(max_peak)
    peak = np.array(peak)

    # peaks_right, _ = sp.find_peaks(min_values[0])
    # peaks_left, _ = sp.find_peaks(min_values[1])

    # _, _, right_left_ips, right_right_ips = sp.peak_widths(min_values[0], peaks_right)
    # _, _, left_left_ips, left_right_ips = sp.peak_widths(min_values[1], peaks_left)

    margin = get_margin_for_valley(peak, min_values)

    # Edit this function to only have peaks under certain value -> this value being the distance of the feet when next to each other
    _, valley_right_min = peakdet(min_values[0], 1, margin)
    _, valley_left_min = peakdet(min_values[1], 1, margin)

    valley_right_x = np.array(valley_right_min[:, 0]).astype(int)
    valley_left_x = np.array(valley_left_min[:, 0]).astype(int)

    valley_right_y = np.array(valley_right_min[:, 1])
    valley_left_y = np.array(valley_left_min[:, 1])

    print(f"x-values: {valley_right_x}")
    print(f"IC right foot: {valley_right_y}")
    print(f"Presw left foot: {min_values[0][valley_left_x]}")
    print(f"---")
    print(f"x-values: {valley_left_x}")
    print(f"IC left foot: {valley_left_y}")
    print(f"Presw right foot: {min_values[1][valley_right_x]}")
    print(f"######################################\n")

    if np.min(valley_left_x) < np.min(valley_right_x):
        # first IC left then right
        # => first we calculate stride right then left
        print("here")
    else:
        # first IC right then left
        # => first we calculate stride left then right
        print("here2")

    fig, axis = plt.subplots(1, 3, figsize=(25,10))
    fig.suptitle(f"Initial Contact foot A compared to Preswing foot B with ground margin: {selected_margin}", fontsize=16)

    # Minimum
    axis[0].plot(min_values[0], color="red", label="right foot")
    axis[0].plot(min_values[1], color="blue", label="left foot")
    color_scatter(axis[0], valley_left_x, valley_left_min[:,1], valley_right_x, valley_right_min[:,1], True)
    # axis[0].scatter(peaks_right, min_right[peaks_right], color="yellow")
    # axis[0].scatter(peaks_left, min_left[peaks_left], color="yellow")
    axis[0].set_title("Minimum (Initial Contact)")
    axis[0].set_ylabel('depth')
    axis[0].set_xlabel('frames', loc="left")
    axis[0].legend(bbox_to_anchor=(0.5,-0.1,0.5,0.2),
                mode="expand", borderaxespad=0, ncol=3)

    # Minimum
    axis[1].plot(min_values[0], color="red", label="right foot")
    axis[1].plot(min_values[1], color="blue", label="left foot")
    # color_scatter(axis[1], valley_left_x, valley_left_min[:,1], valley_right_x, valley_right_min[:,1])
    color_scatter(axis[1], valley_left_x, min_values[0][valley_left_x], valley_right_x, min_values[1][valley_right_x], True)
    # draw_lines(axis[1], valley_left_x, valley_left_min[:,1], valley_right_x, valley_right_min[:,1], min_values[0][valley_left_x], min_values[1][valley_right_x])
    # axis[0].scatter(peaks_right, min_right[peaks_right], color="yellow")
    # axis[0].scatter(peaks_left, min_left[peaks_left], color="yellow")
    axis[1].set_title("Minimum (Preswing)")
    axis[1].set_xlabel('frames', loc="left")
    axis[1].legend(bbox_to_anchor=(0.5,-0.1,0.5,0.2),
                mode="expand", borderaxespad=0, ncol=3)

    # Minimum Peak
    axis[2].plot(min_peak[0], color="red", label="right foot")
    axis[2].plot(min_peak[1], color="blue", label="left foot")
    # color_scatter(axis[2], valley_left_x, valley_left_min[:,1], valley_right_x, valley_right_min[:,1])
    color_scatter(axis[2], valley_left_x, min_peak[0][valley_left_x], valley_right_x, min_peak[1][valley_right_x], True)
    # draw_lines(axis[2], valley_left_x, valley_left_min[:,1], valley_right_x, valley_right_min[:,1], min_peak[0][valley_left_x], min_peak[1][valley_right_x])
    axis[2].set_title("Minimum Peak Matrix (Preswing)")
    axis[2].set_xlabel('frames', loc="left")
    axis[2].legend(bbox_to_anchor=(0.5,-0.1,0.5,0.2),
                mode="expand", borderaxespad=0, ncol=3)

    # Maximum Peak
    # axis[1, 1].plot(max_peak[0], color="red")
    # axis[1, 1].plot(max_peak[1], color="blue")
    # color_scatter(axis[1, 1], valley_left_x, valley_left_min[:,1], valley_right_x, valley_right_min[:,1])
    # color_scatter(axis[1, 1], valley_left_x, max_peak[0][valley_left_x], valley_right_x, max_peak[1][valley_right_x])
    # draw_lines(axis[1, 1], valley_left_x, valley_left_min[:,1], valley_right_x, valley_right_min[:,1], max_peak[0][valley_left_x], max_peak[1][valley_right_x])
    # axis[1, 1].set_title("Maximum Peak (Preswing)")
    # axis[1, 1].legend()

    # Peak Min Distance
    min_a = np.array(real_values[0] - min_values[0][valley_left_x])
    min_b = np.array(real_values[1] - min_values[1][valley_right_x])
    min_stack[selected_margin] = np.hstack([min_a, min_b])
    # min_stack[selected_margin] = np.average(np.absolute(np.hstack([min_a, min_b])))
    min_peak_a = np.array(real_values[0] - min_peak[0][valley_left_x])
    min_peak_b = np.array(real_values[1] - min_peak[1][valley_right_x])
    min_peak_stack[selected_margin] = np.hstack([min_peak_a, min_peak_b])
    # min_peak_stack[selected_margin] = np.average(np.absolute(np.hstack([min_peak_a, min_peak_b])))
    # max_peak_a = np.array(real_values[0] - max_peak[0][valley_left_x])
    # max_peak_b = np.array(real_values[1] - max_peak[1][valley_right_x])
    # max_peak_stack[selected_margin] = np.hstack([max_peak_a, max_peak_b])
    # print(min_peak[1][valley_right_x] - valley_right_min[:,1])
    # print(min_peak[0][valley_left_x] - valley_left_min[:,1])

    # Peak Max Distance
    # print("Maximum van piek - minimum van initial contact")
    # print(max_peak[1][valley_right_x] - valley_right_min[:,1])
    # print(max_peak[0][valley_left_x] - valley_left_min[:,1])

    # Combine all the operations and display
    # plt.show()
print("min_values")
print(min_stack)
# print(min(zip(min_stack.values(), min_stack.keys())))
print("min peak")
print(min_peak_stack)
# print(min(zip(min_peak_stack.values(), min_peak_stack.keys())))
# print("max peak")
# print(max_peak_stack)
# print(min(zip(max_peak_stack.values(), max_peak_stack.keys())))
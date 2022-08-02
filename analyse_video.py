# Math time
import numpy as np
import matplotlib.pyplot as plt
import csv

def get_distance(first, second):
    first_distance = []
    second_distance = []

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

def get_time_difference(first, second):
    first_times = []
    second_times = []

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
        first_times.append(first[i])
    for i in second_index_list:
        second_times.append(second[i])
    
    first_times = np.array(first_times)
    second_times = np.array(second_times)

    time_differences = []
    for i in range(len(second_times)):
        if i % 2 == 0:
            time_differences.append(second_times[i] - first_times[i])
        else:
            time_differences.append(first_times[i] - second_times[i])

    time_differences = np.array(time_differences) / 30 * 1000

    return time_differences[1:][::2], time_differences[0:][::2]

def limp_calc(right, left):
    right_avg = np.average(right)
    left_avg = np.average(left)

    return right_avg - left_avg

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

def main(min_values, peak, f_name):
    min_values = np.array(min_values)
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
        left_step_to, right_step_to = get_time_difference(valley_left_x, valley_right_x)
    else:
        # If this is the right foot
        right_stride_lengths, left_stride_lengths = get_distance(min_values[1][valley_right_x] - valley_right_y, min_values[0][valley_left_x] - valley_left_y)
        right_step_to, left_step_to = get_time_difference(valley_right_x, valley_left_x)
    
    right_stride_times = get_time(valley_right_x)
    left_stride_times = get_time(valley_left_x)

    limp_distance = limp_calc(valley_right_y, valley_left_y)
    limp_time = limp_calc(right_step_to, left_step_to)

    # write to file
    f = open(f"graphs/{f_name}.txt", "w")
    f.write(f"##############\n")
    f.write(f"# Raw Values #\n")
    f.write(f"##############\n")
    f.write(f"IC's Right Foot: {valley_right_x}\n")
    f.write(f"IC's Depth Right Foot: {valley_right_y}\n")
    f.write(f"---\n")
    f.write(f"IC's Left Foot: {valley_left_x}\n")
    f.write(f"IC's Depth Right Foot: {valley_left_y}\n\n")
    f.write(f"##############################\n")
    f.write(f"# Stride Distance & Duration #\n")
    f.write(f"##############################\n")
    f.write(f"Stride distance(s) right foot: {right_stride_lengths}\n")
    f.write(f"Stride duration(s) right foot (ms): {right_stride_times}\n")
    f.write(f"Stride step to duration(s) right foot (ms): {right_step_to}\n")
    f.write(f"---\n")
    f.write(f"Stride distance(s) left foot: {left_stride_lengths}\n")
    f.write(f"Stride duration(s) left foot (ms): {left_stride_times}\n")
    f.write(f"Stride step to duration(s) left foot (ms): {left_step_to}\n\n")
    f.write(f"###################\n")
    f.write(f"# Stride Analysis #\n")
    f.write(f"###################\n")
    f.write(f"IC difference duration (ms): {limp_time}\n")
    if abs(limp_time) > 100:
        if limp_time > 0:
            f.write(f"Potential limp left leg\n")
        else:
            f.write(f"Potential limp right leg\n")
    f.write(f"---\n")
    f.write(f"IC difference distance: {limp_distance}\n")
    if abs(limp_distance) > 20:
        if limp_distance < 0:
            f.write(f"Potential limp left leg")
        else:
            f.write(f"Potential limp right leg")
    f.close()

    # Display Plot

    plt.plot(min_values[0], color="red", label="right foot")
    plt.plot(min_values[1], color="blue", label="left foot")

    plt.scatter(valley_right_x, valley_right_y, color="yellow")
    plt.scatter(valley_right_x, min_values[1][valley_right_x], color="yellow")
    plt.scatter(valley_left_x, valley_left_y, color="yellow")
    plt.scatter(valley_left_x, min_values[0][valley_left_x], color="yellow")

    plt.plot([valley_right_x, valley_right_x], [valley_right_y, min_values[1][valley_right_x]], color="yellow")
    plt.plot([valley_left_x, valley_left_x], [valley_left_y, min_values[0][valley_left_x]], color="yellow")

    plt.title("Minimum Depth of Right and Left Foot")
    plt.ylabel('depth')
    plt.xlabel('frames', loc="left")
    plt.legend(bbox_to_anchor=(0.5,-0.1,0.5,0.2),
                mode="expand", borderaxespad=0, ncol=3)
    plt.savefig(f"graphs/{f_name}.png")
    plt.show()
    plt.close()

def get_csv(f_name):
    min_values = []
    peaks = []
    with open(f"csv/{f_name}_min_values.csv") as csvfile:
        reader = csv.reader(csvfile, delimiter=';', quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
        for row in reader: # each row is a list
            min_values.append(row)
    with open(f"csv/{f_name}_peaks.csv") as csvfile:
        reader = csv.reader(csvfile, delimiter=';', quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
        for row in reader: # each row is a list
            peaks.append(row)
    return min_values, peaks

if __name__ == "__main__":
    f_name = "video1"
    min_value, peak = get_csv(f_name)
    main(min_value, peak, f_name)
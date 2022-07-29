## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#####################################################
##              Align Depth to Color               ##
#####################################################

# First import the library
from turtle import color
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2

# Import multiprocessing
# import multiprocessing as mp

# Math time
import matplotlib.pyplot as plt
from scipy import signal as sp

# OS time
# Import argparse for command-line options
import argparse
# Import os.path for file path manipulation
import os.path

def matrix_to_csv(matrix, filename):
    np.savetxt(f"csv/{filename}", matrix, delimiter=';', fmt='%i')

def remove_sound(matrix):
    y_min = 289 ## Dont do correction above certain pixel in image 
    y_axis = range(len(matrix))[y_min:]
    for i in y_axis:
        x = matrix[i]
        previous_j = x[0]
        if previous_j == 0:
            for k in range(1,len(x)):
                if x[k] != 0:
                    previous_j = x[k]
                    break
        for j in range(1,len(x)):
            if x[j] == 0:
                matrix[i][j] = previous_j
            previous_j = matrix[i][j]
    return matrix

def real_distance(matrix):
    # Merk op, deze formule maakt gebruik van de hoogte van de camera. 
    # Een alternatief is gebruik maken van de IMU van de camera om zo de hoek van de kanteling naar voor te meten
    # en dan sin(alpha) = Overstaande zijde / Schuine Zijde <=> sin(alpha) * Schuine Zijde
    # Dit is altijd zo lang de camera onder een punt staat, Eens hieboven moeten we gebruik maken van
    # cos(alpha) = Aanliggende zijde / Schuine Zijde
    # Deze manier brengt ons tot een accurater en automatischer resultaat
    def pythagoras(c):
        a = 230 # Height of the camera
        # a^2 + b^2 = c^2 -> (c^2 - a^2)^0.5 = b
        # c = distance to camera (points of the depth matrix)
        b_exp = (c * c)  - (a * a)
        if b_exp <= 0:
            return 0
        else:
            return np.sqrt(b_exp)
    return np.vectorize(pythagoras)(matrix)

def remove_ground(matrix):
    def compare(x, median):
        if np.absolute(x - median) <= 50: return 0 # if difference bigger than 15cm, replace with 0
        else: return x
    vcompare = np.vectorize(compare)

    # for i in range(len(matrix)):
    # # for i in range(len(matrix)):
    #     row = matrix[i]
    #     row_median = np.average(row[row != 0])
    #     # row_median = np.median(row)
    #     if not np.isnan(row_median):
    #         matrix[i] = vcompare(row, row_median)
    #         # matrix[i:i+median_of_i_rows] = vcompare(row, row_median)

    new_matrix = []
    for i in range(len(matrix)):
    # for i in range(len(matrix)):
        row = matrix[i]
        row_median = np.median(row[row != 0])
        # row_median = np.median(row)
        if not np.isnan(row_median):
            new_matrix.append(vcompare(row, row_median))
            # matrix[i:i+median_of_i_rows] = vcompare(row, row_median)
    
    return np.array(new_matrix)

def remove_background(depth_matrix, matrix_remove_background):
    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_scale = 0.0010000000474974513

    # We will be removing the background of objects more than
    #  clipping_distance_in_meters meters away
    clipping_distance_in_meters = 1.5 #1 meter
    clipping_distance = clipping_distance_in_meters / depth_scale

    # minimum distance of matrix = 340 (foot min distance), smaller than this is noise
    min_distance = 330

    # Notice that there is still noise

    grey_color = 0
    return np.where((depth_matrix > clipping_distance) | (depth_matrix <= min_distance) , grey_color, matrix_remove_background)

def colorize_depth(matrix, alpha_value = 0.1):
    #tilde will reverse the grayscale (from 0- 255) as this is better for the colormap (red close, blue far)
    return cv2.applyColorMap(~cv2.convertScaleAbs(matrix, alpha=alpha_value), cv2.COLORMAP_TURBO)


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

def split_equal(matrix):
    # Cleanup Manual
    matrix = matrix[:, 50:-50] # cut sides

    # Cleanup Vertical (bottom)
    non_zero_bottom_index = np.nonzero(np.count_nonzero(matrix, axis=1))[0][-1]
    matrix = matrix[:non_zero_bottom_index, :]


    # Cleanup Horizontal
    non_zero_column = np.count_nonzero(matrix, axis=0) # count the numbers that are not 0 for each column

    peaks, _ = sp.find_peaks(non_zero_column, height=200, distance=50, width=10)
    widths, widths_heights, left_ips, right_ips = sp.peak_widths(non_zero_column, peaks, rel_height=0.90)

    left_ips = left_ips.astype(int)
    right_ips = right_ips.astype(int)

    margin_horizontal = 10

    left_valley = add_margin([left_ips[0], right_ips[0]], margin_horizontal)
    right_valley = add_margin([left_ips[1], right_ips[1]], margin_horizontal)

    # display plot
    # plt.plot(non_zero_column)
    # plt.scatter(peaks, non_zero_column[peaks], color="yellow")
    # plt.scatter(left_valley, non_zero_column[left_valley], color="red")
    # plt.scatter(right_valley, non_zero_column[right_valley], color="blue")
    # plt.show()

    # Remove vertical (top)
    margin_vertical = 150
    top_index = 0 if matrix.shape[0] - margin_vertical < 0 else matrix.shape[0] - margin_vertical
    matrix = matrix[top_index:, :]

    # Remove horizontal by value & split
    left_matrix = matrix[:, left_valley[0]:left_valley[1]]
    right_matrix = matrix[:, right_valley[0]:right_valley[1]]

    left_peak_matrix = matrix[:, peaks[0]]
    right_peak_matrix = matrix[:, peaks[1]]

    return left_matrix, right_matrix, non_zero_column[peaks], left_peak_matrix, right_peak_matrix

# Streaming loop
def main():
    # cpu_count = mp.cpu_count()
    # print(f"cpu count is: {cpu_count}")
    
    # Create object for parsing command-line options
    parser = argparse.ArgumentParser(description="Read recorded bag file and display depth stream in jet colormap.\
                                    Remember to change the stream fps and format to match the recorded.")
    # Add argument which takes path to a bag file as an input
    parser.add_argument("-i", "--input", type=str, help="Path to the bag file")
    # Parse the command line arguments to an object
    args = parser.parse_args()
    # Safety if no parameter have been given
    if not args.input:
        print("No input paramater have been given.")
        print("For help type --help")
        exit()
    # Check if the given file have bag extension
    if os.path.splitext(args.input)[1] != ".bag":
        print("The given file is not of correct file format.")
        print("Only .bag files are accepted")
        exit()

    try:
        # Create pipeline
        pipeline = rs.pipeline()

        # Create a config object
        config = rs.config()

        # Tell config that we will use a recorded device from file to be used by the pipeline through playback.
        rs.config.enable_device_from_file(config, args.input)

        # Configure the pipeline to stream the depth stream
        # Change this parameters according to the recorded bag file resolution
        config.enable_stream(rs.stream.depth, rs.format.z16, 30)

        # Start streaming from file
        profile = pipeline.start(config)
        
        # set playback realtime to false -> otherwise it will drop frames to keep it at the same speed as 30fps
        playback = profile.get_device().as_playback()
        playback.set_real_time(False)

        cv2.namedWindow('Image Feed Left leg', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Image Feed Right leg', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
        
        fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
        out = cv2.VideoWriter('videos/output_50.avi', fourcc, 30, (640, 480))

        min_right = []
        min_left = []
        min_peak_right = []
        min_peak_left = []
        max_peak_right = []
        max_peak_left = []
        peak_right = []
        peak_left = []

        i = 0
        print(f"Skipping to frame 20")
        while True:
            while i % 367 < 20:
                # Get frameset of color and depth
                frames = pipeline.wait_for_frames()
                i += 1
            print(f"frame: {i}")
            # Get frameset of color and depth
            frames = pipeline.wait_for_frames()

            # Get depth frame
            depth_frame = frames.get_depth_frame()

            depth_image = np.asanyarray(depth_frame.get_data())

            ################
            # Remove Noise #
            ################
            # Add Missing Values
            # depth_image = remove_sound(depth_image)

            # Real Distance
            # depth_image = real_distance(depth_image)

            # Remove Ground
            depth_image_rg = remove_ground(depth_image)
            # depth_image_bg = remove_background(depth_image, depth_image)

            # Remove background
            depth_image_bg = remove_background(depth_image_rg, depth_image_rg)
            # depth_image_rg = remove_ground(depth_image_bg)

            ##############
            # Split View #
            ##############
            # depth_image_left, depth_image_right = split_equal(depth_image_bg)
            depth_image_left, depth_image_right, peak_values, left_peak_matrix, right_peak_matrix  = split_equal(depth_image_bg)
            if i == 97 or i == 171 or i == 244 or i == 307:
                print("---------------")

            ############
            # Analysis #
            ############
            # Staplengte & tijd (afstand tussen afstand van de linker en rechter voet bij initial contact)
            # Initial Contact
            min_right.append(np.min(depth_image_left[depth_image_left!=0]))
            min_left.append(np.min(depth_image_right[depth_image_right!=0]))

            # Lengte bepalen
            max_peak_right.append(np.max(left_peak_matrix))
            max_peak_left.append(np.max(right_peak_matrix))
            min_peak_right.append(np.min(left_peak_matrix[left_peak_matrix!=0]))
            min_peak_left.append(np.min(right_peak_matrix[right_peak_matrix!=0]))

            # Moment dat benen naast elkaar staan
            peak_right.append(peak_values[0])
            peak_left.append(peak_values[1])
            if peak_values[0] in range(peak_values[1] - 5, peak_values[1] + 5):
                print("---------------------------")


            #################
            # Render images #
            #################
            #   depth align to color on left
            #   depth on right
            depth_colormap_left = colorize_depth(depth_image_left)
            depth_colormap_right = colorize_depth(depth_image_right)
            original = colorize_depth(depth_image_bg)

            # note that left leg = right image and vise versa
            # reshape needs to be bigger
            cv2.resizeWindow('Image Feed Left leg', depth_image_right.shape[1] * 3, depth_image_right.shape[0] * 3)
            cv2.resizeWindow('Image Feed Right leg', depth_image_left.shape[1] * 3, depth_image_left.shape[0] * 3)
            cv2.resizeWindow('Original', 1920, 1440)
            cv2.imshow('Image Feed Left leg', depth_colormap_right)
            cv2.imshow('Image Feed Right leg', depth_colormap_left)
            cv2.imshow('Original', original)

            out.write(original)

            i += 1
            if i == 367:
                figure, axis = plt.subplots(2, 2)

                # Minimum
                matrix_to_csv(np.vstack([min_right, min_left]), "min.csv")
                axis[0, 0].plot(min_right, color="red")
                axis[0, 0].plot(min_left, color="blue")
                axis[0, 0].set_title("Minimum")

                # Maximum
                matrix_to_csv(np.vstack([max_peak_right, max_peak_left]), "max_peak.csv")
                axis[0, 1].plot(max_peak_right, color="red")
                axis[0, 1].plot(max_peak_left, color="blue")
                axis[0, 1].set_title("Maximum on peak matrix")

                matrix_to_csv(np.vstack([min_peak_right, min_peak_left]), "min_peak.csv")
                axis[1, 0].plot(min_peak_right, color="red")
                axis[1, 0].plot(min_peak_left, color="blue")
                axis[1, 0].set_title("Minimum on peak matrix")

                # Peaks
                matrix_to_csv(np.vstack([peak_right, peak_left]), "peak.csv")
                axis[1, 1].plot(peak_right, color="red")
                axis[1, 1].plot(peak_left, color="blue")
                axis[1, 1].set_title("Peaks")

                # Combine all the operations and display
                plt.show()
                break

            #######################
            # Wait for keypresses #
            #######################
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                break
        out.release()
        cv2.destroyAllWindows()







    finally:
        pipeline.stop()

if __name__ == "__main__":
    main()
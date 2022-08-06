## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#####################################################
##              Align Depth to Color               ##
#####################################################

# First import the library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
import matplotlib.pyplot as plt
# Import OpenCV for easy image rendering
import cv2

# Math time
from scipy import signal as sp

# OS time
# Import argparse for command-line options
import argparse
# Import os.path for file path manipulation
import os.path

from numba import vectorize, float64, uint16
from analyse_video import main as analyse
from progress import Progress

ground_margin = 60

def matrix_to_csv(matrix, filename):
    np.savetxt(f"csv/{filename}.csv", matrix, delimiter=';', fmt='%i')

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

@vectorize([uint16(uint16, float64)], nopython=True)
def compare(x, median):
    if np.absolute(x - median) <= ground_margin: return 0 # if difference bigger than 5cm, replace with 0
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
# @jit(nopython=True)
def remove_background(depth_matrix, matrix_remove_background):
    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_scale = 0.0010000000474974513

    # We will be removing the background of objects more than
    #  clipping_distance_in_meters meters away
    clipping_distance_in_meters = 1.5 #1 meter
    clipping_distance = clipping_distance_in_meters / depth_scale

    grey_color = 0

    # everything above row 150 smaller than 300 is not correct
    matrix_removed_background = np.where((depth_matrix > clipping_distance), grey_color, matrix_remove_background)

    return matrix_removed_background

# remove noise above certain row
def remove_noise(matrix, distance = 500):
    grey_color = 0
    height = matrix.shape[0]
    height = int(height / 2)
    matrix_rn = np.where((matrix[:height, :] < distance), grey_color, matrix[:height, :])
    matrix = np.vstack([matrix_rn, matrix[height:, :]])
    return matrix

# Colorize image for view
def colorize_depth(matrix, alpha_value = 0.1):
    #tilde will reverse the grayscale (from 0- 255) as this is better for the colormap (red close, blue far)
    return cv2.applyColorMap(~cv2.convertScaleAbs(matrix, alpha=alpha_value), cv2.COLORMAP_TURBO)

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
def split_equal(matrix):
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

    # matrix_to_csv(matrix, "test")
    # # display plot
    # plt.plot(non_zero_column)
    # plt.scatter(peaks, non_zero_column[peaks], color="yellow")
    # plt.scatter(left_valley, non_zero_column[left_valley], color="red")
    # plt.scatter(right_valley, non_zero_column[right_valley], color="blue")
    # plt.show()

    return left_matrix, right_matrix, non_zero_column[peaks]

# Streaming loop
def main():
    # Create object for parsing command-line options
    parser = argparse.ArgumentParser(description="Read recorded bag file and display depth stream in jet colormap.\
                                    Remember to change the stream fps and format to match the recorded.")
    # Add argument which takes path to a bag file as an input
    parser.add_argument("-ng", "--no_gui", help="Run file without GUI", action="store_true")
    parser.add_argument("-vo", "--video_output", help="Bag file to avi", action="store_true")
    parser.add_argument("-i", "--input", type=str, help="Path to the bag file", default="video1.bag")
    # Parse the command line arguments to an object
    args = parser.parse_args()
    # Check if the given file have bag extension
    if os.path.splitext(args.input)[1] != ".bag":
        print("The given file is not of correct file format.")
        print("Only .bag files are accepted")
        exit()

    total_frames = 500
    frame_skip = 0
    
    if "video1.bag" in args.input:  # Slow walking
        total_frames = 367
        frame_skip = 20
    elif "video2.bag" in args.input: # Normal walking
        total_frames = 307
        frame_skip = 190
    elif "video3.bag" in args.input: # Limp walking
        total_frames = 364
        frame_skip = 10
    elif "video4.bag" in args.input: # Limp walking female
        total_frames = 367
        frame_skip = 10
    elif "video5.bag" in args.input: # Reflective ground
        total_frames = 270
        frame_skip = 10

    f_name = os.path.basename(args.input)[:-4] # Get base name, remove extention

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

        if not args.no_gui:
            cv2.namedWindow('Image Feed Left leg', cv2.WINDOW_NORMAL)
            cv2.namedWindow('Image Feed Right leg', cv2.WINDOW_NORMAL)

        if args.video_output:
            # Define video
            fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
            out = cv2.VideoWriter(f"videos/{f_name}_background.avi", fourcc, 30, (640, 480))

        progress = Progress()

        min_right = []
        min_left = []
        peak_right = []
        peak_left = []

        i = 0
        print(f"Skipping to frame {frame_skip}")
        progress.start()
        while True:
            while i % total_frames < frame_skip:
                # Get frameset of color and depth
                frames = pipeline.wait_for_frames()
                i += 1
            # print(f"frame: {i}")
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

            # Remove background
            depth_image_bg = remove_background(depth_image_rg, depth_image_rg)

            ##############
            # Split View #
            ##############
            depth_image_left, depth_image_right, peak_values = split_equal(depth_image_bg)

            # Remove noise above certain row
            depth_image_left = remove_noise(depth_image_left)
            depth_image_right = remove_noise(depth_image_right)

            ############
            # Analysis #
            ############
            # Staplengte & tijd (afstand tussen afstand van de linker en rechter voet bij initial contact)
            # Initial Contact
            min_right.append(np.min(depth_image_left[depth_image_left!=0]))
            min_left.append(np.min(depth_image_right[depth_image_right!=0]))

            # Moment dat benen naast elkaar staan
            peak_right.append(peak_values[0])
            peak_left.append(peak_values[1])

            if not args.no_gui or args.video_output:
                original = colorize_depth(depth_image_bg)

            if not args.no_gui:
                #################
                # Render images #
                #################
                #   depth align to color on left
                #   depth on right
                depth_colormap_left = colorize_depth(depth_image_left)
                depth_colormap_right = colorize_depth(depth_image_right)

                # note that left leg = right image and vise versa
                # reshape needs to be bigger
                cv2.resizeWindow('Image Feed Left leg', depth_image_right.shape[1] * 3, depth_image_right.shape[0] * 3)
                cv2.resizeWindow('Image Feed Right leg', depth_image_left.shape[1] * 3, depth_image_left.shape[0] * 3)
                # cv2.resizeWindow('Original', 1920, 1440)
                cv2.imshow('Image Feed Left leg', depth_colormap_right)
                cv2.imshow('Image Feed Right leg', depth_colormap_left)
                # cv2.imshow('Original', original)

            if args.video_output:
                ##################
                # Write to video #
                ##################
                out.write(original)

            ################
            # Progress Bar #
            ################

            i += 1

            percentage = (i - frame_skip) / (total_frames - frame_skip) * 100
            if percentage < 100:
                progress.update(percentage)
            else:
                progress.update(100)
            
            #################
            # Analyse Video #
            #################

            if i == total_frames:
                progress.finish()
                min_values = np.vstack([min_right, min_left])
                peak = np.vstack([peak_right, peak_left])

                # End video
                if args.video_output:
                    out.release()

                # Write values to csv
                matrix_to_csv(min_values, f"{f_name}_min_values")
                matrix_to_csv(peak, f"{f_name}_peaks")

                if not args.no_gui:
                    cv2.destroyAllWindows()
                analyse(min_values, peak, f_name, not args.no_gui)
                break

            #######################
            # Wait for keypresses #
            #######################
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                break

    finally:
        pipeline.stop()

if __name__ == "__main__":
    main()
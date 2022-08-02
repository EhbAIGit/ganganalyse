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

# Import multiprocessing
# import multiprocessing as mp

# Math time
from scipy import signal as sp

from numba import jit, vectorize, float64, uint16
from analyse_video import main as analyse

ground_margin = 60

def matrix_to_csv(matrix, filename):
    np.savetxt(f"csv/{filename}.csv", matrix, delimiter=';', fmt='%i')

def colorize_depth(matrix, alpha_value = 0.04):
    #tilde will reverse the grayscale (from 0- 255) as this is better for the colormap (red close, blue far)
    return cv2.applyColorMap(~cv2.convertScaleAbs(matrix, alpha=alpha_value), cv2.COLORMAP_TURBO)

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

def remove_colour(depth_matrix, matrix_remove_background):
    matrix3d = np.dstack((depth_matrix,depth_matrix,depth_matrix))
    grey_color = 0
    return np.where((matrix3d == 0), grey_color, matrix_remove_background)

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
    try:
        # Create a pipeline
        pipeline = rs.pipeline()

        # Create a config and configure the pipeline to stream
        #  different resolutions of color and depth streams
        config = rs.config()

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))

        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        if not found_rgb:
            print("The demo requires Depth camera with Color sensor")
            exit(0)

        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        if device_product_line == 'L500':
            config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
        else:
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start streaming
        profile = pipeline.start(config)

        cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Original', 1280, 480)
        cv2.namedWindow('Image Feed Left leg', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Image Feed Right leg', cv2.WINDOW_NORMAL)

        # Define video
        fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
        out = cv2.VideoWriter(f"videos/live_background.avi", fourcc, 30, (1280, 480))

        start_record = False

        min_right = []
        min_left = []
        peak_right = []
        peak_left = []

        while True:
            # Get frameset of color and depth
            frames = pipeline.wait_for_frames()

            # Get depth frame
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

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

            # Colorize original
            depth_colormap = colorize_depth(depth_image)
            original = np.hstack((color_image, depth_colormap))
            ##############
            # Split View #
            ##############
            try:
                depth_image_left, depth_image_right, peak_values = split_equal(depth_image_bg)

                # Remove noise above certain row
                depth_image_left = remove_noise(depth_image_left)
                depth_image_right = remove_noise(depth_image_right)

                #################
                # Render images #
                #################
                #   depth align to color on left
                #   depth on right
                depth_colormap_left = colorize_depth(depth_image_left)
                depth_colormap_right = colorize_depth(depth_image_right)

                cv2.resizeWindow('Image Feed Left leg', depth_image_right.shape[1] * 3, depth_image_right.shape[0] * 3)
                cv2.resizeWindow('Image Feed Right leg', depth_image_left.shape[1] * 3, depth_image_left.shape[0] * 3)
                cv2.imshow('Image Feed Left leg', depth_colormap_right)
                cv2.imshow('Image Feed Right leg', depth_colormap_left)

                if start_record:
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
                    
                    ##################
                    # Write to video #
                    ##################
                    out.write(original)

            except ValueError:
                print("frame skipped")
            except IndexError:
                print("frame skipped")

            cv2.imshow('Original', original)

            #######################
            # Wait for keypresses #
            #######################
            key = cv2.waitKey(1)
            # Toggle record
            if key & 0xFF == ord('s'):
                start_record = not start_record
                if start_record == False:
                    min_values = np.vstack([min_right, min_left])
                    peak = np.vstack([peak_right, peak_left])

                    # Write values to csv
                    matrix_to_csv(min_values, f"live_min_values")
                    matrix_to_csv(peak, f"live_peaks")

                    cv2.destroyAllWindows()
                    analyse(min_values, peak, "live")
                    break
            
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                break

    finally:
        out.release() # End video
        pipeline.stop() # Stop pipeline

if __name__ == "__main__":
    main()
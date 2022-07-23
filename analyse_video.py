## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#####################################################
##              Align Depth to Color               ##
#####################################################

# First import the library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2

import time

# Import multiprocessing
import multiprocessing as mp

# Import argparse for command-line options
import argparse
# Import os.path for file path manipulation
import os.path

def matrix_to_csv(matrix, filename):
    np.savetxt(filename, matrix, delimiter=';', fmt='%i')

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
    def compare(x, est):
        if np.absolute(x - est) <= 250: return 0 # if difference bigger than 15cm, replace with 0
        else: return x
    vcompare = np.vectorize(compare)
    # ground_det_average = []
    # for row in matrix:
    #     # row_average = np.average(row[row != 0])
    #     row_average = np.average(row)
    #     if not np.isnan(row_average):
    #         new_row = vcompare(row, row_average)
    #         ground_det_average.append(new_row)
    #     else:
    #         ground_det_average.append(row)

    # ground_det_average = np.array(ground_det_average)
    # return ground_det_average
    ground_det_mean = []
    for row in matrix:
        # row_mean = np.mean(row[row != 0])
        row_mean = np.mean(row)
        if not np.isnan(row_mean):
            new_row = vcompare(row, row_mean)
            ground_det_mean.append(new_row)
        else:
            ground_det_mean.append(row)
    
    ground_det_mean = np.array(ground_det_mean)
    return ground_det_mean

def remove_background(depth_matrix, matrix_remove_background):
    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_scale = 0.0010000000474974513

    # We will be removing the background of objects more than
    #  clipping_distance_in_meters meters away
    clipping_distance_in_meters = 1.5 #1 meter
    clipping_distance = clipping_distance_in_meters / depth_scale

    grey_color = 0
    return np.where((depth_matrix > clipping_distance) | (depth_matrix <= 0), grey_color, matrix_remove_background)

def colorize_depth(matrix, alpha_value = 0.1):
    #tilde will reverse the grayscale (from 0- 255) as this is better for the colormap (red close, blue far)
    return cv2.applyColorMap(~cv2.convertScaleAbs(matrix, alpha=alpha_value), cv2.COLORMAP_TURBO)

# Streaming loop
def main():
    cpu_count = mp.cpu_count()
    print(f"cpu count is: {cpu_count}")
    
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

        cv2.namedWindow('Image Feed', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Image Feed', 640,480)
        i = 0
        while True:
            i += 1
            print(f"frame: {i}")
            # Get frameset of color and depth
            frames = pipeline.wait_for_frames()
            # frames.get_depth_frame() is a 640x360 depth image

            # Get depth frame
            depth_frame = frames.get_depth_frame()

            depth_image = np.asanyarray(depth_frame.get_data())

            # Split array
            # split_array = np.split(depth_image, cpu_count)
            
            # pool = mp.Pool(cpu_count)
            # results = pool.map(remove_ground, split_array)
            # pool.close()

            # depth_image = np.vstack(results)


            # Real Distance
            # depth_image = real_distance(depth_image)

            # Remove Ground
            depth_image_rg = remove_ground(depth_image)
            # depth_image_bg = remove_background(depth_image, depth_image)

            # Remove background
            depth_image_bg = remove_background(depth_image_rg, depth_image_rg)
            # depth_image_rg = remove_ground(depth_image_bg)


            # Render images:
            #   depth align to color on left
            #   depth on right
            depth_colormap = colorize_depth(np.vstack(depth_image_bg))
            # depth_colormap = colorize_depth(np.vstack(depth_image_rg))


            cv2.imshow('Image Feed', depth_colormap)
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
            
            # Print depth_image matrix to csv
            if key & 0xFF == ord('a') or i == 150:
                matrix_to_csv(depth_image, "matrix.csv")

            # Print depth_image matrix no background 
            if key & 0xFF == ord('b'):
                depth_bg_removed = remove_background(depth_image, depth_image)

                depth_colormap_bg_removed = colorize_depth(depth_bg_removed, 0.1)
                cv2.namedWindow('Background_Removed', cv2.WINDOW_NORMAL)
                cv2.imshow('Background_Removed', depth_colormap_bg_removed)
            
            # Print depth_image matrix no background and coordinate correction
            if key & 0xFF == ord('c'):
                real_coords = real_distance(depth_image)
                depth_colormap_coords = colorize_depth(real_coords)
                cv2.namedWindow('Real_Coords', cv2.WINDOW_NORMAL)
                cv2.imshow('Real_Coords', depth_colormap_coords)
            
            # Print depth_image matrix no background and floor detection (on mean)
            if key & 0xFF == ord('d'):
                # depth_bgr_sr = depth_image[:, 100:-100] # remove sides of the matrix to avoid extra unnecesary details
                ground_det_mean = remove_ground(depth_image, "mean")
                ground_det_mean_bg = remove_background(ground_det_mean, ground_det_mean)

                depth_colormap_mean_bg = colorize_depth(ground_det_mean_bg, 0.1)
                cv2.namedWindow('Ground_removed_mean', cv2.WINDOW_NORMAL)
                cv2.imshow('Ground_removed_mean', depth_colormap_mean_bg)
            
            # Print depth_image matrix no background and floor detection (on average)
            if key & 0xFF == ord('e'):
                ground_det_average = remove_ground(depth_image, "average")
                ground_det_average_bg = remove_background(ground_det_average, ground_det_average)

                depth_colormap_average_bg = colorize_depth(ground_det_average_bg, 0.1)
                cv2.namedWindow('Ground_removed_average', cv2.WINDOW_NORMAL)
                cv2.imshow('Ground_removed_average', depth_colormap_average_bg)
            
            # Print depth_image matrix no background and floor detection with coord_correction (on mean)
            if key & 0xFF == ord('f'):
                real_coords = real_distance(depth_image)
                ground_det_mean = remove_ground(real_coords, "mean")
                ground_det_mean_bg = remove_background(ground_det_mean, ground_det_mean)

                depth_colormap_cc_mean_bg = colorize_depth(ground_det_mean_bg, 0.1)
                cv2.namedWindow('Ground_removed_cc_mean', cv2.WINDOW_NORMAL)
                cv2.imshow('Ground_removed_cc_mean', depth_colormap_cc_mean_bg)
            
            
            # Print depth_image matrix no background and floor detection with coord_correction (on average)
            if key & 0xFF == ord('g'):
                real_coords = real_distance(depth_image)
                ground_det_average = remove_ground(real_coords, "average")
                ground_det_average_bg = remove_background(ground_det_average, ground_det_average)

                depth_colormap_cc_average_bg = colorize_depth(ground_det_average_bg, 0.1)
                cv2.namedWindow('Ground_removed_cc_average', cv2.WINDOW_NORMAL)
                cv2.imshow('Ground_removed_cc_average', depth_colormap_cc_average_bg)




    finally:
        pipeline.stop()

if __name__ == "__main__":
    main()
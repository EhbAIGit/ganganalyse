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
import math


# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()

# Create colorizer
colorizer = rs.colorizer()

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

# Enable recording
config.enable_record_to_file('test.bag')

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 1.5 #1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)


# used to record the time when we processed last frame
prev_frame_time = 0
 
# used to record the time at which we processed current frame
new_frame_time = 0


def matrix_to_csv(matrix, filename):
    np.savetxt(filename, matrix, delimiter=';', fmt='%i')

def remove_sound(matrix):
    y_min = 289 ## Dont do correction above certain pixel in image 
    y_axis = range(len(depth_image))[y_min:]
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
                depth_image[i][j] = previous_j
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

def remove_ground(matrix, mean_avg):
    def compare(x, est):
        if np.absolute(x - est) < 150: return 0 # if difference bigger than 15cm, replace with 0
        else: return x
    vcompare = np.vectorize(compare)


    if mean_avg == "mean":
        ground_det_mean = []
        for row in matrix:
            row_mean = np.mean(row[row != 0])
            if not np.isnan(row_mean):
                new_row = vcompare(row, row_mean)
                ground_det_mean.append(new_row)
            else:
                ground_det_mean.append(row)
        
        ground_det_mean = np.array(ground_det_mean)
        # remove the backgroudn > 1.5m from image
        return ground_det_mean
    
    if mean_avg == "average":
        ground_det_average = []
        for row in matrix:
            row_average = np.average(row[row != 0])
            if not np.isnan(row_average):
                new_row = vcompare(row, row_average)
                ground_det_average.append(new_row)
            else:
                ground_det_average.append(row)

        ground_det_average = np.array(ground_det_average)
        # remove the backgroudn > 1.5m from image
        return ground_det_average

def remove_background(depth_matrix, matrix_remove_background):
    grey_color = 0
    return np.where((depth_matrix > clipping_distance) | (depth_matrix <= 0), grey_color, matrix_remove_background)

def colorize_depth(matrix, alpha_value = 0.04):
    #tilde will reverse the grayscale (from 0- 255) as this is better for the colormap (red close, blue far)
    return cv2.applyColorMap(~cv2.convertScaleAbs(matrix, alpha=alpha_value), cv2.COLORMAP_TURBO)

# Streaming loop
try:
    while True:
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Remove background - Set pixels further than clipping_distance to grey
        grey_color = 50

        depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
        bg_removed = remove_background(depth_image_3d, color_image)

        # Render images:
        #   depth align to color on left
        #   depth on right
        depth_colormap = colorize_depth(depth_image)
        # depth_colormap = np.asanyarray(
        #     colorizer.colorize(aligned_depth_frame).get_data())
        images = np.hstack((bg_removed, depth_colormap))


        # Check if camera is horizontally aligned
        horizontal = False
        left = depth_image[-1][0]
        right = depth_image[-1][-1]
        if left in range(right-3, right+3):
            horizontal = True

        # FPS
        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time

        fps = int(fps)

        cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
        font = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(images,f'Horizontally Aligned: {horizontal}',(0,15),font,1,(255,255,255),1)  #text,coordinate,font,size of text,color,thickness of font
        cv2.putText(images,f'{left}',(0,470),font,1,(255,255,255),1)  #text,coordinate,font,size of text,color,thickness of font
        cv2.putText(images,f'{right}',(605,470),font,1,(255,255,255),1)  #text,coordinate,font,size of text,color,thickness of font
        cv2.putText(images, f'fps: {fps}', (570, 15), font, 1, (255,255,255),1, cv2.LINE_AA)
        cv2.imshow('Align Example', images)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
        
        # Print depth_image matrix to csv
        if key & 0xFF == ord('a'):
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
        
        # Print d -> g
        if key & 0xFF == ord('h'):
            ground_det_mean = remove_ground(depth_image, "mean")
            ground_det_mean_bg = remove_background(ground_det_mean, ground_det_mean)

            depth_image_3d_mean_bg = np.dstack((ground_det_mean_bg,ground_det_mean_bg,ground_det_mean_bg))
            bg_removed_mean_bg = remove_background(depth_image_3d_mean_bg, color_image)

            depth_colormap_mean_bg = colorize_depth(ground_det_mean_bg, 0.1)

            images_mean = np.hstack((bg_removed_mean_bg, depth_colormap_mean_bg))
            cv2.namedWindow('Ground_removed_mean', cv2.WINDOW_NORMAL)
            cv2.imshow('Ground_removed_mean', images_mean)
            matrix_to_csv(ground_det_mean_bg, "Ground_removed_mean.csv")


            ground_det_average = remove_ground(depth_image, "average")
            ground_det_average_bg = remove_background(ground_det_average, ground_det_average)

            depth_image_3d_average_bg = np.dstack((ground_det_average_bg,ground_det_average_bg,ground_det_average_bg))
            bg_removed_average_bg = remove_background(depth_image_3d_average_bg, color_image)

            depth_colormap_average_bg = colorize_depth(ground_det_average_bg, 0.1)

            images_average = np.hstack((bg_removed_average_bg, depth_colormap_average_bg))
            cv2.namedWindow('Ground_removed_average', cv2.WINDOW_NORMAL)
            cv2.imshow('Ground_removed_average', images_average)
            matrix_to_csv(ground_det_average_bg, "Ground_removed_average.csv")
            

            real_coords = real_distance(depth_image)
            ground_det_cc_mean = remove_ground(real_coords, "mean")
            ground_det_cc_mean_bg = remove_background(ground_det_cc_mean, ground_det_cc_mean)

            depth_image_3d_cc_mean_bg = np.dstack((ground_det_cc_mean_bg,ground_det_cc_mean_bg,ground_det_cc_mean_bg))
            bg_removed_cc_mean_bg = remove_background(depth_image_3d_cc_mean_bg, color_image)

            depth_colormap_cc_mean_bg = colorize_depth(ground_det_cc_mean_bg, 0.1)

            images_cc_mean = np.hstack((bg_removed_cc_mean_bg, depth_colormap_cc_mean_bg))
            cv2.namedWindow('Ground_removed_cc_mean', cv2.WINDOW_NORMAL)
            cv2.imshow('Ground_removed_cc_mean', images_cc_mean)
            matrix_to_csv(ground_det_cc_mean_bg, "Ground_removed_cc_mean.csv")
            

            real_coords = real_distance(depth_image)
            ground_det_cc_average = remove_ground(real_coords, "average")
            ground_det_cc_average_bg = remove_background(ground_det_cc_average, ground_det_cc_average)

            depth_image_3d_cc_average_bg = np.dstack((ground_det_cc_average_bg,ground_det_cc_average_bg,ground_det_cc_average_bg))
            bg_removed_cc_average_bg = remove_background(depth_image_3d_cc_average_bg, color_image)

            depth_colormap_cc_average_bg = colorize_depth(ground_det_cc_average_bg, 0.1)

            images_cc_average = np.hstack((bg_removed_cc_average_bg, depth_colormap_cc_average_bg))
            cv2.namedWindow('Ground_removed_cc_average', cv2.WINDOW_NORMAL)
            cv2.imshow('Ground_removed_cc_average', images_cc_average)
            matrix_to_csv(ground_det_cc_average_bg, "Ground_removed_cc_average.csv")




finally:
    pipeline.stop()
#####################################################
##               Read bag from file                ##
#####################################################


# First import library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2
# Import argparse for command-line options
import argparse
# Import os.path for file path manipulation
import os.path

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


def matrix_to_csv(matrix, filename):
    np.savetxt(filename, matrix, delimiter=';', fmt='%i')

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

    playback = profile.get_device().as_playback()
    playback.set_real_time(False)

    # Create opencv window to render image in
    cv2.namedWindow("Depth Stream", cv2.WINDOW_AUTOSIZE)
    cv2.resizeWindow('Depth Stream', 1920, 1440)
    
    # Create colorizer object
    colorizer = rs.colorizer()

    # fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    f_name = args.input[2:-4]
    # out = cv2.VideoWriter(f"videos/{f_name}_original.avi", fourcc, 30, (640, 480))

    i = 0
    # Streaming loop
    while True:
        print (i)
        # Get frameset of depth
        frames = pipeline.wait_for_frames()

        # Get depth frame
        depth_frame = frames.get_depth_frame()
        depth_image = np.asanyarray(depth_frame.get_data())

        def mouse_callback(event, x, y, flags, params):
            if event == 1:
                print(f"coords {x, y}, {depth_image[y, x]}")

        cv2.setMouseCallback("Depth Stream", mouse_callback)
        # Colorize depth frame to jet colormap
        depth_color_frame = colorizer.colorize(depth_frame)

        # Convert depth_frame to numpy array to render image in opencv
        depth_color_image = np.asanyarray(depth_color_frame.get_data())

        # Render image in opencv window
        cv2.imshow("Depth Stream", depth_color_image)
        # out.write(depth_color_image)
        key = cv2.waitKeyEx(0)
        # if pressed escape exit program
        if key == 27:
            cv2.destroyAllWindows()
            break
        if key == 2424832:
            j = 0
            total_frames = 172
            while j < total_frames - 2:
                frames = pipeline.wait_for_frames()
                j += 1
            i -= 2

        # Print depth_image matrix to csv
        if key & 0xFF == ord('a'):
            matrix_to_csv(depth_image, "matrix.csv")

        # if i == 96 or i == 243 or i == 170 or i == 306:
        #     matrix_to_csv(depth_frame.get_data(), f"matrix_{i}.csv")

        i = i + 1
    # out.release()
    cv2.destroyAllWindows()

finally:
    pass
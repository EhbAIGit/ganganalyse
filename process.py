import argparse
import os.path

from classes.video import Video
from classes.device import Device
from classes.progress import Progress
from classes.window import Window
from classes.recorder import Recorder
from classes.vault import Vault
from classes.frame import Frame
from classes.clean import *
from classes.signal import *
from analyse_file import main as analyse
import cv2

# Streaming loop
def main():
    # Create object for parsing command-line options
    parser = argparse.ArgumentParser(description="Read recorded bag file and display depth stream in jet colormap.\
                                    Remember to change the stream fps and format to match the recorded.")
    # Add argument which takes path to a bag file as an input
    parser.add_argument("-ng", "--no_gui", help="Run file without GUI", action="store_true")
    parser.add_argument("-vo", "--video_output", help="Bag file to avi", action="store_true")
    parser.add_argument("-i", "--input", type=str, help="Path to the bag file")
    # Parse the command line arguments to an object
    args = parser.parse_args()

    f_name = "live"
    live = True
    complete = False
    start_record = False

    if args.input:
        # Check if the given file have bag extension
        if os.path.splitext(args.input)[1] != ".bag":
            print("The given file is not of correct file format.")
            print("Only .bag files are accepted")
            exit()
        
        video = Video(args.input)

        total_frames = video.total_frames
        frame_skip = video.frame_skip

        f_name = video.f_name
        live = False

        progress = Progress()
        progress.start()

        i = 0
        print(f"Skipping to frame {frame_skip}")

    try:
        device = Device(live, args.input)

        if not args.no_gui:
            left_leg_window = Window('Image Feed Left leg')
            right_leg_window = Window('Image Feed Right leg')
            original_window = Window('Original')
            if live:
                original_window.resize_display((1280, 480))
            else:
                original_window.resize_display((640, 480))

        if args.video_output:
            recorder = Recorder(f_name, live)

        vault = Vault(f_name)

        while True:
            if not live:
                while i % total_frames < frame_skip:
                    # Get frameset of color and depth
                    device.pipeline.wait_for_frames()
                    i += 1

            # Get frameset of color and depth
            frame = Frame(device.pipeline.wait_for_frames())
            if not args.no_gui or args.video_output:
                frame.set_original(live)

            ################
            # Remove Noise #
            ################

            # Remove Ground
            frame.depth_image = remove_ground(frame.depth_image)

            # Remove background
            frame.depth_image = remove_background(frame.depth_image)

            try:
                ##############
                # Split View #
                ##############
                depth_image_left, depth_image_right, peak_values = split(frame.depth_image)

                # Remove noise above certain row
                depth_image_left = remove_noise(depth_image_left)
                depth_image_right = remove_noise(depth_image_right)

                frame.set_legs(depth_image_right, depth_image_left)

                ############
                # Analysis #
                ############
                # Staplengte & tijd (afstand tussen afstand van de linker en rechter voet bij initial contact)
                # Initial Contact
                if not live or start_record:
                    vault.min_right.append(frame.get_leg_zeroless("right"))
                    vault.min_left.append(frame.get_leg_zeroless("left"))

                    # Moment dat benen naast elkaar staan
                    vault.peak_right.append(peak_values[0])
                    vault.peak_left.append(peak_values[1])

                    if args.video_output:
                        ##################
                        # Write to video #
                        ##################
                        recorder.write(frame.original)

                if not args.no_gui:
                    #################
                    # Render images #
                    #################
                    #   depth align to color on left
                    #   depth on right
                    frame.set_color_legs()

                    left_leg_window.resize_display(frame.get_shape("left"))
                    right_leg_window.resize_display(frame.get_shape("right"))
                    
                    left_leg_window.display(frame.left_leg_color)
                    right_leg_window.display(frame.right_leg_color)

            except ValueError:
                pass
            except IndexError:
                pass

            if not args.no_gui:
                original_window.display(frame.original)

            ################
            # Progress Bar #
            ################
            if not live:
                i += 1

                percentage = (i - frame_skip) / (total_frames - frame_skip) * 100
                if percentage < 100:
                    progress.update(percentage)
                else:
                    progress.finish()
                    complete = True

            key = cv2.waitKey(1)
            if live and key & 0xFF == ord('s'):
                start_record = not start_record
                if start_record == False:
                    print("Recording Stopped")
                    complete = True
                else:
                    print("Recording Started")

            #################
            # Analyse Video #
            #################
            if complete:
                vault.store()

                if not args.no_gui:
                    cv2.destroyAllWindows()

                analyse(vault.min_values, vault.peak, f_name, not args.no_gui)
                break

    finally:
        # End video
        if args.video_output:
            recorder.release()
        device.pipeline.stop()

if __name__ == "__main__":
    main()
import numpy as np
import cv2

class Frame:
    def __init__(self, frame) -> None:
        self.frame = frame
        self.depth_image = self.setup(frame)
    
    def setup(self, frame):
        # Get depth frame
        depth_frame = frame.get_depth_frame()
        depth_image = np.asanyarray(depth_frame.get_data())

        return depth_image

    def colorize_depth(self, image):
        return cv2.applyColorMap(~cv2.convertScaleAbs(image, alpha=0.1), cv2.COLORMAP_TURBO)
    
    def set_original(self, live):
        depth_colormap = self.colorize_depth(self.depth_image)

        if live:
            color_frame = self.frame.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())
            self.original = np.hstack((color_image, depth_colormap))
        else:
            self.original = depth_colormap
    
    def set_legs(self, left_leg, right_leg):
        self.left_leg = left_leg
        self.right_leg = right_leg
    
    def set_color_legs(self):
        self.left_leg_color = self.colorize_depth(self.left_leg)
        self.right_leg_color = self.colorize_depth(self.right_leg)
    
    def get_leg_zeroless(self, leg):
        if leg == "left":
            return np.min(self.left_leg[self.left_leg!=0])
        else:
            return np.min(self.right_leg[self.right_leg!=0])
    
    def get_shape(self, leg):
        if leg == "left":
            return (self.left_leg.shape[1] * 3, self.left_leg.shape[0] * 3)
        else:
            return (self.right_leg.shape[1] * 3, self.right_leg.shape[0] * 3)
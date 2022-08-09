import cv2

class Recorder:
    def __init__(self, f_name, live):
        self.output = self.setup(f_name, live)
    
    def setup(self, f_name, live):
        fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
        if live:
            return cv2.VideoWriter(f"videos/live_background.avi", fourcc, 30, (1280, 480))
        else:
            return cv2.VideoWriter(f"videos/{f_name}_background.avi", fourcc, 30, (640, 480))
    
    def write(self, image):
        self.output.write(image)
    
    def release(self):
        self.output.release()
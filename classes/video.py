import os

class Video:
    def __init__(self, input):
        self.total_frames, self.frame_skip = self.get_video_setup(input)
        self.f_name = self.get_name(input)

    def get_video_setup(self, input):
        if "video1.bag" in input:  # Slow walking
            return 367, 20
        elif "video2.bag" in input: # Normal walking
            return 307, 190
        elif "video3.bag" in input: # Limp walking
            return 356, 10
        elif "video4.bag" in input: # Limp walking female
            return 367, 10
        elif "video5.bag" in input: # Reflective ground
            return 270, 10
    
    def get_name(self, input):
        return os.path.basename(input)[:-4] # Get base name, remove extention
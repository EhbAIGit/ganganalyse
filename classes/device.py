import pyrealsense2 as rs

class Device:
    def __init__(self, live=True, f_name=None):
        self.pipeline = self.setup(live, f_name)
        self.live = live

    def setup(self, live, f_name):
        pipeline = rs.pipeline()
        if live:
            self.get_live_pipeline(pipeline)
        else:
            self.get_video_pipeline(pipeline, f_name)
        
        return pipeline

    def get_live_pipeline(self, pipeline):
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

    def get_video_pipeline(self, pipeline, f_name):
        config = rs.config()

        # Tell config that we will use a recorded device from file to be used by the pipeline through playback.
        rs.config.enable_device_from_file(config, f_name)

        # Configure the pipeline to stream the depth stream
        # Change this parameters according to the recorded bag file resolution
        config.enable_stream(rs.stream.depth, rs.format.z16, 30)
        
        # Start streaming from file
        profile = pipeline.start(config)
        
        # set playback realtime to false -> otherwise it will drop frames to keep it at the same speed as 30fps
        playback = profile.get_device().as_playback()
        playback.set_real_time(False)
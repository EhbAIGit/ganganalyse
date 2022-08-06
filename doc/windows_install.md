# Final Work Jetson nano installation

1. Make sure no RealSense device is connected
2. Download & install latest .exe release from https://github.com/IntelRealSense/librealsense/releases/
3. This project is written for python 3.9 but is backwards compatible to 3.7 (potentially earlier version but this was not tested), at the moment of writing this `pyrealsense2` is updated for 3.9. Check https://pypi.org/project/pyrealsense2/#description for other versions
4. Download the latest release of this repo at https://github.com/TiboDeMunck/final-work/releases
5. Install the needed python packages with 
```sh
py -m venv venv
./venv/Scripts/activate
py -m pip install --upgrade -r requirements.txt
```
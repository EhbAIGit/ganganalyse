# Final Work Jetson nano installation

1. Make sure no RealSense device is connected
2. Open the terminal, run:
```sh
$ wget https://github.com/TiboDeMunck/final-work/blob/main/doc/jetson_nano_install.sh
$ chmod +x ./jetson_nano_install.sh
$ ./jetson_nano_install.sh
```
3. Wait untill "Final work installation complete"
5. Connect RealSense device
6. Run `rs-enumerate-devices` from the terminal to verify the RealSense installation


> At the moment, the script assumes Ubuntu 18 with graphic subsystem
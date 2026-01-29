# LCM adapter for unitree sdk2
## Build
```
cd unitree_sdk2
rm -rf build/
mkdir build && cd build
cmake ..
make -j8
cp bin/g1_control ~/
```
then run `./bin/g1_control eth0`
the lcm adapter will transmit low_cmd and low_state between policy and C++ sdk

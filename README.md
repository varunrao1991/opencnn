# opencnn
YOLO implementation using C++ with no additional libraries other than opencl

Steps:
1. Open visual studio solution located ./project/Segmentation.sln
2. Build in release mode.
3. Run run.bat
4. You should able to see bin files created under ./kernels folder
5. Verify that the file ./inputs/face.bmp is of size 416X416 is taken and output is generated under ./outputs/face.bmp

Note: These bin files are device specific. Do not forget to clean if you are moving to other device.

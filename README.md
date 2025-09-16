# Python based framework for indoor flight of Crazyflie using OptiTrack
This repository includes the Python codes needed to achieve controlled and stabilised flight of a Crazyflie drone in an indoor flight arena, using OptiTrack as a method for external localisation in GPS-denied environments.

To start with, NatNet SDK from OptiTrack website (https://optitrack.com/software/natnet-sdk/) will need to be downloaded onto the host PC. Once extracted, the folder "Samples -> PythonClient" will have 3 python codes named "DataDescriptions.py", "MoCapData.py", "NatNetClient.py" (also included in the repository). These python codes must be in the same folder as the main control codes.

Also, make sure you have downloaded the Crazyflie Python Library (cflib).

In Motive: Make sure to create a rigid body for the drone and check for no occlusions before running the code. Also label rigid body as "cf01", "cf02"...



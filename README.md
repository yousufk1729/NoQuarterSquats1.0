# NoQuarterSquats1.0
Squat depth checking with MediaPipe.   

## Overview
This is a quick program I put together to test high-level computer vision. I’m not happy with MediaPipe’s joint tracking and the knee depth heuristic that I used. When I tested the official IPF definition of squat depth (top of hip joint below top of knee joint) with a video with a better angle, the pose tracking was unable to consistently track the hip and knee joints as needed, so I just used knee angle. 

The examples in the data folder are from a random training video (335lb single). 

## References
- https://docs.python.org/3/library/venv.html 
- https://github.com/nicknochnack/MediaPipePoseEstimation 
- https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker 

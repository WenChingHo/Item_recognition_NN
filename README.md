# Real-time Object Recognition (openCV + YOLOv4)

## Table of contents
* [Intro / General info](#general-info)
* [Technologies](#technologies)


## General info: 
Create a model that can identify *sticky notes, colorful soap, and ACNH switch console* to learn aboutt he underlying process and technicality behind traiing a object detection NN model. <br><br>
<img src="https://github.com/WenChingHo/Item_recognition_NN/blob/main/outcome.png" width="350"> | <img src="https://github.com/WenChingHo/RT_Object_Detection/blob/main/RT_detection.png" width="350"><br><br>
\-Confidence with different depths  &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;-Confidence with hands holding the object (webcam)
### Process:
- Use ffmpeg to cut video recording of target objects into jpeg file 
- Use labelImg to label the objects
- Generate YOLOv4 readable xml annotation file and split train/test data
- Modify darknet/cfg/yolov4-tiny.cfg configuration file based on official recomendation
- Train model with the help of pretrained data: 
> $ DARKNET_HOME/darknet detector train obj.data yolov4-tiny-myobj.cfg yolov4-
tiny.conv.29 -map

<br><br>
## Technologies:
- Python 3.9
- OpenCV 4.5.2 ([Configuration](https://github.com/WenChingHo/RT_Object_Detection/blob/main/opencv_config.txt))
- [darknet (YOLOv4-tiny.weights)](https://github.com/AlexeyAB/darknet)
- [LabelImg](https://github.com/tzutalin/labelImg)
- ffmpeg
- Multiprocess
- Numpy



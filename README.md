# Object Detection with YOLO-V3 

YOLO (You Only Look Once) is a state-of-the-art real-time object detection system. YOLOv3 is an improved version of the YOLO algorithm that is faster and more accurate. In this project, we train YOLOv3 on the Pascal VOC dataset, which is a widely used benchmark dataset for object detection.

## Mosaic Augmentation
Mosaic data augmentation combines 4 training images into one in random proportions. The algorithms is the following:

* Take 4 images from the train set;
* Resize them into the same size;
* Integrate the images into a 4x4 grid;
* Crop a random image patch from the center. This will be the final augmented image.



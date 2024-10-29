# YOLOv7-dfc

This improved method for vehicle detection is based on the YOLOv7-tiny, It can be used in the same way as YOLOv7.

Specifically, you can view or apply our improved model by accessing the dfc-nn.yaml file under the path yolov7-dfc/cfg/deploy. This .yaml file can be called during training in train.py, where you may also select the appropriate dataset to test the model’s performance.

For certain modules and network structures in the improved model, you can find the defined functions in common.py under the path yolov7-dfc/models, as well as in loss.py and general.py under yolov7-dfc/utils. By modifying these files, you can replace corresponding modules and adjust the structure to implement other customized operations.

The model has been tested and confirmed to run successfully in the following environment:

PyTorch 1.11.0, 
Python 3.8, 
CUDA 11.3, 

For other virtual environments or hardware configurations with similar versions, there should generally be no compatibility issues. Please feel free to use it.
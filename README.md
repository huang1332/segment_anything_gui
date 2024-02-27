# segment_anything_gui



## 使用方法

这是一个类似PS的抠图工具，支持cpu和英伟达gpu。推荐opencv-python版本4.5.5.64（4.x大概都能跑）

使用方法

1.将待抠图的图片放到input文件夹中，然后启动程序。

https://github.com/facebookresearch/segment-anything
记得在这里下载vit_h的模型

2.选点模式（一个一个来，一次性多个点抠图效果一般）：

在图像上左键单击选择前景点（绿色），右键单击选择背景点（红色）。

按下a或d键切换到上一张或下一张图片。按下空格键清除所有选点和mask。按下q键删除最后一个选点。

按下s键保存抠图结果（如果有生成过Mask的话）。


3.Mask选取模式：

按下w键使用模型进行预测，进入Mask选取模式。

在Mask选取模式下，可以按下a和d键切换不同的Mask。

按下s键保存抠图结果。

按下w键返回选点模式，下次模型将会在此mask基础上进行预测

4.返回选点模式，迭代优化选点

把不需要的地方右键点一下，需要但mask没覆盖的地方左键点一下，几个点就行，别太多了

5.保存切出来的图片

程序将在output文件夹中生成抠好的图片，新切割出来图片的文件名会自增。

--- 
## **也可以直接将seg5.py文件复制到segment_anything项目下使用**

---



Segment-Anything-GUI

This is a Photoshop-like image segmentation and extraction tool, supporting both CPU and NVIDIA GPU. It is recommended to use the OpenCV-Python version 4.5.5.64 (4.x should also work).

How to Use

Place the images to be segmented into the input folder, then run the program.

Point selection mode (one by one, multiple points at once may have average results):

Left-click on the image to select foreground points (green).

Right-click to select background points (red).

Press the a or d key to switch to the previous or next image.

Press the spacebar to clear all selected points and masks.

Press the q key to delete the last selected point.

Press the s key to save the segmentation result (if a mask has been generated).

Mask selection mode:

Press the w key to use the model for prediction and enter the mask selection mode.

In the mask selection mode, you can press the a and d keys to switch between different masks.

Press the s key to save the segmentation result.

Press the w key to return to point selection mode. The model will predict based on this mask the next time.

Return to point selection mode to iteratively optimize selected points:

Right-click on the areas you don't need and left-click on the areas you need but are not covered by the mask. Just a few points are enough; don't use too many.

Save the cropped images:

The program will generate segmented images in the output folder. The file name of the new cropped image will increment automatically.

Notes

It is better to press the w key to predict after adding each point. If you want to run the program with the CPU, the speed will be much slower. Comment out the line with _ = sam.to(device="cuda").

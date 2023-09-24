# Hmm... What is this?

### Image Segmentation
To put plainly: *image segmentation is a pixel-wise mask for each of the objects present in an image.* This means we apply a mask (a filter of specific dimension) to separate each object we think is present in the image:![[Pasted image 20230919224708.png]]
These masks are obtained by predicting the class of each pixel in the image. This can be furthered to text and character recognition, where we segment the text from the background. 

In this proof-of-concept, this is done using a UNet, which is a form of CNN. It downsamples the image then upsamples it back to its original size, allowing the model to preserve important details of the image. 
#### SegmentPage.py
The main purpose of this file is to provide the ability to differentiate lines on a multi-line text document. These cropped lines can then be used to do text recognition later on with smaller samples. 

This uses `cv2` and `tensorflow`. In this proof-of-concept, this is done using a UNet, which is a form of CNN. It downsamples the image then upsamples it back to its original size, allowing the model to preserve important details of the image. 
This is a OCR - optical character recognition engine built on Python. It has the capability to input an entire multi-line text document and separate line-from-line. 

This program has (TBD) main components:
- SegmentPage
- BatchPadSeg
- ModelTest
- CRNN

This is subject to change as I may split the files in the future for ease of reading. Some basic concepts of OCR will be covered to help anyone better understand how this model works.
### Image Segmentation
To put plainly: *image segmentation is a pixel-wise mask for each of the objects present in an image.* This means we apply a mask (a filter of specific dimension) to separate each object we think is present in the image: ![Segmentation Example](/docs/assets/segmentation.png)

These masks are obtained by predicting the class of each pixel in the image. This can be furthered to text and character recognition, where we segment the text from the background. 
### SegmentPage.py
SegmentPage uses a UNet architecture to separate line-from-page and word-from-line. It has the following functions:
#### get_seg_img()
`get_seg_img(img, n_classes)` is a function to preprocess an given image to generate segmentation labels - in our case, text or background. We initial a three-dimensional numpy array of (512, 512, `n_classes`) to store the segmentation labels for the 512x512 labels we have. `img = img[:, :, 0]` extracts a single colour channel from the image. 

The heart of the function lies in `seg_labels[:, :, 0] = (img != 0).astype(int)`. Let's break this line down:
- `img != 0` creates a binary mask where pixels with a value of 0 are set to `False`, identifying non-zero pixels in the image.
- `astype(int)` converts the boolean mask to integers, where True is 1 and False is 0.
- These integer values are assigned to `[:, :, 0]` of `seg_labels`, labelling regions where pixel values are non-zero on the image.

#### batch_generator()
`batch_generator(file_list, batch_size, n_classes)` creates 'batches' of preprocessed images and their segmentation labels. The arrays `x` and `y` are used to store the input images and their segmentation labels.

It uses inverse binary thresholding. **Inverse binary thresholding** is a type of image processing that checks whether a pixel’s intensity exceeds a certain value. If it does, the pixel is changed to a black pixel and white if it does not. This process is called binarisation, and to inversely binarise an image is to perform the same thing but swapping white pixels with black. This helps separate the background from the foreground.

For each iteration:
1. The generator loops `batch_size` times to generate a batch of that particular size.
2. For each iteration, it randomly selects a file from `file_list`.
3. It applies inverse binary thresholding to this image. 
4. The image is resized and an channel is added to fit the input requirements of our UNet. 
5. It is normalised to values between 0 and 1 by dividing the array by 255.
6. `get_seg_img` is called to generate segmentation labels for the currently processed image.

It then yields the `x` and `y` arrays as tuples containing the batch of input images and their segmentation labels, ready to be used for training.
#### unet()
The UNet architecture was one proposed by Jonathan Long et al. in 2014, being an improvement over the existing CNN. It is frequently used in semantic segmentation - the act of labelling every pixel in an image. It is named as such for it's distinctive U-shape as it contracts and expands: ![UNet Diagram](/docs/assets/unet.png)

This model uses 5 encoder and 5 corresponding decoder blocks. At each encoder block, the spatial dimensions are halved and the number of filters are doubled. A filter, or a convolutional kernel, are small matrices applied to an image to extract features and recognise a specific pattern in said image. The decoders double the spatial dimensions and halve the filter size.

We then train the model:
```
model.fit_generator(batch_generator(file_train, 2, 2), steps_per_epoch=1000, epochs=3, callbacks=[mc], validation_data=batch_generator(file_test, 2, 2), validation_steps=400, shuffle=1)
```
#### segment_to_line()
The last function within this file takes an input image to extract and segment lines of text from it - the aforementioned line-from-page functionality. The same preprocessing takes place - the image is resized, inversely thresholded and dimensionally expanded. 

We call upon the UNet we just trained to predict line segmentation on the image, storing it in `pred`. The code normalises the `pred` image and applies two binary thresholding operations: `THRESH_BINARY` and `THRESH_OTSU`. Binary thresholding has been explained before, but Otsu's thresholding concept is new.

**Otsu's thresholding concept** obtains the distribution of pixels in the form of a histogram. The value of thresholding is not arbitrarily set, but instead computed. In the library we use (OpenCV), Otsu's thresholding is done as:
$$\sigma^2_{w}(t)=q_1(t)\sigma^2_{1}(t)+q_2(t)\sigma^2_2(t)$$

The final result, $\sigma^2_{w}(t)$, is the *weighted within-class variance at threshold $t$*. It wants to find the threshold value that minimises this variance, as a lower value implies better separation between classes.

$q_1(t)$ and $q_2(t)$ is the probability that a pixel belongs to the foreground or background, where foreground is class 1 and background is class 2, at threshold value $t$.

$\sigma^2_{1}(t)$ and $\sigma^2_{2}(t)$ represent the variance of the pixel intensity in classes 1 and 2 at threshold value $t$. 

Otsu's method then exhaustively tries all possible thresholding values between the maximum and minimum pixel intensities in the image. 

We find contours in the generated thresholded image. Coutours are connected regions in the image, which are lines of texts in this context. Rectangles to define the coordinates of each line are calculated using `cv2.boundingRect`.
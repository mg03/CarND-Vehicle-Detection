##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/samples_car_non_car.jpg
[image2]: ./examples/car_non_car_hog.jpg
[image3]: ./examples/seq1.jpg
[image4]: ./examples/seq2.jpg
[image5]: ./examples/seq3.jpg
[image6]: ./examples/seq4.jpg
[image7]: ./examples/seq5.jpg
[image8]: ./examples/seq6.jpg
[image9]: ./examples/seq7.jpg
[image10]: ./examples/seq8.jpg
[image11]: 
[image12]: 

[video1]: ./project_video_out1.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the code cell 10 of the IPython notebook.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I used color space `YCrCb` and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters especially pixels_per_cell and cell_per_block and the
parameters I choose worked best for in terms of detection on test images

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I used all hog_featutes, hist_features and spatial_features. Then obtained a scaled feature vector. Trained the featurea and label vector using LinearSVC. Saved the classifier ouput and other params to a pickle file for reuse. Code is in cell 12

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

I used diff window positions of diff scales:

1) 
ystart = 400
ystop = 656
scale = 1.5
2)
ystart = 400
ystop = 500
scale = 1.0

I thought of using 0.5 but that just added time with no other percievable benefit

I generated heatmap with threshold < 1 and labels.

![alt text][image2]

### Here are some test images and their corresponding windows, heatmaps, labels and detections:

For test_image1.jpg
![alt text][image3]
![alt text][image4]

For test_image2.jpg
![alt text][image5]
![alt text][image6]

For test_image3.jpg
![alt text][image7]
![alt text][image8]

For test_image4.jpg
![alt text][image9]
![alt text][image10]

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_out1.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

In cell 16, functions add_heat, apply_threshold, draw_bboxes. IN cell 30 and 38, Class Vehicle_Detection and function process_with_smooting_v1

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected. 


---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Using SVM based approach, the accuray is good, but the speed is an problem as sliding window approach is time consuming! We could use image downsampling, multi-threads, or GPU processing to improve the speed. But, it would need a lot engineering work to make it run in real-time. 

Maybe a faster algorithm/methodology which is Neural Net based like YOLO would be better in real-time scenarios.

Overall, this project gave me insights into the varius tuning paramters and an understanding of classifiers.

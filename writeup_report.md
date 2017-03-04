**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[undistorted-straight]: ./output_images/1-undistorted/straight_lines1.jpg "Undistorted Straight"
[binary-straight]: ./output_images/2-binary/straight_lines1.jpg "Binary Straight"
[birdseye-straight]: ./output_images/3-birdseye/straight_lines1.jpg "Bird's Eye Straight"
[windows-straight]: ./output_images/4-windows/straight_lines1.jpg "Windows Straight"
[reverse_warped-straight]: ./output_images/6-reverse_warped/straight_lines1.jpg "Reverse Warped Straight"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.

You're reading it!
###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "./CarND-Advanced-Lane-Lines.ipynb".  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained these results: 

![alt text][undistorted-straight]

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.

See above for an example of distortion-corrected images.

####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image. Find the code for this under the title "Use color transforms, gradients, etc., to create a thresholded binary image." The function is called `get_binary_image()`.

I converted colors to HLS space because hue and saturation were the best at picking out lane lines. Gradient thresholds helped a little to pick out some of the more subtle sections of lane line.

![alt text][binary-straight]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform is in the code cell below "Apply a perspective transform to rectify binary image ("birds-eye view").". The function is called `warp()`. 

I started with similar src and dst points from the provided "example_writeup.pdf", and modified them slightly to get the best results on straight lines.

```
src = np.float32(
    [[(img_size[0] / 2) - 62, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 65), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 578, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 705, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][birdseye-straight]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

You can find this code under heading "Detect lane pixels and fit to find the lane boundary." in functions `find_lanes()` and `find_lanes_with_hint()`.

The code calculates a histogram of pixels column by column. The peaks in each half of the image are assumed to be the starting point for the left and right lanes. Next, it fits 9 sliding windows along the length of each lane. Proceeding from the bottom, if there are enough pixels within the window, then it slides the window to the mean of those pixels.

Finally, the function fits a second order polynomial to each set of lane pixels using `np.polyfit()`.

Below is an example of identifying lane-line pixels via a sliding window.

![alt text][windows-straight]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

You can find this code under heading "Determine the curvature of the lane and vehicle position with respect to center." in functions `calc_radius_of_curvature_meters()` and `calc_center_offset_meters()`.

For radius of curvature, the function converts lane pixels into real-world points, then fits a 2nd-order polynomial. The radii of curvature are calculated using the formula found [here](http://www.intmath.com/applications-differentiation/8-radius-curvature.php) and described in the lectures. I averaged the left and right radii of curvature to arrive at a single value per frame.

For the vehicle's offset from center, `calc_center_offset_meters()` calculates the x-value for each lane at the bottom of the image then averages them to find the middle of the lane. It assumes the middle of the image is the middle of the vehicle, and calculates the offset by subtracting the lane center from the image center. Finally, the offset is converted from pixels to meters.

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

You can find this code under heading "Warp the detected lane boundaries back onto the original image." in function `reverse_warp()`.

Here is an example of my result on a test image:

![alt text][reverse_warped-straight]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_output_smoothed.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The main problem I faced during implementation was creating the thresholded binary image. I didn't know what threshold values to use and how best to combine the numerous binary images. 

To find the right combinations of thresholds, I experimented with threshold values and with various combinations of binary images. As test input, I used the images under the folder 'test_images'. 

Converting the color image to HLS space (hue, lightness, and saturation) extracted the most useful information. The S channel was the clearest indication for lane lines in the test images. The H channel restricted to the yellow band was good at picking out yellow lane lines. I combined each channel with the L channel via a bitwise-and operator. The foundation of my lane-line picking algorithm was: `(H AND S) OR (L AND S)`, which simplifies to `(H OR L) AND S`. This did the best job of picking out lane lines.

I incorporated the x, y, magnitude, and direction of the gradient via the sobel operator because it was good at picking up smaller sections of dashed lane lines. My expression for this portion was `(magnitude AND direction) OR (x-gradient AND y-gradient)`.

Another problem I found was that the radii of curvature and offset from center calculated by my `Pipeline` class were changing a fair bit between video frames. I implemented a `Smoother` class that kept a rolling window of lane lines and returned an average of the recently found lane lines. I tried different window sizes and decided on 5 frames.

My pipeline will likely fail in conditions where lane lines are less prominent:
* Faded lane lines
* Snow
* Wet road causing reflections on ground
* Speckled road
* Roads with paint test strips

To make it more robust, I could:
* Update `Smoother` class to weight recent images.
* Find more challenging test images and tweak thresholds to perform well on difficult images.
* Use Hough transform to find lines and cluster lines based on position and slope.
* Incorporate a filter that rejects lane lines that are too different from recently found lane lines.

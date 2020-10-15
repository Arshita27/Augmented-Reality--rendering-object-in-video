# Augmented Reality: Render 3D object in a video frame
<b>(Work in Progress) </b>


This repository is a simple implementation of rendering 3D objects in a video frame.
It is insipired by this work: [Augmented reality with Python and OpenCV](https://bitesofcode.wordpress.com/2017/09/12/augmented-reality-with-python-and-opencv-part-1/)

#### Details:
The link provided above gives a very detailed explanation on this topic. To cut it short, given below are some important points that covers the basic idea of this project.

1. Capture a reference image (a 2D flat surface) on which you wish to render your 3D object on (shown below is an example of how it should look like).

  <img src="/test/ref.jpg" height="160" width="120">

  (ps: this is not a paid promotion :P)

2. Capture a video frame that contains the above mentioned refernce image.
3. Download an .obj file. (I used clara.io which was also recomended in the website above).
  1. what is a .obj file ?
    - [Wiki link](https://en.wikipedia.org/wiki/Wavefront_.obj_file#File_format)

4. Extract __feature keypoints__ and __feature descriptors__ from the refernce image.
5. Loop over every frame in the video:

  1. Extract feature keypoints and feature descriptors.
  2. Find __best matches__ between the descriptors from reference image and descriptors extracted from video frame.
  3. Estimate the __homography matrix__ based on these matched descriptors (and keypoints). This homography matrix basically tells us the transformation that the refernce image went through in the video frame.

6. Once we have this transformation matrix, we can trasnform the 3D object to correctly match the orientation of the refrence flat surface in the video frame.
7. Next, is the tricky part.

    Let's start by first describing the homography matrix mentioned above:

    <figure>
      <img src="../extra/homography.png" height="125" width="229"  >
      <font size="1">
      <figcaption>Source: F. Moreno </figcaption>
    </figure>

    We have the calibration matrix or Intrinisc matrix (blue shaded)and external calibration matrix or extrinic matrix (red shaded). The extrinsic matrix is made up of rotation and translation matrix. On the left hand side we have the u, v coordinates (in the image plane) of a given point p (any 3D point denoted as [x,y,z]) expressed in the camera coordinate.  The combination of intrinsic and extrinsic camera parameters is called the projective/homography matrix.

    <img src="/home/arshita/Desktop/Screenshot at 2020-10-14 19:37:21.png", width="349", height="212">

    We  assume that any position on the flat surface plane (reference image) can be described by a 3D position <b>p</b> = [x, y, z]<sup>T</sup>. Here, the  z-coordinate represents directions perpendicular to the plane, and is hence always zero. This modifies the above equation to <b>p</b> = [x, y, 0]<sup>T</sup>.

    Due to the above reason, we drop the third column in the rotation matrix as the z-coordinate of all the points we wanted to map was 0.

    <figure>
      <img src="../extra/selection_003.png" width="297" height="125" >
      <font size="1">
      <figcaption>Source: F. Moreno </figcaption>
    </figure>

    However, in order to project all the points in the 3D object, we now want to project points whose z-coordinate is different than 0.

    <i>
    The basic idea now is to first extract [R1, R2, t] in the extrinsic camera matrix by multiplying the inverse of the intrinsic camera matrix (assuming we know this) with the homography matrix that we calculated earlier.
    </i>

    <i>
    We then find a new pair of orthonormal basis similar to (R1, R2) and then computer R3. [R1, R2, R3, t] is the new extrinsic matrix.
    </i>

    <i>
    Combining this extrinsic matrix with our previously mentioned intrinsic matrix gives us a new homography matrix that will help us to place any point of the 3D object in our video frame.
    </i>

    <b>If this was difficult for you to understand, I highly recommend visiting the website mentioned above </b>

#### Requirements:


#### Implementation Details:

I have tried to make this repository as simple as possible. There are only two things to keep in mind while running this repository.

1. __config.yml__ file:

This yaml file contains the following fields:

* DATASET:

  - INPUT_DIR: Path to the folder where all the images are stored. (default=test)
  - REF_IMG: The reference image that we are looking for in every video frame.
  - VIDEO_PATH: path where the video file is saved.
  - OUTPUT_DIR: Path to the folder where all the results should be stored. (default=results)
  - RENDERED_OBJ: path to the .obj file wthat you wish to render in the video frame.

* FEATURES:
  - FEATURE_DESCRIPTORS: Default is set (Other choices are provided in comments)
  - FEATURE_MATCHING: Default is set (Other choices are provided in comments)
  - FEATURE_MATCHING_THRESHOLD: Default is set (Other choices are provided in comments)

* CAMERA_PARAMETERS:
  - INTRINSIC: intrinsic camera parameters which is a 3x3 matrix (list of list here)

* RENDERING:
  - SCALE_FACTOR:

One can simply change the parameters in the config file to try the effect of the different techniques.

Command to run the program python -m run --c [path to config.yml]

I have kept the path to config.yml as an argument so that the user can have multiple config files corresponding to different projects (with different images and varied feature attributes)



<!-- <p float="left">
  <img src="/test/campus_001.jpg" height="320" width="240">
  <img src="/test/campus_002.jpg" height="320" width="240">
  <img src="/test/campus_003.jpg" height="320" width="240">
  <img src="/test/campus_004.jpg" height="320" width="240">
</p> -->

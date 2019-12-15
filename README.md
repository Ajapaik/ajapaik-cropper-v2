# Introduction to Data Science project

Introduction to Data Science project about automaticaly cropping archival images, so there is no photo border in the image. Team: Alvin Meltsov, Hain Zuppur.

Necessary dependencies for running the code:

* cv2 `pip install opencv-python`

* cv2 `pip install scikit-image`

* argh `pip install argh`

... and ofcourse:

* `pip install numpy pandas` 

## Running the script

To run the algorithm on a single image, run the edge_kernel_single.py and pass the image file name in as a argument.

```
python edge_kernel_single.py [your file]
```

There are optional parameters for the kernel passed on with argh:

* `--rotation [even int]` an even number specifing the rotational range trialed with this algorithm. When using large pictures and you know the angle, set it to smaller number. Default 20 degrees (-10 .. 10).

* `--minlen [float]` float between 1 .. 0 which marks how long lines in relation to the length of the axis of the picture should be detected from the picture. Default 0.59.

* `--thresh [float]` float value which specifies the thresholding z-score where image is cropped. The higher the score the less the chance of cropping. Default 1.98.

* `--sizediff [int]` integer above 1. How much of size of the image must be preserved to crop. If the frame is on average 1/4 of the image, you don't want to set it to 8, which means that only 1/8 of the picture can remain.

To run the algorithm on a directory of images, run the run_batch.sh and pass the directory in as a argument.

```
./run_batch.sh [directory of your pictures]
```


# Stereo Tuner
This is my modified version of [Martin Peris](http://blog.martinperis.com/)'s [StereoBM Tuner](http://blog.martinperis.com/2011/08/opencv-stereo-matching.html).

This is a simple little GTK application that can be used to tune parameters for the [OpenCV](http://opencv.org/) Stereo Vision algorithms.

## Screenshot
![Screenshot](screenshot.png)

## New features
- **New algorithms:** this application supports both the StereoBM and StereoSGBM algorithms
- **Tooltips:** the parameter labels now display tooltips explaining them. Some of them were taken from the OpenCV documentation, and the ones that are not explained there were taken from somewhere else.
- **Execution time:** a (not very useful) indicator of the algorithm execution time on the status bar
- **New Glade file:** the Glade file was recreated from scratch and works with the recent versions of Glade.
- **OpenCV 3.0:** the program now uses OpenCV 3.0 and its C++ API (no more `IplImage`s).

## Installation
Make sure you have GTK3.0, GModule2.0 and OpenCV3.0 installed on your system, as well as a C++ compiler. Then, execute the following:

    wget  https://github.com/guimeira/stereo-tuner/releases/download/v0.1/stereo-tuner.tar.gz
    tar zxvf stereo-tuner.tar.gz
    cd StereoTuner
    make
    ./main
    
## Usage
The user interface should be very intuitive, all you have to do is to adjust the parameters and see the resulting disparity image instantly.

You can use your own pair of images by using the command line parameters `-left` and `-right`. For example:

    ./main -left my_left_image.png -right my_right_image.png
    
Make sure your images are undistorted and rectified.
## Future work
There's a lot of stuff that I'd like to do to improve this application, but I'm not sure if/when I'll have time to do that. Here's a list of new features that could be interesting:
- Select left and right images on the GUI
- Use other sources (webcams, video files, etc)
- Save the parameters in the format that can be loaded by the `read` method of `StereoBM` and `StereoSGBM`
- Read parameters in that same format
- Binary releases (.deb, .rpm, maybe even Windows)
- Add support for other stereo-related stuff such as camera calibration, rectification, undistortion, etc, and then give this application some fancy name

## Bugs, issues, new features
Please, feel free to open an issue if you find a bug or you have a feature request. You can also fork this project and submit a pull request.

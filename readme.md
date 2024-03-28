# Project3_YZ: Project 4: Calibration and Augmented Reality

[Project Repository](https://github.com/jeff8971/Project4_YZ)

[DEMO](https://drive.google.com/file/d/1jM_PgVTPZHYQJVXBxExOj3yt8gHMWUUf/view?usp=drive_link)

## Overview
Project4_YZ is developed as part of an advanced computer vision course, focusing on camera calibration, feature detection, and model projection. The project aims to implement a system capable of calibrating a camera using a chessboard pattern, detecting robust features in images, and visualizing 3D models in the context of augmented reality.


## System Environment
- **IDE**: Visual Studio Code or any preferred C++ IDE
- **Compiler**: C++ Compiler supporting C++20 standard
- **System Configuration**: Compatible with various operating systems, including macOS, Linux, and Windows.
- **Dependencies**: Requires OpenCV for image processing and feature extraction functionalities.

## Project Structure
- `src/`: Source files implementing the core functionality of the project.
  - `calibrate.cpp`: Implements camera calibration using a chessboard pattern.
  - `calibrate_utils.cpp`: Utilities for camera calibration.
  - `csv_utils.cpp`: Utilities for handling CSV files.
  - `feature.cpp`: Implements robust feature detection.
  - `show_model_utils.cpp`: Utilities for visualizing 3D models.
  - `show_model.cpp`: Visualizes 3D models in the context of augmented reality.
- `include/`: Contains header files for the project.
- `bin/`: Executable files generated after building the project are stored here.
- `data/`: Contains calibration data, 3D models, and other resources used by the project.
- `CMakeLists.txt`: Configuration file for building the project using CMake.


## Features
- **Camera Calibration**: Calibrate a camera using images of a chessboard pattern to compute the camera matrix and distortion coefficients.
- **Feature Detection**: Detect robust features in images using methods like SURF, SIFT, Harris corner detection, and Shi-Tomasi corner detection.
- **3D Model Visualization**: Visualize 3D models in the context of augmented reality by overlaying them on a calibrated scene.
- **CSV Utilities**: Tools for efficiently handling CSV files containing calibration data and feature information.


## Getting Started
### Prerequisites
- Install [OpenCV](https://opencv.org/releases/) library.
- Install CMake.
- Ensure C++20 support in your compiler.

### Installation
1. Clone the repository:
```git clone https://github.com/jeff8971/Project4_YZ.git```
2. Navigate to the project directory:```cd Project4_YZ```
3. Build the project using CMake:
```
cd build
cmake ..
make
```


### Running the Applications
After building, the project generates executables for different tasks within the `bin/` directory:
- `./calibrate`: For performing camera calibration using a chessboard pattern.
  - The `calibrate.csv` file in the `data/` directory contains the image paths for calibration.
- `./features`: For detecting robust features in images.
- `./show_models`: For visualizing 3D models in the context of augmented reality.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Utilization of Travel Days
3 days




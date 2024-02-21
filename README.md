# Player Detection

This project is part of the Signal, Image, and Video Processing course taught by Prof. Andrea Rosani for the academic year 2023/2024. Developed by Lorenzo Gandini.
As an introduction to Computer Vision, my goal was to apply traditional methods (Histogram of Oriented Gradients, CAMShift and Kalman filters) to achieve the result. 

If you want to discuss this project with me, free to write me an email! :)

## Introduction

This repository contains a Python application designed to detect players in a selected video file and track their movements by drawing bounding boxes around them.

## Getting Started

### Prerequisites

Ensure you have Python installed on your system. This project requires the use of a virtual environment to manage dependencies.

### Installation

1. Clone the repository to your local machine.
2. Navigate to the cloned directory and create a virtual environment:

python -m venv venv

3. Activate the virtual environment:
- On Windows:
  ```
  venv\Scripts\activate
  ```
- On Unix or MacOS:
  ```
  source venv/bin/activate
  ```
4. Install the required libraries:

pip install -r requirements.txt


### Running the Application

Execute the main script to start the player detection and tracking process:

python src/main.py

The results will be saved in the `Results` directory.

## Project Structure

- `/Data`
  - `/bounding-boxes/frame.json`: Contains the JSON with detected coordinates corresponding to the video in results. This file will be overwritten with new detections upon subsequent runs.
  - `/video/Bundes clip.mp4`: The input video file.
- `/Results`
  - `/video/player-detection.avi`: The final result of the analysis.
- `/src`
  - `drawVideo.py`: Functions for drawing bounding boxes on the video.
  - `function.py`: Common utility functions.
  - `HOG.py`: Functions for detecting new players with HOG.
  - `KALMShift.py`: Functions for tracking bounding boxes in memory.
  - `main.py`: The main script to run the analysis.
  - `matchBoxes.py`: Functions for matching boxes from HOG detection and the tracker.
  - `utilities.py`: File containing variable declarations, kernels, and paths.
- `requirements.txt`: Libraries required for the virtual environment.
- `README.md`: This document.
ANTs registration UI - graphical interface to the Advanced Normalization Toolbox

## Requirements

Python 3.8 or higher is required to run this user interface

## Installation

In your python environment run `pip install git+https://github.com/thladnik/ants_registration_ui` in a terminal/command prompt

## Starting the user interface

From the terminal/command prompt call `antsui`

## Basic usage

This simple user interface is designed to help with anatomical registration, using the Advanced Normalization Toolbox for Python ([ANTsPy](https://github.com/ANTsX/ANTsPy)), for users without programming experience or in cases where registration results may rely heavily on the initial transform. 

* 3D alignment: align the moving stack (green) to the reference stack (red) using keyboard controls:
  * W/A: front/back
  * A/D: left/right
  * X/C: up/down
  * Q/E: rotate around Z-axis CCW/CW
* 2D alignment: same controls as in 3D alignment. View is based on selected layer in the moving stack (slider)
* Show 2D alignment: map the moving stack into the references stack coordinate system using the ANTs transform that will be used as init_transform when running the registration
* Configure registration: set all ANTs registration parameters and run the transform
* Show 2D registration: display the registration result in 2D
* Show 3D registration: display the registration result in 3D


![image](https://github.com/user-attachments/assets/ecb1044a-5a0e-4280-8809-c8364d2f82fd)

![image](https://github.com/user-attachments/assets/58c5cfa9-0ecc-4a51-b497-1b76b3469a7b)

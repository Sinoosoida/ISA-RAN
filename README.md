
# ISA-RAN Model Repository

This repository contains the model for semantic segmentation of olfactory bulbs on phase-contrast computed tomography scans, along with the necessary script to run the model.

## Prerequisites

Ensure that you have Python 3.11 installed. While other versions may work, we recommend Python 3.11 as it was used for testing.

## Setup Instructions

### Cloning the Repository

To clone the repository, open a terminal and execute the following command:
```
git clone https://github.com/Sinoosoida/ISA-RAN.git
```

### Navigating to the Project Directory

After cloning, navigate to the project directory:
```
cd ISA-RAN
```

### Installing Dependencies

Install the required libraries using the following command:
```
pip install -r requirements.txt
```
If you encounter issues with installing dependencies and cannot install a library of the required version, the code will likely work with libraries of other versions, as long as all necessary libraries are installed.

### Running the Script

To run the script, use the Python interpreter. Execute the following command:
```
python3 ./main.py
```
For Windows users, ensure you are using the appropriate command prompt (e.g., Command Prompt or PowerShell) and you may need to use `python` instead of `python3`.

## Directory Structure

- `main.py`: The main script to run the model.
- `model_state.pth`: The saved state of the model.
- `raw_images`: This directory contains the images to be processed by the model.
- `README.md`: This file.

## Additional Notes

- Ensure your working directory is set correctly before running the script.
- If you encounter any issues related to permissions, try running the commands with appropriate administrative rights.
- This script has been tested on Linux, but it should also work on other operating systems with Python installed.
- The images contained in `raw_images` have a specific structure. Although the images have the .tiff extension, they cannot be viewed in a regular photo editor. Therefore, the code includes the ability to read both .png and .jpg images. Additionally, after execution, the code saves the original dataset image it processed in .png format.
- For additional dataset images and further collaboration, please contact m.chukalina@smartengines.ru.

Please follow these instructions carefully to ensure successful execution of the model. If you have any questions or encounter any issues, feel free to reach out for support.


# ISA-RAN Model Repository

This repository contains the ISA-RAN model described in the accompanying article, along with the necessary script to run the model.

## Prerequisites

1. Ensure that you have Python 3.11 installed. While other versions may work, we recommend Python 3.11 as it was used for testing.
2. Install the required libraries listed in `requirements.txt`.

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
- `segmented_images`: Processed images will be saved in this directory as the script runs.
- `README.md`: This file.

## Additional Notes

- Ensure your working directory is set correctly before running the script.
- If you encounter any issues related to permissions, try running the commands with appropriate administrative rights.
- This script has been tested on Linux, but it should also work on other operating systems with Python installed.

Please follow these instructions carefully to ensure successful execution of the model. If you have any questions or encounter any issues, feel free to reach out for support.

# AI Recovery Exercise & Rehabilitation Assistant

A computer vision-powered fitness and rehabilitation assistant using pose estimation and machine learning to create personalised recovery plans, analyse exercise form, and provide real-time feedback through a desktop app.

## Features

- Personalised recovery plans: Generates a recovery or workout plan from user input using a decision tree classifier.
- Webcam-based exercise tracking: Uses MediaPipe AI pose estimation to analyse user movement and posture in real time.
- Form feedback and rep counting: Automatically detects reps/holds and provides suggestions on exercise form.
- Tkinter GUI: User-friendly interface for injury input, plan review, and exercise session control.
- Custom fault detection and scoring: Applies biomechanical rules to give tailored feedback on movement quality.

## Prerequisites

- **Python 3.10 or 3.11 (64-bit)** is required for package compatibility, especially with MediaPipe. Python 3.12 or newer is not currently supported.
- A working webcam
- pip (Python package installer)

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/MitulAggarwal1/KI-Challenge.git
cd KI-Challenge
```

### 2. Install Python 3.10 or 3.11

Download and install Python 3.10 or 3.11 from the official site: https://www.python.org/downloads/windows/

Make sure to check **Add Python to PATH** during installation.

### 3. Create and activate a virtual environment with the correct Python version

On Windows PowerShell:

```bash
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
```

*If the `py -3.11` command fails, specify the full path to the Python 3.11 executable.*

On Mac/Linux:

```bash
python3.11 -m venv venv
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

If you encounter build errors on Windows, install the Microsoft C++ Build Tools with the "Desktop development with C++" workload and restart your terminal before retrying the installation.


## Running the App

```bash
python main.py
```

- Complete the GUI pain/symptom survey.
- Review the recommended plan.
- Select an exercise to begin real-time tracking and feedback.

## Troubleshooting

- If you see errors about missing modules, ensure your virtual environment is activated and run `pip install -r requirements.txt` again.
- If the webcam does not work, ensure it is not being used by another application.
- If you encounter errors installing MediaPipe, verify you are running Python 3.10 or 3.11.

## Project Structure

- `main.py` – Main app logic and GUI  
- `requirements.txt` – Python packages  
- `.gitignore` – Virtual environment and OS file exclusions

## Acknowledgments

- [MediaPipe](https://mediapipe.dev/) for pose estimation  
- scikit-learn for machine learning plan classifier  
- OpenCV for video processing

## License

Please contact the author for reuse or collaboration.

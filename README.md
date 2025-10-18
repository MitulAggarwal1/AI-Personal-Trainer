AI Recovery Exercise & Rehabilitation Assistant

A computer vision-powered fitness and rehabilitation assistant using pose estimation and machine learning to create personalised recovery plans, analyse exercise form, and provide real-time feedback through a desktop app.

Features

- Personalized recovery plans: Generates a recovery or workout plan from user input using a decision tree classifier.
- Webcam-based exercise tracking: Uses MediaPipe AI pose estimation to analyze user movement and posture in real time.
- Form feedback and rep counting: Automatically detects reps/holds and provides suggestions on exercise form.
- Tkinter GUI: User-friendly interface for injury input, plan review, and exercise session control.
- Custom fault detection and scoring: Applies biomechanical rules to give tailored feedback on movement quality.

Setup Instructions

Prerequisites

- Python 3.10+ (recommended for package compatibility)
- A working webcam
- pip (Python package installer)

Clone the repo

```bash
git clone https://github.com/MitulAggarwal1/KI-Challenge.git
cd KI-Challenge
```

Create and activate a virtual environment (Windows)

```bash
python -m venv venv
venv\Scripts\Activate.ps1     Or: venv\Scripts\activate.bat in CMD
```

(Mac/Linux users: `source venv/bin/activate`)

Install dependencies

```bash
pip install -r requirements.txt
```
Note for Windows users:
If you encounter build errors during pip install, please install the Microsoft C++ Build Tools with the "Desktop development with C++" workload selected. Restart your terminal afterwards and retry the install. This is required for building some Python packages with native code extensions.

Required file

- The model file `movenet_thunder.tflite` must be present in the project root directory. This file is used for pose estimation AI.



Running the App

```bash
python main.py
```

- Complete the GUI pain/symptom survey.
- Review the recommended plan.
- Select an exercise to begin real-time tracking and feedback.



Troubleshooting

If you see errors about missing modules, ensure your virtual environment is activated and run `pip install -r requirements.txt` again.

If the webcam does not work, make sure it is not being used by another application.



Project Structure

- `main.py` – Main app logic and GUI  
- `requirements.txt` – Python packages
- `movenet_thunder.tflite` – AI pose estimation model file
- `.gitignore` – Virtual environment and OS file exclusions



Acknowledgments

- [MediaPipe](https://mediapipe.dev/) for pose estimation
- scikit-learn for the machine learning plan classifier
- OpenCV for video processing



License

This project is submitted as part of the MIT Maker Portfolio. Please contact the author for reuse or collaboration.


import tkinter as tk
from tkinter import messagebox, scrolledtext
import numpy as np
import cv2
import mediapipe as mp
import time
import pyttsx3
import re


# ============================
# 1. HARD-CODED PLANS DICTIONARY
# ============================


plans = {
    # Lower Back Pain Plans
    'plan_rest_doctor': (
        "🛑 Rest & See Medical Professional\n"
        "- Avoid strenuous activities\n"
        "- Use ice, monitor symptoms\n"
        "- Seek immediate consultation\n"
    ),
    'plan_lower_back_gentle': (
        "🧘 Lower Back Gentle Recovery (7 days)\n"
        "- Cat-Cow stretch 3x10 reps\n"
        "- Pelvic Tilts 3x10 reps\n"
        "- Child’s Pose hold 30s × 3\n"
        "- Knee-to-Chest & Hamstring stretches gradually\n\n"
        "🏋️ Beginner Reintroduction:\n"
        "- Week 2: Push-ups 3x5 reps\n"
        "- Week 3: Light deadlifts 2x8 reps, goblet squats 2x10 reps\n"
    ),
    'plan_lower_back_strengthen': (
        "💪 Lower Back Strengthening\n"
        "- Deadlifts (light) 3x8 reps\n"
        "- Bird-Dog 3x10 per side\n"
        "- Planks 3x20 sec hold\n"
        "- Gradually increase weights and sets\n"
    ),


    # Hip Pain Plans
    'plan_hip_gentle': (
        "🧘 Hip Mobility & Gentle Stretch\n"
        "- Hip Flexor stretch 3x30s per side\n"
        "- Figure-4 stretch 3×30s per side\n"
        "- Avoid deep squats for 1 week\n\n"
        "🏋️ Beginner Reintroduction:\n"
        "- Week 2: Light deadlifts 2x10 reps\n"
        "- Week 3: Glute bridges and step-ups 2x12 reps\n"
    ),
    'plan_hip_strengthen': (
        "💪 Hip Strengthening Plan\n"
        "- Glute bridges 3x12 reps\n"
        "- Step-ups 3x10 reps per leg\n"
        "- Bodyweight squats 3x10 reps\n"
        "- Introduce resistance gradually\n"
    ),


    # Leg Pain Plans
    'plan_leg_gentle': (
        "🧘 Gentle Leg Recovery\n"
        "- Hamstring stretch hold 30s × 2 per leg\n"
        "- Quad stretch hold 30s × 2 per leg\n"
        "- Calf raises 2x15 reps\n\n"
        "🏋️ Beginner Reintroduction:\n"
        "- Week 2: Bodyweight squats 3x10\n"
        "- Week 3: Controlled lunges 2x10 each leg\n"
    ),
    'plan_leg_strengthen': (
        "💪 Leg Strengthening\n"
        "- Squats 3x8-12 reps\n"
        "- Lunges 3x10 reps each leg\n"
        "- Calf raises 3x15 reps\n"
        "- Gradual load/increase based on tolerance\n"
    ),


    # Shoulder Pain Plans
    'plan_shoulder_gentle': (
        "🧘 Shoulder Stretch & Mobility\n"
        "- Cross-body shoulder stretch hold 30s each side\n"
        "- Wall slides 2x10 reps\n"
        "- Avoid overhead pressing for 1 week\n\n"
        "🏋️ Beginner Reintroduction:\n"
        "- Week 2: Incline push-ups 3x8 reps\n"
        "- Week 3: Light shoulder press 2x10 reps\n"
    ),
    'plan_shoulder_strengthen': (
        "💪 Shoulder Strengthening\n"
        "- Push-ups 3x10 reps\n"
        "- Bicep curls 3x12 reps\n"
        "- Shoulder press 3x10 reps\n"
        "- Add dumbbell rows week 2\n"
    ),


    # Arm Pain Plans
    'plan_arm_gentle': (
        "🧘 Arm Recovery & Mobility (7 days)\n"
        "- Wrist circles 3x15 reps\n"
        "- Gentle arm raises 3x10 reps\n"
        "- Avoid heavy lifting first week\n\n"
        "🏋️ Beginner Reintroduction:\n"
        "- Week 2: Light bicep curls 3x8 reps\n"
        "- Week 3: Shoulder press 3x8 reps\n"
    ),
    'plan_arm_strengthen': (
        "💪 Arm Strengthening\n"
        "- Bicep curls 3x12 reps\n"
        "- Shoulder press 3x10 reps\n"
        "- Push-ups 3x10 reps\n"
        "- Progressive weight increase\n"
    ),


    # Stretch Plans
    'plan_hamstring_stretch': (
        "🧘 Hamstring Stretch\n"
        "- Hold for 30 seconds × 3 sets per leg\n"
        "- Focus on straight leg and gentle bend at hips\n"
    ),
    'plan_quad_stretch': (
        "🧘 Quad Stretch\n"
        "- Hold for 30 seconds × 3 sets per leg\n"
        "- Keep knees aligned and pull heel close to glute\n"
    ),
    'plan_shoulder_stretch': (
        "🧘 Shoulder Stretch\n"
        "- Cross arm over chest and hold 30 seconds × 3 sets\n"
        "- Maintain shoulder alignment\n"
    ),
    'plan_triceps_stretch': (
        "🧘 Triceps Stretch\n"
        "- Raise arm overhead, bend elbow and press down\n"
        "- Hold for 30 seconds × 3 sets\n"
    ),
    'plan_hip_flexor_stretch': (
        "🧘 Hip Flexor Stretch\n"
        "- Lunge forward and push hips down\n"
        "- Hold for 30 seconds × 3 sets per side\n"
    ),
    'plan_cat_cow_stretch': (
        "🧘 Cat-Cow Stretch\n"
        "- Alternate arching and dipping back\n"
        "- Perform 10 cycles slow and controlled\n"
    ),
}


# ============================
# 2. PLAN SELECTION LOGIC (RULE-BASED)
# ============================


def select_plan_key(severity, duration, pain_type, location, activity, can_walk, history, improving):
    sev = severity.lower()
    loc = location.lower()


    if sev == "severe":
        return 'plan_rest_doctor'


    if loc == 'shoulder':
        if sev == "mild":
            return 'plan_shoulder_gentle'
        else:
            return 'plan_shoulder_strengthen'


    if loc == 'lower back':
        if sev == "mild":
            return 'plan_lower_back_gentle'
        else:
            return 'plan_lower_back_strengthen'


    if loc == 'hip':
        if sev == "mild":
            return 'plan_hip_gentle'
        else:
            return 'plan_hip_strengthen'


    if loc == 'leg':
        if sev == "mild":
            return 'plan_leg_gentle'
        else:
            return 'plan_leg_strengthen'


    if loc == 'arm':
        if sev == "mild":
            return 'plan_arm_gentle'
        else:
            return 'plan_arm_strengthen'


    return 'plan_rest_doctor'  # default fallback


# ============================
# 3. BACKEND: Helpers and Config
# ============================


engine = pyttsx3.init()
engine.setProperty('rate', 150)


SQUAT_DEPTH_TRIGGER = 85
COMPLETION_TRIGGER = 160
TARGET_DEPTH_LIMIT = 80
FAULT_COOLDOWN = 2
STRETCH_HOLD_TIME = 30  # seconds


EXERCISE_VIEWS = {
    "squat": "side",
    "deadlift": "side",
    "pushup": "side",
    "bicep curl": "front",
    "shoulder press": "front",
    "hamstring stretch": "side",
    "quad stretch": "side",
    "shoulder stretch": "front",
    "triceps stretch": "front",
    "hip flexor stretch": "side",
    "cat-cow stretch": "side",
}


stretch_instructions = {
    "hamstring stretch": "Stand tall, extend one leg forward, hinge at the hips and reach toward your toes.",
    "quad stretch": "Stand on one leg, grab the opposite ankle behind you, and gently pull toward your glutes.",
    "shoulder stretch": "Bring one arm across your chest and use the other to gently press it toward your body.",
    "triceps stretch": "Raise one arm overhead, bend the elbow, and use the other hand to press it downward.",
    "hip flexor stretch": "Step one foot forward into a lunge, push hips down and forward while keeping torso upright.",
    "cat-cow stretch": "On all fours, alternate between arching your back (cat) and dipping it (cow) with deep breaths."
}


workout_instructions = {
    "squat": "Stand with feet shoulder-width apart. Keep your back straight and squat down until knees bend about 90 degrees.",
    "deadlift": "Stand with feet shoulder-width apart. Hinge at hips keeping back straight and lift the weight.",
    "pushup": "Keep your body straight. Lower yourself until your elbows bend to about 90 degrees, then push back up.",
    "bicep curl": "Stand straight. Curl weights by bending elbows while keeping upper arms fixed.",
    "shoulder press": "With weights at shoulder level, press overhead while keeping torso stable."
}


mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def calculate_angle(a, b, c):
    radians = np.arctan2(c.y - b.y, c.x - b.x) - np.arctan2(a.y - b.y, a.x - b.x)
    angle = abs(np.degrees(radians))
    return angle if angle <= 180 else 360 - angle


def vertical_angle(p1, p2):
    return abs(p1.y - p2.y) * 100


def horizontal_alignment(p1, p2):
    return abs(p1.x - p2.x) * 100


def estimate_orientation(landmarks):
    ls = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    rs = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    z_diff = abs(ls.z - rs.z)
    return "front" if z_diff < 0.15 else "side"


def show_break_timer(duration):
    start_time = time.time()
    while time.time() - start_time < duration:
        remaining = duration - int(time.time() - start_time)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, f"Break Time: {remaining}s", (150, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        cv2.imshow("Break", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyWindow("Break")


def detect_faults(exercise, metrics, profile, phase):
    faults = []


    if exercise in ["squat", "deadlift"]:
        if metrics.get("back_angle", 180) < 120 and phase in ["down", "hold"]:
            faults.append(("Back rounding", "critical"))
        elif metrics.get("back_angle", 180) < 140:
            faults.append(("Back leaning", "moderate"))


        if metrics.get("knee_angle", 180) > 110 and phase == "down":
            faults.append(("Squat too shallow", "moderate"))


        if metrics.get("torso_angle", 0) > 60:
            faults.append(("Torso too forward", "critical"))


        if metrics.get("neck_angle", 0) > 45:
            faults.append(("Head not aligned", "moderate"))


    elif exercise == "pushup":
        if metrics.get("elbow_angle", 180) > 160:
            faults.append(("Elbows not bending", "moderate"))
        if metrics.get("back_angle", 180) < 140:
            faults.append(("Hips sagging", "critical"))


    elif exercise == "bicep curl":
        if metrics.get("elbow_angle", 180) > 160:
            faults.append(("Arm not curling enough", "moderate"))


    elif exercise == "shoulder press":
        if metrics.get("arm_verticality", 90) < 60:
            faults.append(("Arms not vertical", "moderate"))


    # Stretches
    elif exercise == "hamstring stretch":
        if metrics.get("leg_straightness", 180) < 160:
            faults.append(("Leg not straight", "moderate"))
        if metrics.get("torso_angle", 0) < 30:
            faults.append(("Torso not bent enough", "moderate"))


    elif exercise == "quad stretch":
        if metrics.get("heel_to_glute", 180) > 60:
            faults.append(("Heel too far from glute", "moderate"))


    elif exercise == "shoulder stretch":
        if metrics.get("arm_across_chest", 180) > 100:
            faults.append(("Arm not across chest enough", "mild"))
        if metrics.get("shoulder_alignment", 100) > 20:
            faults.append(("Arm not aligned horizontally", "mild"))


    elif exercise == "triceps stretch":
        if metrics.get("elbow_overhead", 100) < 20:
            faults.append(("Elbow not overhead", "moderate"))
        if metrics.get("arm_verticality", 180) > 100:
            faults.append(("Arm not vertical", "mild"))


    elif exercise == "hip flexor stretch":
        if metrics.get("front_leg_angle", 180) < 90:
            faults.append(("Front leg not bent enough", "moderate"))
        if metrics.get("torso_upright", 100) < 30:
            faults.append(("Torso leaning too far forward", "mild"))


    elif exercise == "cat-cow stretch":
        if metrics.get("spine_curve", 180) > 150:
            faults.append(("Spine not curved enough", "mild"))
        if metrics.get("shoulder_hip_alignment", 100) < 20:
            faults.append(("Shoulders and hips not aligned", "mild"))


    return faults


def extract_pose_metrics(exercise, landmarks):
    metrics = {}


    if exercise in ["squat", "deadlift"]:
        hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
        ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
        shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR]


        metrics["knee_angle"] = round(calculate_angle(hip, knee, ankle), 1)
        metrics["back_angle"] = round(calculate_angle(shoulder, hip, knee), 1)
        metrics["torso_angle"] = round(vertical_angle(shoulder, hip), 1)
        metrics["neck_angle"] = round(calculate_angle(ear, shoulder, hip), 1)


    elif exercise == "pushup":
        shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
        wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
        hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]


        metrics["elbow_angle"] = round(calculate_angle(shoulder, elbow, wrist), 1)
        metrics["back_angle"] = round(calculate_angle(shoulder, hip, landmarks[mp_pose.PoseLandmark.LEFT_KNEE]), 1)


    elif exercise == "bicep curl":
        shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
        wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]


        metrics["elbow_angle"] = round(calculate_angle(shoulder, elbow, wrist), 1)


    elif exercise == "shoulder press":
        elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
        shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]


        metrics["arm_verticality"] = round(vertical_angle(wrist, shoulder), 1)


    elif exercise == "hamstring stretch":
        hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
        ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
        shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        metrics["leg_straightness"] = round(calculate_angle(hip, knee, ankle), 1)
        metrics["torso_angle"] = round(vertical_angle(shoulder, hip), 1)


    elif exercise == "quad stretch":
        hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
        ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
        metrics["heel_to_glute"] = round(calculate_angle(knee, ankle, hip), 1)


    elif exercise == "shoulder stretch":
        shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
        wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
        metrics["arm_across_chest"] = round(calculate_angle(elbow, shoulder, wrist), 1)
        metrics["shoulder_alignment"] = round(horizontal_alignment(shoulder, wrist), 1)


    elif exercise == "triceps stretch":
        shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
        wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
        metrics["elbow_overhead"] = round(vertical_angle(wrist, elbow), 1)
        metrics["arm_verticality"] = round(calculate_angle(shoulder, elbow, wrist), 1)


    elif exercise == "hip flexor stretch":
        hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
        ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
        shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        metrics["front_leg_angle"] = round(calculate_angle(hip, knee, ankle), 1)
        metrics["torso_upright"] = round(vertical_angle(shoulder, hip), 1)


    elif exercise == "cat-cow stretch":
        shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
        metrics["spine_curve"] = round(calculate_angle(shoulder, hip, knee), 1)
        metrics["shoulder_hip_alignment"] = round(vertical_angle(shoulder, hip), 1)


    return metrics


def calculate_form_score(exercise, metrics):
    score = 100


    if exercise in ["squat", "deadlift"]:
        if metrics.get("back_angle", 180) < 140:
            score -= 30
        if metrics.get("knee_angle", 180) > 110:
            score -= 20
        if metrics.get("torso_angle", 0) > 60:
            score -= 30
        if metrics.get("neck_angle", 0) > 45:
            score -= 10


    elif exercise == "pushup":
        if metrics.get("elbow_angle", 180) > 160:
            score -= 20
        if metrics.get("back_angle", 180) < 140:
            score -= 30


    elif exercise == "bicep curl":
        if metrics.get("elbow_angle", 180) > 160:
            score -= 20


    elif exercise == "shoulder press":
        if metrics.get("arm_verticality", 90) < 60:
            score -= 20


    elif exercise == "hamstring stretch":
        if metrics.get("leg_straightness", 180) < 160:
            score -= 20
        if metrics.get("torso_angle", 0) < 30:
            score -= 20


    elif exercise == "quad stretch":
        if metrics.get("heel_to_glute", 180) > 60:
            score -= 20


    elif exercise == "shoulder stretch":
        if metrics.get("arm_across_chest", 180) > 100:
            score -= 15
        if metrics.get("shoulder_alignment", 100) > 20:
            score -= 10


    elif exercise == "triceps stretch":
        if metrics.get("elbow_overhead", 100) < 20:
            score -= 20
        if metrics.get("arm_verticality", 180) > 100:
            score -= 10


    elif exercise == "hip flexor stretch":
        if metrics.get("front_leg_angle", 180) < 90:
            score -= 20
        if metrics.get("torso_upright", 100) < 30:
            score -= 10


    elif exercise == "cat-cow stretch":
        if metrics.get("spine_curve", 180) > 150:
            score -= 15
        if metrics.get("shoulder_hip_alignment", 100) < 20:
            score -= 10


    return max(score, 0)


# Helper function for showing instructions popup in Tkinter
def show_instruction_popup(exercise):
    instructions = ""


    if exercise in workout_instructions:
        instructions = workout_instructions[exercise]
    elif exercise in stretch_instructions:
        instructions = stretch_instructions[exercise]
    else:
        instructions = "Follow safe form and perform the exercise carefully."


    # Create a popup window that blocks until the user clicks Start
    popup = tk.Tk()
    popup.title(f"Instructions: {exercise.capitalize()}")
    popup.geometry("450x200")
    popup.resizable(False, False)


    label = tk.Label(popup, text=f"Instructions for {exercise.capitalize()}:\n\n{instructions}", wraplength=430, justify='left')
    label.pack(padx=15, pady=15)


    button = tk.Button(popup, text="Start", command=popup.destroy)
    button.pack(pady=10)


    popup.mainloop()


# ============================
# 3. THE EXERCISE SESSION FUNCTION
# ============================


def run_exercise_session(exercise, sets, reps_per_set, break_time=60):
    DEPTH_TARGET = TARGET_DEPTH_LIMIT if "ACL" in "".join([]).upper() else SQUAT_DEPTH_TRIGGER
    set_count = 0


    # Show instructions popup before tracking starts
    show_instruction_popup(exercise)


    while set_count < sets:
        rep = 0
        rep_started = False
        frame_buffer = []
        last_fault_time = 0
        stretch_hold_start = None
        stretch_completed = False


        pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        cap = cv2.VideoCapture(0)


        for _ in range(10):  # warm-up
            cap.read()


        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break


            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            portrait = frame[:, int(w/4):int(3*w/4)]
            portrait = cv2.resize(portrait, (480, 640))


            rgb = cv2.cvtColor(portrait, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)


            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                orientation = estimate_orientation(landmarks)
                mp_drawing.draw_landmarks(portrait, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)


                expected_view = EXERCISE_VIEWS.get(exercise, "side")
                if orientation != expected_view:
                    cv2.putText(
                        portrait,
                        f"Face {expected_view}-on for {exercise}",
                        (30, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2
                    )
                    cv2.imshow("AI Trainer", portrait)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        return
                    continue  # wait for correct orientation


                metrics = extract_pose_metrics(exercise, landmarks)


                # Phase detection & rep/hold counting


                if exercise in ["squat", "deadlift"]:
                    knee_angle = metrics.get("knee_angle", 180)
                    phase = "down" if knee_angle < 100 else "up"
                    frame_buffer.append(knee_angle)
                    if len(frame_buffer) > 5:
                        smoothed_knee_angle = np.mean(frame_buffer[-5:])
                    else:
                        smoothed_knee_angle = knee_angle


                    if not rep_started and smoothed_knee_angle < DEPTH_TARGET:
                        rep_started = True
                    elif rep_started and smoothed_knee_angle > COMPLETION_TRIGGER:
                        rep += 1
                        rep_started = False


                elif exercise == "pushup":
                    elbow_angle = metrics.get("elbow_angle", 180)
                    phase = "down" if elbow_angle < 90 else "up"
                    if not rep_started and elbow_angle < 90:
                        rep_started = True
                    elif rep_started and elbow_angle > 160:
                        rep += 1
                        rep_started = False


                elif exercise == "bicep curl":
                    elbow_angle = metrics.get("elbow_angle", 180)
                    phase = "curl" if elbow_angle < 90 else "extend"
                    if not rep_started and elbow_angle < 90:
                        rep_started = True
                    elif rep_started and elbow_angle > 160:
                        rep += 1
                        rep_started = False


                elif exercise == "shoulder press":
                    arm_verticality = metrics.get("arm_verticality", 90)
                    phase = "press" if arm_verticality < 60 else "rest"
                    if not rep_started and arm_verticality < 60:
                        rep_started = True
                    elif rep_started and arm_verticality > 80:
                        rep += 1
                        rep_started = False


                # Stretch hold logic: Hold steady pose for 30s
                elif exercise in [
                    "hamstring stretch",
                    "quad stretch",
                    "shoulder stretch",
                    "triceps stretch",
                    "hip flexor stretch",
                    "cat-cow stretch"
                ]:
                    phase = "hold"
                    if not stretch_hold_start:
                        stretch_hold_start = time.time()
                        stretch_completed = False
                    else:
                        elapsed = time.time() - stretch_hold_start
                        cv2.putText(
                            portrait,
                            f"Holding: {int(elapsed)}s / {STRETCH_HOLD_TIME}s",
                            (30, 100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 255),
                            2
                        )
                        if elapsed >= STRETCH_HOLD_TIME and not stretch_completed:
                            stretch_completed = True
                            rep += 1  # count one hold as a rep for logging


                    # In stretches, once hold is done: end set after first hold
                    if stretch_completed and rep >= reps_per_set:
                        break


                else:
                    phase = "hold"


                # Live fault detection and form score
                live_faults = detect_faults(exercise, metrics, {}, phase)
                form_score = calculate_form_score(exercise, metrics)


                for i, (fault, severity) in enumerate(live_faults):
                    cv2.putText(
                        portrait,
                        fault,
                        (30, 210 + 20 * i),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255), 2
                    )


                cv2.putText(
                    portrait,
                    f"Form Score: {form_score}/100",
                    (30, 180),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0) if form_score > 70 else (0, 0, 255),
                    2
                )


            # On-screen progress info
            cv2.putText(
                portrait,
                f"{exercise.capitalize()} Set {set_count+1}/{sets} Rep/Hold: {rep}/{reps_per_set}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )
            cv2.imshow("AI Trainer", portrait)


            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return


            # Break loop to next set when reps or holds complete
            if rep >= reps_per_set:
                break


        cap.release()
        cv2.destroyAllWindows()
        set_count += 1
        if set_count < sets:
            show_break_timer(break_time)




# ============================
# 4. TKINTER APP with scrollable exercise list
# ============================


class RecoveryApp:
    def __init__(self, master):
        self.master = master
        master.title("AI Recovery Plan Assistant")
        master.geometry('600x700')
        master.resizable(True, True)


        self.fields = {
            "Pain Severity": ["mild", "moderate", "severe"],
            "Pain Duration (days)": [],
            "Pain Type": ["dull", "sharp", "radiating"],
            "Pain Location": ["lower back", "hip", "leg", "shoulder", "arm"],
            "Activity Level": ["active", "sedentary"],
            "Can Walk Comfortably?": ["yes", "no"],
            "Previous Injury History?": ["yes", "no"],
            "Symptoms Improving?": ["yes", "no"]
        }
        self.inputs = {}
        row = 0
        for label, options in self.fields.items():
            tk.Label(master, text=label).grid(row=row, column=0, sticky='w')


            if options:
                var = tk.StringVar(value=options[0])
                tk.OptionMenu(master, var, *options).grid(row=row, column=1, sticky="ew")
                self.inputs[label] = var
            else:
                entry = tk.Entry(master)
                entry.grid(row=row, column=1, sticky="ew")
                self.inputs[label] = entry
            row += 1


        tk.Button(master, text="Generate Plan", command=self.predict).grid(row=row, column=0, columnspan=2, pady=10, sticky="ew")


        self.output = scrolledtext.ScrolledText(master, width=70, height=20)
        self.output.grid(row=row + 1, column=0, columnspan=2, sticky="nsew")


        master.grid_rowconfigure(row + 1, weight=1)
        master.grid_columnconfigure(1, weight=1)


        # Scrollable exercise selection frame setup
        container = tk.Frame(master)
        container.grid(row=row + 2, column=0, columnspan=2, sticky='nsew')
        master.grid_rowconfigure(row + 2, weight=1)


        canvas = tk.Canvas(container)
        scrollbar = tk.Scrollbar(container, orient="vertical", command=canvas.yview)
        self.scrollable_frame = tk.Frame(canvas)


        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )


        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)


        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")


    def predict(self):
        try:
            severity = self.inputs["Pain Severity"].get()
            duration = int(self.inputs["Pain Duration (days)"].get())
            pain_type = self.inputs["Pain Type"].get()
            location = self.inputs["Pain Location"].get()
            activity = self.inputs["Activity Level"].get()
            can_walk = 1 if self.inputs["Can Walk Comfortably?"].get() == "yes" else 0
            history = 1 if self.inputs["Previous Injury History?"].get() == "yes" else 0
            improving = 1 if self.inputs["Symptoms Improving?"].get() == "yes" else 0

            plan_key = select_plan_key(severity, duration, pain_type, location, activity, can_walk, history, improving)
            plan_text = plans.get(plan_key, "No plan found.")

            self.output.delete("1.0", tk.END)
            self.output.insert(tk.END, plan_text)

            global generated_workout
            generated_workout = []

            for line in plan_text.splitlines():
                line = line.strip().lower()

                # Skip empty or very short lines
                if not line or len(line) < 5:
                    continue

                # Remove bullet points or dashes at line start
                line = re.sub(r"^[\-\u2022]\s*", "", line)

                # Try to parse sets x reps e.g. "3x10 squat"
                match = re.search(r"(\d+)[x×](\d+)\s+([a-z\s\-]+)", line)
                if match:
                    sets = int(match.group(1))
                    reps = int(match.group(2))
                    exercise_text = match.group(3).strip()
                else:
                    # Try to parse stretch-style "hold 30s x 3" or "hold 30s × 3"
                    match_two = re.search(r"hold\s*(\d+)\s*s?\s*[×x]\s*(\d+)", line)
                    if match_two:
                        reps = int(match_two.group(2))
                        sets = 1  # Usually one set, multiple holds
                        # Remove the hold pattern from line to get exercise name
                        exercise_text = re.sub(r"hold\s*\d+\s*s?\s*[×x]\s*\d+", "", line).strip()
                    else:
                        # Line does not match expected format; skip
                        continue

                # Find the known exercise key in the exercise text
                found_exercise = None
                for key in EXERCISE_VIEWS.keys():
                    if key in exercise_text:
                        found_exercise = key
                        break

                if found_exercise:
                    generated_workout.append((found_exercise, sets, reps))

            self.show_exercise_selection()

        except Exception as e:
            messagebox.showerror("Error", str(e))



    def show_exercise_selection(self):
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()


        tk.Label(self.scrollable_frame, text="Choose exercise to start:", font=('Arial', 12, 'bold')).pack(anchor="w", pady=(10, 5))


        for name, sets, reps in generated_workout:
            btn_text = f"{name.capitalize()} ({sets}x{reps})"
            btn = tk.Button(self.scrollable_frame, text=btn_text,
                            command=lambda n=name, s=sets, r=reps: run_exercise_session(n, s, r))
            btn.pack(fill="x", pady=2, padx=5)




# ========================
# 5. MAIN ENTRY POINT
# ========================


if __name__ == "__main__":
    root = tk.Tk()
    app = RecoveryApp(root)
    root.mainloop()



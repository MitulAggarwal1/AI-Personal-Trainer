import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import time

# === Voice setup ===
engine = pyttsx3.init()
engine.setProperty('rate', 150)

#def speak(text):
 #   engine.say(text)
  #  engine.runAndWait()

# === User Profile ===
user_profile = {
    "age": 16,
    "gender": "male",
    "height_cm": 180,
    "weight_kg": 70,
    "injuries": ["torn ACL"],
    "goals": ["rebuild strength", "protect knee", "improve mobility"]
}

# === Constants ===
SQUAT_DEPTH_TRIGGER = 85
COMPLETION_TRIGGER = 160
TARGET_DEPTH_LIMIT = 80
DEPTH_TARGET = TARGET_DEPTH_LIMIT if "ACL" in "".join(user_profile["injuries"]).upper() else SQUAT_DEPTH_TRIGGER
FAULT_COOLDOWN = 2


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
    "break": "none"
}

stretch_instructions = {
    "hamstring stretch": "Stand tall, extend one leg forward, hinge at the hips and reach toward your toes.",
    "quad stretch": "Stand on one leg, grab the opposite ankle behind you, and gently pull toward your glutes.",
    "shoulder stretch": "Bring one arm across your chest and use the other to gently press it toward your body.",
    "triceps stretch": "Raise one arm overhead, bend the elbow, and use the other hand to press it downward.",
    "hip flexor stretch": "Step one foot forward into a lunge, push hips down and forward while keeping torso upright.",
    "cat-cow stretch": "On all fours, alternate between arching your back (cat) and dipping it (cow) with deep breaths."
}


# === MediaPipe Setup ===
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# === Angle Calculators ===
def calculate_angle(a, b, c):
    radians = np.arctan2(c.y - b.y, c.x - b.x) - np.arctan2(a.y - b.y, a.x - b.x)
    angle = abs(np.degrees(radians))
    return angle if angle <= 180 else 360 - angle

def vertical_angle(a, b):
    radians = np.arctan2(b.y - a.y, b.x - a.x)
    angle = abs(np.degrees(radians))
    return angle

def vertical_angle(p1, p2):
    return abs(p1.y - p2.y) * 100  # crude vertical distance metric

def horizontal_alignment(p1, p2):
    return abs(p1.x - p2.x) * 100  # crude horizontal distance metric

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



# === Orientation Estimator ===
def estimate_orientation(landmarks):
    ls = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    rs = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    z_diff = abs(ls.z - rs.z)
    return "front" if z_diff < 0.15 else "side"

# === Pose Metrics Extraction ===
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
        torso = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        metrics["leg_straightness"] = round(calculate_angle(hip, knee, ankle), 1)
        metrics["torso_angle"] = round(vertical_angle(torso, hip), 1)

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

# === Fault Detection ===
def detect_faults(metrics, profile, phase):
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

    elif exercise == "hamstring stretch":
        if metrics.get("leg_straightness", 180) < 160:
            faults.append(("Leg not straight", "moderate"))
        if metrics.get("torso_angle", 0) < 30:
            faults.append(("Torso not bent enough", "moderate"))

    elif exercise == "quad stretch":
        if metrics.get("heel_to_glute", 180) > 60:
            faults.append(("Heel too far from glute", "moderate"))
        # Optional: check for balance or torso lean

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
    
    elif exercise == "break":
        break_duration = 60 if previous_exercise in ["squat", "push ups"] else 30
        show_break_timer(break_duration)
        return  # Skip pose tracking



    return faults

# === Form Score Calculator ===
def calculate_form_score(metrics):
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

# === Rep Tracking State ===
rep = 0
rep_started = False
frame_buffer = []
rep_fault_log = []
last_fault_time = 0
phase_state = "hold"

# === Input Exercise ===
exercise = input("Enter your exercise (e.g. squat, deadlift, pushup, bicep curl, shoulder press, break, hamstring stretch): ").strip().lower()
expected_view = EXERCISE_VIEWS.get(exercise, "side")
print(f"\nPlease stand {expected_view}-on to the camera for {exercise.upper()}.\n")

# === Warmup Frames to Avoid Lag ===
cap = cv2.VideoCapture(0)
for _ in range(10): cap.read()

stretch_hold_start = None
stretch_completed = False


while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    portrait = frame[:, int(w / 4):int(3 * w / 4)]
    portrait = cv2.resize(portrait, (480, 640))
    rgb = cv2.cvtColor(portrait, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        orientation = estimate_orientation(landmarks)
        mp_drawing.draw_landmarks(portrait, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        if orientation != expected_view:
            cv2.putText(portrait, f"Face {expected_view}-on for {exercise}", (30, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            metrics = extract_pose_metrics(exercise, landmarks)

            # Phase detection
            if exercise in ["squat", "deadlift"]:
                knee_angle = metrics.get("knee_angle", 180)
                phase = "down" if knee_angle < 100 else "up"

                frame_buffer.append(knee_angle)
                if len(frame_buffer) > 5:
                    smoothed_knee_angle = np.mean(frame_buffer[-5:])
                else:
                    smoothed_knee_angle = knee_angle

                if not rep_started and smoothed_knee_angle < DEPTH_TARGET:
                    #speak("Depth reached. Hold or push up!")
                    cv2.putText(portrait, "Depth Reached", (30, 250),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            elif exercise == "pushup":
                elbow_angle = metrics.get("elbow_angle", 180)
                phase = "down" if elbow_angle < 90 else "up"

            elif exercise == "bicep curl":
                elbow_angle = metrics.get("elbow_angle", 180)
                phase = "curl" if elbow_angle < 90 else "extend"

            elif exercise == "shoulder press":
                arm_verticality = metrics.get("arm_verticality", 90)
                phase = "press" if arm_verticality < 60 else "rest"

            elif exercise == "hamstring stretch":
                torso_angle = metrics.get("torso_angle", 90)
                phase = "hold" if torso_angle < 70 else "setup"

            elif exercise == "quad stretch":
                knee_alignment = metrics.get("knee_alignment", 0)
                phase = "hold" if abs(knee_alignment) < 20 else "setup"

            elif exercise == "shoulder stretch":
                arm_cross_angle = metrics.get("arm_cross_angle", 90)
                phase = "hold" if arm_cross_angle < 40 else "setup"

            elif exercise == "triceps stretch":
                elbow_lift = metrics.get("elbow_lift", 0)
                phase = "hold" if elbow_lift > 60 else "setup"

            elif exercise == "hip flexor stretch":
                hip_extension = metrics.get("hip_extension", 0)
                phase = "hold" if hip_extension > 30 else "setup"

            elif exercise == "cat-cow stretch":
                spine_curve = metrics.get("spine_curve", 0)
                phase = "cat" if spine_curve < -20 else "cow" if spine_curve > 20 else "neutral"


            # Fault detection
            live_faults = detect_faults(metrics, user_profile, phase)
            if time.time() - last_fault_time > FAULT_COOLDOWN:
                for fault, severity in live_faults:
                    #speak(fault)
                    last_fault_time = time.time()

            for i, (fault, _) in enumerate(live_faults):
                cv2.putText(portrait, f"{fault}", (30, 220 + 20 * i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Rep tracking
            if exercise in ["squat", "deadlift"]:
                if not rep_started and smoothed_knee_angle < SQUAT_DEPTH_TRIGGER:
                    rep_started = True
                    phase_state = "down"

                elif rep_started and smoothed_knee_angle > COMPLETION_TRIGGER:
                    rep += 1
                    rep_started = False
                    phase_state = "up"
                    form_score = calculate_form_score(metrics)
                    rep_fault_log.append({"rep": rep, "faults": live_faults, "score": form_score})
                    summary = f"Rep {rep} complete. Form Score: {form_score}"
                    if live_faults:
                        summary += " | Issues: " + ", ".join(f[0] for f in live_faults)
                    #else:
                        #speak("Great form! Keep it up!")
                    #speak(summary)
                    cv2.putText(portrait, summary, (30, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            elif exercise == "pushup":
                if not rep_started and elbow_angle < 90:
                    rep_started = True
                elif rep_started and elbow_angle > 160:
                    rep += 1
                    rep_started = False
                    form_score = calculate_form_score(metrics)
                    rep_fault_log.append({"rep": rep, "faults": live_faults, "score": form_score})
                    summary = f"Rep {rep} complete. Form Score: {form_score}"
                    if live_faults:
                        summary += " | Issues: " + ", ".join(f[0] for f in live_faults)
                    #else:
                        #speak("Great form! Keep it up!")
                    #speak(summary)
                    cv2.putText(portrait, summary, (30, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            elif exercise == "bicep curl":
                if not rep_started and elbow_angle < 90:
                    rep_started = True
                elif rep_started and elbow_angle > 160:
                    rep += 1
                    rep_started = False
                    form_score = calculate_form_score(metrics)
                    rep_fault_log.append({"rep": rep, "faults": live_faults, "score": form_score})
                    summary = f"Rep {rep} complete. Form Score: {form_score}"
                    if live_faults:
                        summary += " | Issues: " + ", ".join(f[0] for f in live_faults)
                    #else:
                        #speak("Great form! Keep it up!")
                    #speak(summary)
                    cv2.putText(portrait, summary, (30, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            elif exercise == "shoulder press":
                if not rep_started and arm_verticality < 60:
                    rep_started = True
                elif rep_started and arm_verticality > 80:
                    rep += 1
                    rep_started = False
                    form_score = calculate_form_score(metrics)
                    rep_fault_log.append({"rep": rep, "faults": live_faults, "score": form_score})
                    summary = f"Rep {rep} complete. Form Score: {form_score}"
                    if live_faults:
                        summary += " | Issues: " + ", ".join(f[0] for f in live_faults)
                    #else:
                        #speak("Great form! Keep it up!")
                    #speak(summary)
                    cv2.putText(portrait, summary, (30, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            elif exercise == "hamstring stretch":
                if phase == "hold":
                    if stretch_hold_start is None:
                        stretch_hold_start = time.time()
                    elif time.time() - stretch_hold_start >= 30 and not stretch_completed:
                        stretch_completed = True
                        form_score = calculate_form_score(metrics)
                        rep_fault_log.append({"stretch": exercise, "faults": live_faults, "score": form_score})
                        summary = f"Hamstring Stretch complete. Form Score: {form_score}"
                        if live_faults:
                            summary += " | Issues: " + ", ".join(f[0] for f in live_faults)
                        cv2.putText(portrait, summary, (30, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                        instruction = stretch_instructions.get(exercise, "")
                        cv2.putText(portrait, instruction, (30, 160),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                else:
                    stretch_hold_start = None
                    stretch_completed = False

            elif exercise == "quad stretch":
                if phase == "hold":
                    if stretch_hold_start is None:
                        stretch_hold_start = time.time()
                    elif time.time() - stretch_hold_start >= 30 and not stretch_completed:
                        stretch_completed = True
                        form_score = calculate_form_score(metrics)
                        rep_fault_log.append({"stretch": exercise, "faults": live_faults, "score": form_score})
                        summary = f"Quad Stretch complete. Form Score: {form_score}"
                        if live_faults:
                            summary += " | Issues: " + ", ".join(f[0] for f in live_faults)
                        cv2.putText(portrait, summary, (30, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                        instruction = stretch_instructions.get(exercise, "")
                        cv2.putText(portrait, instruction, (30, 160),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)                        
                else:
                    stretch_hold_start = None
                    stretch_completed = False

            elif exercise == "shoulder stretch":
                if phase == "hold":
                    if stretch_hold_start is None:
                        stretch_hold_start = time.time()
                    elif time.time() - stretch_hold_start >= 30 and not stretch_completed:
                        stretch_completed = True
                        form_score = calculate_form_score(metrics)
                        rep_fault_log.append({"stretch": exercise, "faults": live_faults, "score": form_score})
                        summary = f"Shoulder Stretch complete. Form Score: {form_score}"
                        if live_faults:
                            summary += " | Issues: " + ", ".join(f[0] for f in live_faults)
                        cv2.putText(portrait, summary, (30, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                        instruction = stretch_instructions.get(exercise, "")
                        cv2.putText(portrait, instruction, (30, 160),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)                        
                else:
                    stretch_hold_start = None
                    stretch_completed = False

            elif exercise == "triceps stretch":
                if phase == "hold":
                    if stretch_hold_start is None:
                        stretch_hold_start = time.time()
                    elif time.time() - stretch_hold_start >= 30 and not stretch_completed:
                        stretch_completed = True
                        form_score = calculate_form_score(metrics)
                        rep_fault_log.append({"stretch": exercise, "faults": live_faults, "score": form_score})
                        summary = f"Triceps Stretch complete. Form Score: {form_score}"
                        if live_faults:
                            summary += " | Issues: " + ", ".join(f[0] for f in live_faults)
                        cv2.putText(portrait, summary, (30, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                        instruction = stretch_instructions.get(exercise, "")
                        cv2.putText(portrait, instruction, (30, 160),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)                        
                else:
                    stretch_hold_start = None
                    stretch_completed = False

            elif exercise == "hip flexor stretch":
                if phase == "hold":
                    if stretch_hold_start is None:
                        stretch_hold_start = time.time()
                    elif time.time() - stretch_hold_start >= 30 and not stretch_completed:
                        stretch_completed = True
                        form_score = calculate_form_score(metrics)
                        rep_fault_log.append({"stretch": exercise, "faults": live_faults, "score": form_score})
                        summary = f"Hip Flexor Stretch complete. Form Score: {form_score}"
                        if live_faults:
                            summary += " | Issues: " + ", ".join(f[0] for f in live_faults)
                        cv2.putText(portrait, summary, (30, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                        instruction = stretch_instructions.get(exercise, "")
                        cv2.putText(portrait, instruction, (30, 160),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)                        
                else:
                    stretch_hold_start = None
                    stretch_completed = False

            elif exercise == "cat-cow stretch":
                if phase in ["cat", "cow"]:
                    if stretch_hold_start is None:
                        stretch_hold_start = time.time()
                    elif time.time() - stretch_hold_start >= 30 and not stretch_completed:
                        stretch_completed = True
                        form_score = calculate_form_score(metrics)
                        rep_fault_log.append({"stretch": exercise, "faults": live_faults, "score": form_score})
                        summary = f"Cat-Cow Stretch complete. Form Score: {form_score}"
                        if live_faults:
                            summary += " | Issues: " + ", ".join(f[0] for f in live_faults)
                        cv2.putText(portrait, summary, (30, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                        instruction = stretch_instructions.get(exercise, "")
                        cv2.putText(portrait, instruction, (30, 160),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)                        
                else:
                    stretch_hold_start = None
                    stretch_completed = False

   

            # Live metrics display
            form_score = calculate_form_score(metrics)
            cv2.putText(portrait, f"Rep: {rep}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            if "knee_angle" in metrics:
                cv2.putText(portrait, f"Knee: {metrics['knee_angle']} deg", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if "back_angle" in metrics:
                cv2.putText(portrait, f"Back: {metrics['back_angle']} deg", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            cv2.putText(portrait, f"Phase: {phase}", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 100), 2)
            cv2.putText(portrait, f"View: {orientation}", (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 2)
            cv2.putText(portrait, f"Form Score: {form_score}/100", (30, 190),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if form_score > 70 else (0, 0, 255), 2)

    else:
        cv2.putText(portrait, "No Pose Detected", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("AI Trainer - Verbal Coach", portrait)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    

cap.release()
cv2.destroyAllWindows()


# === Post-Session Summary ===
print("\n=== SESSION SUMMARY ===")
for log in rep_fault_log:
    print(f"Rep {log['rep']} - Score: {log['score']} - Faults: {[f[0] for f in log['faults']]}")

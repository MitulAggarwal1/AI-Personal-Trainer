import tkinter as tk
from tkinter import messagebox, scrolledtext
import numpy as np
import cv2
import mediapipe as mp
import time
import re
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# ============================
# 1. HARD-CODED PLANS DICTIONARY
# ============================

plans = {
    'plan_rest_doctor': (
        "🛑 Rest & See Medical Professional\n"
        "- Avoid strenuous activities\n"
        "- Use ice, monitor symptoms\n"
        "- Seek immediate consultation\n"
    ),
    'plan_lower_back_gentle': (
        "🧘 Lower Back Gentle Recovery\n"
        "- cat-cow stretch 3x10\n"
        "- pelvic tilts 3x10\n"
        "- childs pose 3x30s\n"
        "- knee-to-chest 3x30s\n"
        "- hamstring stretch 3x30s\n"
        "- pushup 3x5\n"
        "- deadlift 2x8\n"
        "- goblet squat 2x10\n"
    ),
    'plan_lower_back_strengthen': (
        "💪 Lower Back Strengthening\n"
        "- deadlift 3x8\n"
        "- bird-dog 3x10\n"
        "- plank 3x20s\n"
    ),
    'plan_hip_gentle': (
        "🧘 Hip Mobility & Gentle Stretch\n"
        "- hip flexor stretch 3x30s\n"
        "- figure-4 stretch 3x30s\n"
        "- deadlift 2x10\n"
        "- glute bridge 2x12\n"
        "- step-ups 2x12\n"
    ),
    'plan_hip_strengthen': (
        "💪 Hip Strengthening Plan\n"
        "- glute bridge 3x12\n"
        "- step-ups 3x10\n"
        "- squat 3x10\n"
    ),
    'plan_leg_gentle': (
        "🧘 Gentle Leg Recovery\n"
        "- hamstring stretch 3x30s\n"
        "- quad stretch 3x30s\n"
        "- calf raises 2x15\n"
        "- squat 3x10\n"
        "- lunge 2x10\n"
    ),
    'plan_leg_strengthen': (
        "💪 Leg Strengthening\n"
        "- squat 3x10\n"
        "- lunge 3x10\n"
        "- calf raises 3x15\n"
    ),
    'plan_shoulder_gentle': (
        "🧘 Shoulder Stretch & Mobility\n"
        "- shoulder stretch 3x30s\n"
        "- wall slides 2x10\n"
        "- incline pushup 3x8\n"
        "- shoulder press 2x10\n"
    ),
    'plan_shoulder_strengthen': (
        "💪 Shoulder Strengthening\n"
        "- pushup 3x10\n"
        "- bicep curl 3x12\n"
        "- shoulder press 3x10\n"
        "- dumbbell row 3x10\n"
    ),
    'plan_arm_gentle': (
        "🧘 Arm Recovery & Mobility\n"
        "- wrist circles 3x15\n"
        "- arm raises 3x10\n"
        "- bicep curl 3x8\n"
        "- shoulder press 3x8\n"
    ),
    'plan_arm_strengthen': (
        "💪 Arm Strengthening\n"
        "- bicep curl 3x12\n"
        "- shoulder press 3x10\n"
        "- pushup 3x10\n"
    ),
    # Stretches
    'plan_hamstring_stretch': "- hamstring stretch 3x30s\n",
    'plan_quad_stretch': "- quad stretch 3x30s\n",
    'plan_shoulder_stretch': "- shoulder stretch 3x30s\n",
    'plan_triceps_stretch': "- triceps stretch 3x30s\n",
    'plan_hip_flexor_stretch': "- hip flexor stretch 3x30s\n",
    'plan_cat_cow_stretch': "- cat-cow stretch 3x10\n",
}

# ============================
# 2. PLAN SELECTOR SETUP (Pretrained once)
# ============================

def build_plan_selector():
    data = [
        ['mild', 2, 'dull', 'leg', 'active', 1, 0, 1, 'plan_leg_gentle'],
        ['mild', 15, 'dull', 'leg', 'sedentary', 1, 1, 0, 'plan_leg_gentle'],
        ['mild', 5, 'sharp', 'leg', 'active', 0, 0, 1, 'plan_rest_doctor'],
        ['moderate', 3, 'radiating', 'leg', 'active', 1, 0, 1, 'plan_leg_gentle'],
        ['moderate', 10, 'dull', 'leg', 'sedentary', 1, 0, 1, 'plan_leg_strengthen'],
        ['moderate', 7, 'sharp', 'leg', 'active', 0, 0, 0, 'plan_rest_doctor'],
        ['severe', 1, 'sharp', 'leg', 'active', 0, 0, 0, 'plan_rest_doctor'],
        ['severe', 12, 'radiating', 'leg', 'sedentary', 0, 1, 0, 'plan_rest_doctor'],
        ['mild', 5, 'dull', 'hip', 'active', 1, 0, 1, 'plan_hip_gentle'],
        ['mild', 20, 'dull', 'hip', 'sedentary', 1, 1, 0, 'plan_hip_gentle'],
        ['moderate', 2, 'radiating', 'hip', 'active', 1, 0, 1, 'plan_hip_gentle'],
        ['moderate', 15, 'dull', 'hip', 'active', 1, 0, 1, 'plan_hip_strengthen'],
        ['moderate', 7, 'sharp', 'hip', 'active', 0, 0, 0, 'plan_rest_doctor'],
        ['severe', 1, 'sharp', 'hip', 'sedentary', 0, 1, 0, 'plan_rest_doctor'],
        ['severe', 14, 'dull', 'hip', 'active', 1, 0, 0, 'plan_rest_doctor'],
        ['mild', 3, 'dull', 'lower back', 'active', 1, 0, 1, 'plan_lower_back_gentle'],
        ['mild', 12, 'sharp', 'lower back', 'sedentary', 0, 0, 1, 'plan_rest_doctor'],
        ['moderate', 5, 'radiating', 'lower back', 'active', 1, 0, 1, 'plan_lower_back_gentle'],
        ['moderate', 15, 'dull', 'lower back', 'sedentary', 1, 0, 1, 'plan_lower_back_strengthen'],
        ['moderate', 10, 'sharp', 'lower back', 'active', 0, 0, 0, 'plan_rest_doctor'],
        ['severe', 2, 'sharp', 'lower back', 'active', 0, 0, 0, 'plan_rest_doctor'],
        ['severe', 20, 'radiating', 'lower back', 'sedentary', 0, 1, 0, 'plan_rest_doctor'],
        ['mild', 4, 'dull', 'shoulder', 'active', 1, 0, 1, 'plan_shoulder_gentle'],
        ['mild', 18, 'radiating', 'shoulder', 'sedentary', 1, 1, 0, 'plan_shoulder_gentle'],
        ['moderate', 3, 'dull', 'shoulder', 'active', 1, 0, 1, 'plan_shoulder_strengthen'],
        ['moderate', 8, 'sharp', 'shoulder', 'active', 0, 0, 0, 'plan_rest_doctor'],
        ['moderate', 12, 'radiating', 'shoulder', 'sedentary', 1, 1, 0, 'plan_shoulder_gentle'],
        ['severe', 1, 'sharp', 'shoulder', 'active', 0, 0, 0, 'plan_rest_doctor'],
        ['severe', 20, 'dull', 'shoulder', 'active', 1, 0, 0, 'plan_rest_doctor'],
        ['mild', 2, 'dull', 'arm', 'active', 1, 0, 1, 'plan_arm_gentle'],
        ['mild', 10, 'sharp', 'arm', 'sedentary', 0, 0, 1, 'plan_rest_doctor'],
        ['moderate', 5, 'dull', 'arm', 'active', 1, 0, 1, 'plan_arm_strengthen'],
        ['moderate', 8, 'sharp', 'arm', 'active', 0, 0, 0, 'plan_rest_doctor'],
        ['moderate', 14, 'radiating', 'arm', 'sedentary', 1, 1, 0, 'plan_arm_gentle'],
        ['severe', 3, 'sharp', 'arm', 'active', 0, 0, 0, 'plan_rest_doctor'],
        ['severe', 20, 'dull', 'arm', 'sedentary', 0, 1, 0, 'plan_rest_doctor'],
    ]
    X_raw = [row[:-1] for row in data]
    y = [row[-1] for row in data]
    cat_idxs = [0, 2, 3, 4]
    encoders = [LabelEncoder() for _ in cat_idxs]

    for enc, idx in zip(encoders, cat_idxs):
        enc.fit([r[idx] for r in X_raw])

    def encode_rows(rows):
        X = []
        for r in rows:
            encoded = list(r)
            for enc, idx in zip(encoders, cat_idxs):
                encoded[idx] = enc.transform([encoded[idx]])[0]
            X.append(encoded)
        return np.array(X)

    X = encode_rows(X_raw)
    clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    clf.fit(X, y)

    def predict(severity, duration, pain_type, location, activity, can_walk, history, improving):
        query = [
            severity.lower(), duration, pain_type.lower(), location.lower(),
            activity.lower(), int(can_walk), int(history), int(improving)
        ]
        q_enc = encode_rows([query])
        return clf.predict(q_enc)[0]

    return predict

select_plan_key = build_plan_selector()

# ============================
# 3. BACKEND CONFIG 
# ============================

SQUAT_DEPTH_TRIGGER = 85
COMPLETION_TRIGGER = 160
TARGET_DEPTH_LIMIT = 80
STRETCH_HOLD_TIME = 30

# ============================
# 4. EXERCISE CONFIGURATION
# ============================

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
    "glute bridge": "side",
    "step-ups": "side",
    "lunge": "side",
    "plank": "front",
    "bird-dog": "side",
    "wrist circles": "front",
    "arm raises": "front",
    "wall slides": "front",
    "incline pushup": "side",
    "dumbbell row": "front",
    "goblet squat": "front",
    "childs pose": "side",
    "figure-4 stretch": "side"
}

stretch_instructions = {
    "hamstring stretch": "Stand tall, extend one leg forward, hinge at the hips and reach toward your toes.",
    "quad stretch": "Stand on one leg, grab the opposite ankle behind you, and gently pull toward your glutes.",
    "shoulder stretch": "Bring one arm across your chest and use the other to gently press it toward your body.",
    "triceps stretch": "Raise one arm overhead, bend the elbow and use the other hand to press it downward.",
    "hip flexor stretch": "Step one foot forward into a lunge, push hips down and forward while keeping torso upright.",
    "cat-cow stretch": "On all fours, alternate between arching your back (cat) and dipping it (cow) with deep breaths.",
    "childs pose": "Kneel on floor, sit back onto heels, and reach arms forward on the floor.",
    "figure-4 stretch": "Lie on your back, cross one ankle over the opposite knee, and pull leg toward your chest."
}

workout_instructions = {
    "squat": "Stand with feet shoulder-width apart. Keep back straight and squat until hips are level with knees.",
    "deadlift": "Stand with feet shoulder-width apart. Hinge at hips keeping back flat, lift smoothly.",
    "goblet squat": "Hold a dumbbell at your chest and squat, keeping torso upright and knees out.",
    "pushup": "Body straight, lower until elbows at 90°, push back up.",
    "incline pushup": "Hands on bench, body straight, lower to 90° elbows, push back.",
    "bicep curl": "Stand tall, curl weights by bending elbows, keep upper arms still.",
    "shoulder press": "Press weights overhead from shoulder height, avoid leaning back.",
    "dumbbell row": "Flat back, pull dumbbell toward torso keeping elbow close.",
    "lunge": "Step forward, back knee near floor, front knee over ankle.",
    "step-ups": "Step onto platform, drive through whole foot, step down slowly.",
    "glute bridge": "Lie on back, feet flat, lift hips until body forms a straight line.",
    "plank": "Elbows under shoulders, body straight, hold core tight.",
    "bird-dog": "Hands under shoulders, extend opposite arm & leg, keep back stable.",
    "wrist circles": "Extend arms, slowly rotate wrists in both directions.",
    "arm raises": "Raise straight arms to shoulder height, palms down.",
    "wall slides": "Back to wall, slide arms overhead, keep elbows and wrists touching the wall."
}

# ============================
# 5. POSE ANGLE HELPERS
# ============================

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

# ============================
# 6. METRIC EXTRACTION
# ============================

def extract_pose_metrics(exercise, lm):
    m = {}
    L = mp_pose.PoseLandmark

    if exercise in ["squat", "deadlift", "goblet squat"]:
        hip, knee, ankle = lm[L.LEFT_HIP], lm[L.LEFT_KNEE], lm[L.LEFT_ANKLE]
        shoulder, ear = lm[L.LEFT_SHOULDER], lm[L.LEFT_EAR]
        m["knee_angle"] = round(calculate_angle(hip, knee, ankle), 1)
        m["back_angle"] = round(calculate_angle(shoulder, hip, knee), 1)
        m["torso_angle"] = round(vertical_angle(shoulder, hip), 1)
        m["neck_angle"] = round(calculate_angle(ear, shoulder, hip), 1)

    elif exercise in ["pushup", "incline pushup"]:
        shoulder, elbow, wrist = lm[L.LEFT_SHOULDER], lm[L.LEFT_ELBOW], lm[L.LEFT_WRIST]
        hip = lm[L.LEFT_HIP]
        m["elbow_angle"] = round(calculate_angle(shoulder, elbow, wrist), 1)
        m["back_angle"] = round(calculate_angle(shoulder, hip, lm[L.LEFT_KNEE]), 1)

    elif exercise == "bicep curl":
        shoulder, elbow, wrist = lm[L.LEFT_SHOULDER], lm[L.LEFT_ELBOW], lm[L.LEFT_WRIST]
        m["elbow_angle"] = round(calculate_angle(shoulder, elbow, wrist), 1)

    elif exercise == "shoulder press":
        wrist, shoulder = lm[L.LEFT_WRIST], lm[L.LEFT_SHOULDER]
        m["arm_verticality"] = round(vertical_angle(wrist, shoulder), 1)

    elif exercise == "dumbbell row":
        wrist, elbow, shoulder = lm[L.LEFT_WRIST], lm[L.LEFT_ELBOW], lm[L.LEFT_SHOULDER]
        m["row_angle"] = round(calculate_angle(wrist, elbow, shoulder), 1)
        m["back_angle"] = round(calculate_angle(shoulder, lm[L.LEFT_HIP], lm[L.LEFT_KNEE]), 1)

    elif exercise in ["lunge", "step-ups"]:
        hip, knee, ankle = lm[L.LEFT_HIP], lm[L.LEFT_KNEE], lm[L.LEFT_ANKLE]
        m["front_knee_angle"] = round(calculate_angle(hip, knee, ankle), 1)

    elif exercise == "glute bridge":
        hip, knee, ankle = lm[L.LEFT_HIP], lm[L.LEFT_KNEE], lm[L.LEFT_ANKLE]
        shoulder = lm[L.LEFT_SHOULDER]
        m["hip_extension"] = round(calculate_angle(shoulder, hip, knee), 1)

    elif exercise == "plank":
        shoulder, hip, ankle = lm[L.LEFT_SHOULDER], lm[L.LEFT_HIP], lm[L.LEFT_ANKLE]
        m["body_line"] = round(calculate_angle(shoulder, hip, ankle), 1)

    elif exercise == "bird-dog":
        shoulder, hip, knee = lm[L.LEFT_SHOULDER], lm[L.LEFT_HIP], lm[L.LEFT_KNEE]
        m["spine_alignment"] = round(calculate_angle(shoulder, hip, knee), 1)

    elif exercise == "hamstring stretch":
        hip, knee, ankle = lm[L.LEFT_HIP], lm[L.LEFT_KNEE], lm[L.LEFT_ANKLE]
        shoulder = lm[L.LEFT_SHOULDER]
        m["leg_straightness"] = round(calculate_angle(hip, knee, ankle), 1)
        m["torso_angle"] = round(vertical_angle(shoulder, hip), 1)

    elif exercise == "quad stretch":
        hip, knee, ankle = lm[L.LEFT_HIP], lm[L.LEFT_KNEE], lm[L.LEFT_ANKLE]
        m["heel_to_glute"] = round(calculate_angle(knee, ankle, hip), 1)

    elif exercise == "shoulder stretch":
        shoulder, elbow, wrist = lm[L.LEFT_SHOULDER], lm[L.LEFT_ELBOW], lm[L.LEFT_WRIST]
        m["arm_across_chest"] = round(calculate_angle(elbow, shoulder, wrist), 1)
        m["shoulder_alignment"] = round(horizontal_alignment(shoulder, wrist), 1)

    elif exercise == "triceps stretch":
        shoulder, elbow, wrist = lm[L.LEFT_SHOULDER], lm[L.LEFT_ELBOW], lm[L.LEFT_WRIST]
        m["elbow_overhead"] = round(vertical_angle(wrist, elbow), 1)
        m["arm_verticality"] = round(calculate_angle(shoulder, elbow, wrist), 1)

    elif exercise == "hip flexor stretch":
        hip, knee, ankle = lm[L.LEFT_HIP], lm[L.LEFT_KNEE], lm[L.LEFT_ANKLE]
        shoulder = lm[L.LEFT_SHOULDER]
        m["front_leg_angle"] = round(calculate_angle(hip, knee, ankle), 1)
        m["torso_upright"] = round(vertical_angle(shoulder, hip), 1)

    elif exercise == "cat-cow stretch":
        shoulder, hip, knee = lm[L.LEFT_SHOULDER], lm[L.LEFT_HIP], lm[L.LEFT_KNEE]
        m["spine_curve"] = round(calculate_angle(shoulder, hip, knee), 1)
        m["shoulder_hip_alignment"] = round(vertical_angle(shoulder, hip), 1)

    elif exercise == "childs pose":
        hip, shoulder = lm[L.LEFT_HIP], lm[L.LEFT_SHOULDER]
        m["torso_lowering"] = round(vertical_angle(shoulder, hip), 1)

    elif exercise == "figure-4 stretch":
        hip, knee, ankle = lm[L.LEFT_HIP], lm[L.LEFT_KNEE], lm[L.LEFT_ANKLE]
        m["hip_opening"] = round(calculate_angle(hip, knee, ankle), 1)

    return m

# ============================
# 7. FAULT DETECTION (Declarative)
# ============================

fault_rules = {
    "squat": [
        ("back_angle", "<", 35, "Back rounding - keep spine neutral", "critical"),
        ("back_angle", "<", 60, "Slight back lean - engage core", "moderate"),
        ("knee_angle", ">", 110, "Squat deeper - aim for thighs parallel", "moderate", "phase", "down"),
        ("torso_angle", ">", 55, "Torso leaning forward", "moderate")
    ],
    "deadlift": [
        ("back_angle", "<", 35, "Back rounding - hinge at hips", "critical"),
        ("back_angle", "<", 60, "Slight back lean", "moderate"),
        ("torso_angle", ">", 75, "Torso too low - bend knees", "moderate"),
    ],
    "pushup": [
        ("elbow_angle", ">", 160, "Bend elbows more during descent", "moderate"),
        ("back_angle", "<", 50, "Hips sagging - keep core tight", "critical")
    ],
    "bicep curl": [
        ("elbow_angle", ">", 165, "Not curling enough - full range", "moderate")
    ],
    "shoulder press": [
        ("arm_verticality", "<", 55, "Press arms fully overhead", "moderate")
    ],
    "dumbbell row": [
        ("row_angle", "<", 55, "Pull elbow higher - close to torso", "moderate"),
        ("back_angle", "<", 35, "Back rounding - keep flat", "critical")
    ],
    "lunge": [
        ("front_knee_angle", ">", 115, "Step deeper - bend front knee more", "moderate")
    ],
    "step-ups": [
        ("front_knee_angle", ">", 115, "Step deeper - bend front knee more", "moderate")
    ],
    "glute bridge": [
        ("hip_extension", "<", 55, "Lift hips higher", "moderate")
    ],
    "plank": [
        ("body_line", "<", 55, "Sagging hips - engage core", "critical")
    ],
    "bird-dog": [
        ("spine_alignment", "<", 55, "Back not stable - avoid sag", "moderate")
    ],
    "hamstring stretch": [
        ("leg_straightness", "<", 165, "Straighten knee more", "moderate"),
        ("torso_angle", "<", 25, "Hinge further at hips", "moderate")
    ],
    "quad stretch": [
        ("heel_to_glute", ">", 75, "Pull ankle closer to glute", "moderate")
    ],
    "shoulder stretch": [
        ("arm_across_chest", ">", 100, "Bring arm further across chest", "mild")
    ],
    "triceps stretch": [
        ("elbow_overhead", "<", 20, "Raise elbow higher", "moderate")
    ],
    "hip flexor stretch": [
        ("front_leg_angle", "<", 90, "Bend front knee more", "moderate"),
        ("torso_upright", "<", 35, "Keep torso upright", "mild")
    ],
    "cat-cow stretch": [
        ("spine_curve", ">", 165, "Round spine more in cat position", "mild")
    ],
    "childs pose": [
        ("torso_lowering", "<", 25, "Reach arms further forward", "mild")
    ],
    "figure-4 stretch": [
        ("hip_opening", "<", 85, "Pull ankle closer to open hip", "moderate")
    ],
}

def detect_faults(exercise, m, profile, phase):
    faults = []
    rules = fault_rules.get(exercise, [])
    for rule in rules:
        key, op, thresh, msg, sev = rule[:5]
        extra = rule[5:]  # could include 'phase' check
        ignore = False
        if extra and extra[0] == "phase":
            need_phase = extra[1]
            if phase != need_phase:
                ignore = True
        if ignore:
            continue
        val = m.get(key)
        if val is None:
            continue
        condition = (val < thresh) if op == "<" else (val > thresh)
        if condition:
            faults.append((msg, sev))
    return faults

# ============================
# 8. FORM SCORE 
# ============================

form_penalties = {
    "squat": [("back_angle", "<", 110, 25), ("knee_angle", ">", 105, 15), ("torso_angle", ">", 50, 20)],
    "deadlift": [("back_angle", "<", 120, 25), ("torso_angle", ">", 70, 20)],
    "pushup": [("elbow_angle", ">", 155, 15), ("back_angle", "<", 145, 25)],
    "bicep curl": [("elbow_angle", ">", 160, 15)],
    "shoulder press": [("arm_verticality", "<", 65, 20)],
    "dumbbell row": [("back_angle", "<", 140, 20)],
    "lunge": [("front_knee_angle", ">", 110, 15)],
    "step-ups": [("front_knee_angle", ">", 110, 15)],
    "glute bridge": [("hip_extension", "<", 150, 20)],
    "plank": [("body_line", "<", 165, 20)],
    "bird-dog": [("spine_alignment", "<", 150, 15)],
    "hamstring stretch": [("leg_straightness", "<", 165, 20)],
    "quad stretch": [("heel_to_glute", ">", 70, 20)],
    "shoulder stretch": [("arm_across_chest", ">", 95, 10)],
    "triceps stretch": [("elbow_overhead", "<", 25, 15)],
    "hip flexor stretch": [("front_leg_angle", "<", 95, 20)],
    "cat-cow stretch": [("spine_curve", ">", 160, 10)],
    "childs pose": [("torso_lowering", "<", 30, 10)],
    "figure-4 stretch": [("hip_opening", "<", 90, 15)],
}

def calculate_form_score(exercise, m):
    score = 100
    for key, op, thresh, penalty in form_penalties.get(exercise, []):
        val = m.get(key)
        if val is None:
            continue
        condition = (val < thresh) if op == "<" else (val > thresh)
        if condition:
            score -= penalty
    return max(score, 0)

# ============================
# 9. SHOW INSTRUCTION POPUP
# ============================

def show_instruction_popup(exercise):
    instructions = workout_instructions.get(exercise) or stretch_instructions.get(exercise) or \
        "Follow safe form and perform the exercise carefully."
    
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
# 10. RUN EXERCISE SESSION (Simplified rep logic)
# ============================

rep_definitions = {
    "squat": ("knee_angle", lambda a: a < 100, lambda a: a > COMPLETION_TRIGGER),
    "goblet squat": ("knee_angle", lambda a: a < 100, lambda a: a > COMPLETION_TRIGGER),
    "deadlift": ("knee_angle", lambda a: a < 100, lambda a: a > COMPLETION_TRIGGER),
    "pushup": ("elbow_angle", lambda a: a < 90, lambda a: a > 160),
    "incline pushup": ("elbow_angle", lambda a: a < 90, lambda a: a > 160),
    "bicep curl": ("elbow_angle", lambda a: a < 90, lambda a: a > 160),
    "shoulder press": ("arm_verticality", lambda a: a < 60, lambda a: a > 80),
    "dumbbell row": ("row_angle", lambda a: a < 60, lambda a: a > 80),
}

stretch_exercises = {
    "hamstring stretch", "quad stretch", "shoulder stretch", "triceps stretch",
    "hip flexor stretch", "cat-cow stretch", "childs pose", "figure-4 stretch", "plank"
}

def run_exercise_session(exercise, sets, reps_per_set, break_time=60):
    DEPTH_TARGET = TARGET_DEPTH_LIMIT  # Adjust if needed per condition
    set_count = 0

    show_instruction_popup(exercise)

    USE_SMOOTHING = True
    SMOOTH_ALPHA = 0.2

    stretch_exercises = {
        "hamstring stretch", "quad stretch", "shoulder stretch", "triceps stretch",
        "hip flexor stretch", "cat-cow stretch", "childs pose", "figure-4 stretch", "plank"
    }

    knee_smooth = None
    elbow_smooth = None
    arm_vert_smooth = None
    row_ang_smooth = None

    while set_count < sets:
        rep = 0
        rep_started = False
        frame_buffer = []
        stretch_hold_start = None

        pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        cap = cv2.VideoCapture(0)

        for _ in range(5): cap.read()

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
                lm = results.pose_landmarks.landmark
                orientation = estimate_orientation(lm)
                mp_drawing.draw_landmarks(portrait, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                expected_view = EXERCISE_VIEWS.get(exercise, "side")
                if orientation != expected_view:
                    cv2.putText(portrait, f"Face {expected_view}-on for {exercise}", (30, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.imshow("AI Trainer", portrait)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        return
                    continue

                m = extract_pose_metrics(exercise, lm)

                # Apply smoothing per angle used
                if exercise in ["squat", "goblet squat", "deadlift"]:
                    knee_ang = m.get("knee_angle", 180)
                    if USE_SMOOTHING:
                        knee_smooth = knee_ang if knee_smooth is None or np.isnan(knee_smooth) else SMOOTH_ALPHA * knee_ang + (1 - SMOOTH_ALPHA) * knee_smooth
                        knee_to_use = knee_smooth
                    else:
                        knee_to_use = knee_ang
                    phase = "down" if knee_to_use < 100 else "up"
                    frame_buffer.append(knee_to_use)
                    smoothed = np.mean(frame_buffer[-5:])
                    if not rep_started and smoothed < DEPTH_TARGET:
                        rep_started = True
                    elif rep_started and smoothed > COMPLETION_TRIGGER:
                        rep += 1
                        rep_started = False

                elif exercise in ["pushup", "incline pushup"]:
                    elbow_ang = m.get("elbow_angle", 180)
                    if USE_SMOOTHING:
                        elbow_smooth = elbow_ang if elbow_smooth is None or np.isnan(elbow_smooth) else SMOOTH_ALPHA * elbow_ang + (1 - SMOOTH_ALPHA) * elbow_smooth
                        elbow_to_use = elbow_smooth
                    else:
                        elbow_to_use = elbow_ang
                    phase = "down" if elbow_to_use < 90 else "up"
                    if not rep_started and elbow_to_use < 90:
                        rep_started = True
                    elif rep_started and elbow_to_use > 160:
                        rep += 1
                        rep_started = False

                elif exercise == "bicep curl":
                    elbow_ang = m.get("elbow_angle", 180)
                    if USE_SMOOTHING:
                        elbow_smooth = elbow_ang if elbow_smooth is None or np.isnan(elbow_smooth) else SMOOTH_ALPHA * elbow_ang + (1 - SMOOTH_ALPHA) * elbow_smooth
                        elbow_to_use = elbow_smooth
                    else:
                        elbow_to_use = elbow_ang
                    phase = "curl" if elbow_to_use < 90 else "extend"
                    if not rep_started and elbow_to_use < 90:
                        rep_started = True
                    elif rep_started and elbow_to_use > 160:
                        rep += 1
                        rep_started = False

                elif exercise == "shoulder press":
                    arm_vert = m.get("arm_verticality", 90)
                    if USE_SMOOTHING:
                        arm_vert_smooth = arm_vert if arm_vert_smooth is None or np.isnan(arm_vert_smooth) else SMOOTH_ALPHA * arm_vert + (1 - SMOOTH_ALPHA) * arm_vert_smooth
                        arm_vert_to_use = arm_vert_smooth
                    else:
                        arm_vert_to_use = arm_vert
                    phase = "press" if arm_vert_to_use < 60 else "rest"
                    if not rep_started and arm_vert_to_use < 60:
                        rep_started = True
                    elif rep_started and arm_vert_to_use > 80:
                        rep += 1
                        rep_started = False

                elif exercise == "dumbbell row":
                    row_ang = m.get("row_angle", 90)
                    if USE_SMOOTHING:
                        row_ang_smooth = row_ang if row_ang_smooth is None or np.isnan(row_ang_smooth) else SMOOTH_ALPHA * row_ang + (1 - SMOOTH_ALPHA) * row_ang_smooth
                        row_ang_to_use = row_ang_smooth
                    else:
                        row_ang_to_use = row_ang
                    phase = "pull" if row_ang_to_use < 60 else "lower"
                    if not rep_started and row_ang_to_use < 60:
                        rep_started = True
                    elif rep_started and row_ang_to_use > 80:
                        rep += 1
                        rep_started = False

                elif exercise in ["lunge", "step-ups", "glute bridge", "bird-dog"]:
                    rep += 1
                    phase = "hold"

                elif exercise in stretch_exercises:
                    phase = "hold"
                    if not stretch_hold_start:
                        stretch_hold_start = time.time()
                    else:
                        elapsed = time.time() - stretch_hold_start
                        cv2.putText(portrait, f"Holding: {int(elapsed)}s/{STRETCH_HOLD_TIME}s", (30, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        if elapsed >= STRETCH_HOLD_TIME:
                            rep += 1
                            stretch_hold_start = None

                faults = detect_faults(exercise, m, {}, phase)
                score = calculate_form_score(exercise, m)

                color_map = {"critical": (0, 0, 255), "moderate": (0, 165, 255), "mild": (0, 255, 255)}
                if faults:
                    for i, (fault, sev) in enumerate(faults):
                        cv2.putText(portrait, fault, (30, 210 + i*20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_map.get(sev, (255, 255, 255)), 2)
                else:
                    cv2.putText(portrait, "Good form! Keep going!", (30, 210),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.putText(portrait, f"Form Score: {score}/100", (30, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if score >= 80 else (0, 0, 255), 2)

            else:
                phase = "unknown"

            cv2.putText(portrait, f"{exercise.capitalize()} Set {set_count+1}/{sets} Rep/Hold: {rep}/{reps_per_set}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow("AI Trainer", portrait)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return

            if rep >= reps_per_set:
                break

        cap.release()
        cv2.destroyAllWindows()
        set_count += 1
        if set_count < sets:
            show_break_timer(break_time)

# ============================
# 11. TKINTER APP
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
                if not line or len(line) < 4:
                    continue
                line = re.sub(r"^[\-\u2022]\s*", "", line)
                match = re.search(r"([a-z\s\-]+)\s+(\d+)[x×](\d+)", line)
                if match:
                    exercise_text = match.group(1).strip()
                    sets = int(match.group(2))
                    reps = int(match.group(3))
                    for key in EXERCISE_VIEWS:
                        if key in exercise_text:
                            generated_workout.append((key, sets, reps))
                            break

            self.show_exercise_selection()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def show_exercise_selection(self):
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        tk.Label(
            self.scrollable_frame,
            text="Choose exercise to start:",
            font=('Arial', 12, 'bold')
        ).pack(anchor="w", pady=(10, 5))

        for name, sets, reps in generated_workout:
            btn_text = f"{name.capitalize()} ({sets}x{reps})"
            tk.Button(
                self.scrollable_frame,
                text=btn_text,
                command=lambda n=name, s=sets, r=reps: self.start_tracking_and_close(n, s, r)
            ).pack(fill="x", pady=2, padx=5)

    def start_tracking_and_close(self, name, sets, reps):
        self.master.destroy()
        run_exercise_session(name, sets, reps)

# ========================
# 12. MAIN
# ========================

if __name__ == "__main__":
    root = tk.Tk()
    app = RecoveryApp(root)
    root.mainloop()
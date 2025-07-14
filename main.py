import cv2
import mediapipe as mp
import math

# ========== Setup ==========
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

# ========== Angle Calculation ==========
def calculate_angle(a, b, c):
    ax, ay = a.x, a.y
    bx, by = b.x, b.y
    cx, cy = c.x, c.y

    angle = math.degrees(math.atan2(cy - by, cx - bx) -
                         math.atan2(ay - by, ax - bx))
    return abs(angle)

# ========== Squat Form Checker ==========
def check_squat_form(landmarks, frame, profile):
    if not landmarks:
        return

    # Get key landmarks
    left_hip = landmarks[mp_holistic.PoseLandmark.LEFT_HIP]
    left_knee = landmarks[mp_holistic.PoseLandmark.LEFT_KNEE]
    left_ankle = landmarks[mp_holistic.PoseLandmark.LEFT_ANKLE]
    left_shoulder = landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER]

    # Calculate angles
    back_angle = calculate_angle(left_shoulder, left_hip, left_knee)
    knee_angle = calculate_angle(left_hip, left_knee, left_ankle)

    # Feedback based on injury profile
    if profile == "acl":
        if knee_angle < 90:
            cv2.putText(frame, "Avoid deep squats!", (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        if back_angle < 150:
            cv2.putText(frame, "Straighten your back!", (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if knee_angle < 70:
            cv2.putText(frame, "Squat deeper!", (30, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

# ========== Injury Profile ==========
injury_type = input("Enter injury/disability (e.g. ACL, stroke, shoulder): ").lower()

# ========== Webcam Loop ==========
cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(
    model_complexity=2,
    smooth_landmarks=True,
    enable_segmentation=False,
    refine_face_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as holistic:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Flip and convert to RGB
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process frame
        results = holistic.process(rgb)

        # Draw landmarks
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            mp_drawing_styles.get_default_pose_landmarks_style())

        # Check form
        check_squat_form(results.pose_landmarks.landmark, frame, injury_type)

        # UI overlay
        cv2.putText(frame, "AI Personal Trainer", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Adaptive Trainer", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

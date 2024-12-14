import cv2
import mediapipe as mp
import pyautogui

# Initialize Mediapipe Hand model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Constants for screen resolution and smoothing
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()
SMOOTHING = 5
prev_x, prev_y = 0, 0

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a mirrored view
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Convert the frame to RGB for Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get normalized landmark positions
            landmarks = hand_landmarks.landmark
            index_finger_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
            wrist = landmarks[mp_hands.HandLandmark.WRIST]

            # Map landmark coordinates to screen size
            x = int(index_finger_tip.x * SCREEN_WIDTH)
            y = int(index_finger_tip.y * SCREEN_HEIGHT)

            # Smooth movement by interpolating
            curr_x = prev_x + (x - prev_x) / SMOOTHING
            curr_y = prev_y + (y - prev_y) / SMOOTHING
            pyautogui.moveTo(curr_x, curr_y)
            prev_x, prev_y = curr_x, curr_y

            # Calculate finger positions to detect gestures
            fingers_folded = []
            for finger_tip_id in [8, 12, 16, 20]:  # Tip landmarks of all fingers
                finger_tip = landmarks[finger_tip_id]
                finger_dip = landmarks[finger_tip_id - 2]

                # Check if finger is folded by comparing tip and DIP landmarks
                if finger_tip.y > finger_dip.y:
                    fingers_folded.append(True)
                else:
                    fingers_folded.append(False)

            if all(fingers_folded):
                pyautogui.click()  # Left click when all fingers are curled

            if not fingers_folded[1]:  # Index finger pointing
                pyautogui.scroll(20)  # Scroll up

            if not fingers_folded[0]:  # Thumb pointing
                pyautogui.scroll(-20)  # Scroll down

    # Display the frame
    cv2.imshow("Tesla Hand Gesture Mouse Control", frame)

    # Break the loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
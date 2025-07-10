import cv2
import numpy as np
import mediapipe as mp

# Initialize mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Create canvas
canvas = np.zeros((480, 640, 3), dtype=np.uint8)

# Default drawing settings
draw_color = (255, 0, 0)  # Bright blue
thickness = 4
prev_x, prev_y = 0, 0

# Define bright color buttons
color_buttons = {
    "CLEAR": ((10, 10), (100, 60), (80, 80, 80)),         # Dark gray
    "RED":   ((110, 10), (200, 60), (0, 0, 255)),         # Bright red
    "YELLOW":((210, 10), (300, 60), (0, 255, 255)),       # Bright yellow
    "PINK":  ((310, 10), (400, 60), (255, 0, 255)),       # Bright pink
    "BLUE":  ((410, 10), (500, 60), (255, 50, 50)),       # Very bright blue
    "GREEN": ((510, 10), (600, 60), (0, 255, 0)),         # Bright green
}

# Start webcam
cap = cv2.VideoCapture(0)

with mp_hands.Hands(max_num_hands=1,
                    min_detection_confidence=0.7,
                    min_tracking_confidence=0.5) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        # Draw all color buttons
        for name, ((x1, y1), (x2, y2), color) in color_buttons.items():
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
            text_color = (255, 255, 255) if name != "BLACK" else (255, 255, 255)
            cv2.putText(frame, name, (x1 + 10, y2 - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

        # Process hand and get finger position
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                lm = hand_landmarks.landmark[8]  # Index fingertip
                x, y = int(lm.x * w), int(lm.y * h)

                # Check if clicking on any button
                for name, ((x1, y1), (x2, y2), color) in color_buttons.items():
                    if x1 < x < x2 and y1 < y < y2:
                        if name == "CLEAR":
                            canvas = np.zeros((480, 640, 3), dtype=np.uint8)
                        else:
                            draw_color = color
                        prev_x, prev_y = 0, 0  # Reset to prevent drawing while clicking
                        break
                else:
                    # Start drawing if not in button area
                    if prev_x == 0 and prev_y == 0:
                        prev_x, prev_y = x, y
                    cv2.line(canvas, (prev_x, prev_y), (x, y), draw_color, thickness)
                    prev_x, prev_y = x, y

                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        else:
            prev_x, prev_y = 0, 0

        # Combine canvas with live feed
        output = cv2.addWeighted(frame, 1, canvas, 1, 0)
        cv2.imshow("Finger Paint App", output)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()

import cv2
import mediapipe as mp
import autopy
import math
import time
from pynput.mouse import Button, Controller as MouseController

# Initialize Mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Set up Autopy screen size
screen_width, screen_height = autopy.screen.size()

# Initialize OpenCV video capture
cap = cv2.VideoCapture(0)

# Initialize click states
left_click_state = False
right_click_state = False
last_click_time = 0
drag_start_pos = None
last_index_finger_pos = None

# Initialize pynput mouse controller
mouse = MouseController()

# Initialize variables for frame rate calculation and function display
pTime = 0
active_function = None
active_function_time = None

with mp_hands.Hands(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)

        # Calculate and display frame rate
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

        # Check if active function needs to be removed
        if active_function and time.time() - active_function_time > 2:  # Display function name for 2 seconds
            active_function = None

        # Display active function name on screen
        if active_function:
            cv2.putText(frame, active_function, (20, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        rgb_frame.flags.writeable = False
        # Process the image with MediaPipe Hands.
        results = hands.process(rgb_frame)

        # Draw the hand landmarks on the frame.
        rgb_frame.flags.writeable = True
        rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    rgb_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get the landmarks
                index_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                index_finger_x, index_finger_y = int(index_finger.x * screen_width), int(index_finger.y * screen_height)
                middle_finger = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                middle_finger_x, middle_finger_y = int(middle_finger.x * screen_width), int(middle_finger.y * screen_height)
                ring_finger = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                ring_finger_x, ring_finger_y = int(ring_finger.x * screen_width), int(ring_finger.y * screen_height)

                # Move the cursor with index finger
                autopy.mouse.move(index_finger_x, index_finger_y)

                # Check if index, middle, and ring fingers are moving up together for scrolling up
                if index_finger_y < middle_finger_y and middle_finger_y < ring_finger_y:
                    mouse.scroll(0, 1)  # Scroll up
                    active_function = "Scrolling"
                    active_function_time = time.time()

                # Check if index, middle, and ring fingers are moving down together for scrolling down
                elif index_finger_y > middle_finger_y and middle_finger_y > ring_finger_y:
                    mouse.scroll(0, -1)  # Scroll down
                    active_function = "Scrolling"
                    active_function_time = time.time()

                # Check if index and middle fingers are close to each other for left click
                distance_left_click = math.sqrt((index_finger_x - middle_finger_x) ** 2 + (index_finger_y - middle_finger_y) ** 2)
                if distance_left_click < 50:
                    if not left_click_state:
                        current_time = time.time()
                        if current_time - last_click_time < 0.1:  # If the time difference is less than 0.1 seconds
                            mouse.click(Button.left, 2)  # Double-click
                        else:
                            mouse.click(Button.left)  # Single-click
                        last_click_time = current_time
                        left_click_state = True
                        active_function = "Left Clicking"
                        active_function_time = time.time()
                else:
                    left_click_state = False

                # Check if thumb and index fingers are close to each other for right click
                thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                distance_right_click = math.sqrt((index_finger_x - thumb.x * screen_width) ** 2 + (index_finger_y - thumb.y * screen_height) ** 2)
                if distance_right_click < 50:
                    if not right_click_state:
                        mouse.click(Button.right)  # Right-click
                        right_click_state = True
                        active_function = "Right Clicking"
                        active_function_time = time.time()
                else:
                    if right_click_state:
                        right_click_state = False

                # Check if all fingers are closed (fist gesture) for dragging
                if index_finger_y > middle_finger_y \
                        and middle_finger_y > ring_finger_y \
                        and ring_finger_y > hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * screen_height:

                    # Start dragging if not already dragging
                    if not drag_start_pos:
                        drag_start_pos = (index_finger_x, index_finger_y)
                        mouse.press(Button.left)
                        active_function = "Dragging"
                        active_function_time = time.time()
                    else:
                        # Calculate the distance to move the cursor
                        dx = index_finger_x - drag_start_pos[0]
                        dy = index_finger_y - drag_start_pos[1]

                        # Update drag start position
                        drag_start_pos = (index_finger_x, index_finger_y)

                        # Move the cursor
                        mouse.move(dx, dy)

                else:
                    # End dragging if fist gesture is not detected
                    if drag_start_pos:
                        mouse.release(Button.left)
                        drag_start_pos = None

                # Check if hand is open for dropping
                if index_finger_y < middle_finger_y:
                    if drag_start_pos:
                        mouse.release(Button.left)
                        active_function = "Dropping"
                        active_function_time = time.time()
                        drag_start_pos = None

        # Show the frame
        cv2.imshow('AI Virtual Mouse', rgb_frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import random
import time

model_path = 'hand_landmarker.task'
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

def check_gesture(hand_landmarks):
    if not hand_landmarks:
        return 'unknown'
    
    fingers = 0
    
    if hand_landmarks[4].x < hand_landmarks[3].x:
        fingers += 1
    
    tips = [8, 12, 16, 20]
    knuckles = [6, 10, 14, 18]
    
    for tip, knuckle in zip(tips, knuckles):
        if hand_landmarks[tip].y < hand_landmarks[knuckle].y:
            fingers += 1
    
    if fingers == 0:
        return 'rock'
    elif fingers == 5:
        return 'paper'
    elif fingers == 2:
        return 'scissors'
    else:
        return 'unknown'

pink = (220, 160, 200)
white = (255, 255, 255)
gray = (100, 100, 100)

my_score = 0
ai_score = 0
options = ['rock', 'paper', 'scissors']

waiting = 0
counting = 1
showing = 2

current_state = waiting
count_start = 0
show_start = 0
my_choice = None
computer_choice = None
message = ""

camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    height, width = frame.shape[:2]
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_img)
    
    gesture = 'unknown'
    
    if result.hand_landmarks:
        hand = result.hand_landmarks[0]
        
        for point in hand:
            x = int(point.x * width)
            y = int(point.y * height)
            cv2.circle(frame, (x, y), 5, pink, -1)
        
        gesture = check_gesture(hand)
    
    cv2.rectangle(frame, (0, 0), (width, 60), white, -1)
    cv2.putText(frame, 'rock paper scissors', (20, 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, gray, 2)
    
    if current_state == waiting:
        cv2.putText(frame, f'{gesture}', (width//2 - 50, height - 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, pink, 2)
        
        if gesture in options:
            cv2.putText(frame, 'press space', (width//2 - 80, height - 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, white, 2)
    
    elif current_state == counting:
        time_passed = time.time() - count_start
        num = 3 - int(time_passed)
        
        if num > 0:
            cv2.putText(frame, str(num), (width//2 - 30, height//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 4, pink, 4)
        else:
            current_state = showing
            show_start = time.time()
            
            if my_choice == computer_choice:
                message = 'tie'
            elif (my_choice == 'rock' and computer_choice == 'scissors') or \
                 (my_choice == 'paper' and computer_choice == 'rock') or \
                 (my_choice == 'scissors' and computer_choice == 'paper'):
                my_score += 1
                message = 'you win!'
            else:
                ai_score += 1
                message = 'ai wins'
    
    elif current_state == showing:
        cv2.putText(frame, message, (width//2 - 80, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, pink, 3)
        
        cv2.putText(frame, f'you: {my_choice}', (100, height//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, white, 2)
        cv2.putText(frame, f'ai: {computer_choice}', (width - 300, height//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, white, 2)
        
        if time.time() - show_start > 3:
            current_state = waiting
    
    cv2.putText(frame, f'{my_score} - {ai_score}', (width - 100, 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, gray, 2)
    
    cv2.imshow('game', frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
    elif key == ord(' ') and current_state == waiting and gesture in options:
        my_choice = gesture
        computer_choice = random.choice(options)
        current_state = counting
        count_start = time.time()

camera.release()
cv2.destroyAllWindows()
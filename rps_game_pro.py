import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import random
import time
import numpy as np

# Load hand detector
model_path = 'hand_landmarker.task'
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

def classify_gesture(hand_landmarks):
    """Classify rock, paper, scissors based on extended fingers"""
    if not hand_landmarks:
        return 'unknown'
    
    fingers_up = 0
    
    # Thumb
    if hand_landmarks[4].x < hand_landmarks[3].x:
        fingers_up += 1
    
    # Other fingers
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]
    
    for tip, pip in zip(finger_tips, finger_pips):
        if hand_landmarks[tip].y < hand_landmarks[pip].y:
            fingers_up += 1
    
    if fingers_up == 0:
        return 'rock'
    elif fingers_up == 5:
        return 'paper'
    elif fingers_up == 2:
        return 'scissors'
    else:
        return 'unknown'

def draw_text_with_background(img, text, pos, font_scale=1, thickness=2, 
                               text_color=(255,255,255), bg_color=(0,0,0), padding=10):
    """Draw text with a background box"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    
    x, y = pos
    box_coords = (
        (x - padding, y - text_size[1] - padding),
        (x + text_size[0] + padding, y + padding)
    )
    
    cv2.rectangle(img, box_coords[0], box_coords[1], bg_color, -1)
    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness)

def get_emoji_for_choice(choice):
    """Get emoji representation"""
    emoji_map = {
        'rock': 'ROCK',
        'paper': 'PAPER', 
        'scissors': 'SCISSORS',
        'unknown': '???'
    }
    return emoji_map.get(choice, choice.upper())

# Game state
player_score = 0
ai_score = 0
choices = ['rock', 'paper', 'scissors']

# Game states
WAITING = 0
COUNTDOWN = 1
SHOWING_RESULT = 2

game_state = WAITING
countdown_start = 0
result_start = 0
player_choice = None
ai_choice = None
result_message = ""
result_color = (255, 255, 255)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("=== ROCK PAPER SCISSORS - ULTIMATE EDITION ===")
print("Show your gesture and press SPACE to play!")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    
    # Detect hand
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect(mp_image)
    
    current_gesture = 'unknown'
    
    # Draw hand landmarks with glow effect
    if detection_result.hand_landmarks:
        landmarks = detection_result.hand_landmarks[0]
        
        # Draw connections
        connections = [(0,1),(1,2),(2,3),(3,4),  # Thumb
                      (0,5),(5,6),(6,7),(7,8),    # Index
                      (5,9),(9,10),(10,11),(11,12), # Middle
                      (9,13),(13,14),(14,15),(15,16), # Ring
                      (13,17),(17,18),(18,19),(19,20), # Pinky
                      (0,17)]
        
        for connection in connections:
            start_idx, end_idx = connection
            start = landmarks[start_idx]
            end = landmarks[end_idx]
            start_point = (int(start.x * w), int(start.y * h))
            end_point = (int(end.x * w), int(end.y * h))
            cv2.line(frame, start_point, end_point, (0, 255, 150), 3)
        
        # Draw landmarks
        for landmark in landmarks:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(frame, (x, y), 8, (0, 255, 255), -1)
            cv2.circle(frame, (x, y), 10, (255, 255, 255), 2)
        
        current_gesture = classify_gesture(landmarks)
    
    # GAME LOGIC
    if game_state == WAITING:
        # Top banner
        cv2.rectangle(frame, (0, 0), (w, 100), (20, 20, 20), -1)
        draw_text_with_background(frame, "ROCK PAPER SCISSORS", (w//2 - 220, 65), 
                                  1.5, 3, (255, 255, 0), (20, 20, 20), 0)
        
        # Show current gesture
        gesture_display = get_emoji_for_choice(current_gesture)
        gesture_color = (0, 255, 0) if current_gesture in choices else (100, 100, 100)
        draw_text_with_background(frame, f"YOUR HAND: {gesture_display}", 
                                 (50, 150), 0.8, 2, gesture_color, (40, 40, 40))
        
        # Instructions
        if current_gesture in choices:
            draw_text_with_background(frame, "Press SPACE to play!", 
                                     (w//2 - 150, h - 100), 1, 2, (0, 255, 0), (40, 40, 40))
        else:
            draw_text_with_background(frame, "Show Rock, Paper, or Scissors", 
                                     (w//2 - 240, h - 100), 0.8, 2, (200, 200, 200), (40, 40, 40))
    
    elif game_state == COUNTDOWN:
        elapsed = time.time() - countdown_start
        countdown = 3 - int(elapsed)
        
        if countdown > 0:
            # Huge countdown number
            cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 0), -1)
            countdown_text = str(countdown)
            text_size = cv2.getTextSize(countdown_text, cv2.FONT_HERSHEY_SIMPLEX, 10, 20)[0]
            text_x = (w - text_size[0]) // 2
            text_y = (h + text_size[1]) // 2
            
            # Pulsing effect
            pulse = int(50 * (1 - (elapsed % 1)))
            color = (0, 255 - pulse, 255)
            
            cv2.putText(frame, countdown_text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 10, color, 20)
            
            draw_text_with_background(frame, "Get Ready!", (w//2 - 100, 150), 
                                     1.2, 3, (255, 255, 255), (0, 0, 0), 0)
        else:
            # SHOOT!
            game_state = SHOWING_RESULT
            result_start = time.time()
            
            # Determine winner
            if player_choice == ai_choice:
                result_message = "TIE!"
                result_color = (200, 200, 200)
            elif (player_choice == 'rock' and ai_choice == 'scissors') or \
                 (player_choice == 'paper' and ai_choice == 'rock') or \
                 (player_choice == 'scissors' and ai_choice == 'paper'):
                player_score += 1
                result_message = "YOU WIN!"
                result_color = (0, 255, 0)
            else:
                ai_score += 1
                result_message = "AI WINS!"
                result_color = (0, 100, 255)
    
    elif game_state == SHOWING_RESULT:
        # Show result
        cv2.rectangle(frame, (0, 0), (w, h//2), (20, 20, 20), -1)
        
        # Result message
        result_size = cv2.getTextSize(result_message, cv2.FONT_HERSHEY_SIMPLEX, 2, 4)[0]
        result_x = (w - result_size[0]) // 2
        cv2.putText(frame, result_message, (result_x, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, result_color, 4)
        
        # Show choices
        player_display = get_emoji_for_choice(player_choice)
        ai_display = get_emoji_for_choice(ai_choice)
        
        draw_text_with_background(frame, f"YOU: {player_display}", 
                                 (100, 200), 1, 2, (255, 255, 255), (40, 40, 40))
        draw_text_with_background(frame, f"AI: {ai_display}", 
                                 (w - 350, 200), 1, 2, (255, 255, 255), (40, 40, 40))
        
        # Check if result time is over
        if time.time() - result_start > 3:
            game_state = WAITING
    
    # Score display (always visible)
    score_bg = (30, 30, 30)
    cv2.rectangle(frame, (20, h - 80), (300, h - 20), score_bg, -1)
    cv2.rectangle(frame, (20, h - 80), (300, h - 20), (100, 100, 100), 3)
    
    cv2.putText(frame, f"YOU: {player_score}", (40, h - 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"AI: {ai_score}", (180, h - 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)
    
    # Controls
    draw_text_with_background(frame, "Q = Quit | R = Reset Score", 
                             (w - 350, h - 30), 0.5, 1, (150, 150, 150), (20, 20, 20))
    
    cv2.imshow('ROCK PAPER SCISSORS', frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
    elif key == ord('r'):
        player_score = 0
        ai_score = 0
    elif key == ord(' ') and game_state == WAITING and current_gesture in choices:
        # Start countdown
        player_choice = current_gesture
        ai_choice = random.choice(choices)
        game_state = COUNTDOWN
        countdown_start = time.time()

cap.release()
cv2.destroyAllWindows()
print(f"\n=== Final Score - You: {player_score}, AI: {ai_score} ===")
if player_score > ai_score:
    print("YOU ARE THE CHAMPION!")
elif ai_score > player_score:
    print("AI WINS THIS TIME!")
else:
    print("IT'S A TIE!")

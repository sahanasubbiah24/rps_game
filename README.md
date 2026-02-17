# hand gesture rock paper scissors

rock paper scissors game using computer vision for hand tracking

## setup
```bash
pip install mediapipe opencv-python
```

run:
```bash
python game.py
```

**controls:**
- space = play round
- q = quit

## how it works

mediapipe detects 21 landmark points on the hand. the program counts how many fingers are extended to determine the gesture:
- 0 fingers = rock
- 2 fingers = scissors  
- 5 fingers = paper
- 
## future improvements?

- improve gesture recognition to detect both palm-facing and back-of-hand orientations
- add support for more hand positions (tilted, rotated, etc)
- increase accuracy for edge cases where finger positioning is ambiguous

## tech stack

- python
- mediapipe (hand landmark detection)
- opencv (video processing)

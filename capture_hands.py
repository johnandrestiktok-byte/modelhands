import cv2
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR, "hand_signatures_pro")
BOX_SIZE = 500  # Match recognizer
IMAGE_SIZE = (200, 200)
MAX_IMAGES_PER_LETTER = 20
LETTERS = [chr(c) for c in range(ord('A'), ord('Z') + 1)]


def get_saved_count(letter):
    letter_dir = os.path.join(BASE_DIR, letter)
    if not os.path.isdir(letter_dir):
        return 0
    return sum(1 for f in os.listdir(letter_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg')))


def cleanup_old_images(letter_dir, max_images):
    if not os.path.isdir(letter_dir):
        return
    files = [f for f in os.listdir(letter_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    if len(files) <= max_images:
        return
    # Sort by capture number
    def get_number(f):
        try:
            return int(f.split('_')[1].split('.')[0])
        except:
            return 0
    files.sort(key=get_number)
    # Remove oldest (smallest numbers)
    to_remove = files[:len(files) - max_images]
    for f in to_remove:
        os.remove(os.path.join(letter_dir, f))
        print(f"Removed old image: {f}")


def draw_hand_joints(frame, hand_landmarks):
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20),
        (0, 17)
    ]
    h, w, _ = frame.shape
    landmarks = hand_landmarks.landmarks if hasattr(hand_landmarks, 'landmarks') else hand_landmarks
    points = []
    for lm in landmarks:
        x = int(lm.x * w)
        y = int(lm.y * h)
        points.append((x, y))

    for a, b in connections:
        if a < len(points) and b < len(points):
            cv2.line(frame, points[a], points[b], (0, 255, 255), 2)

    for x, y in points:
        cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)


def create_mp_image(mp, img):
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=img)


# MediaPipe setup
model_path = os.path.join(SCRIPT_DIR, "hand_landmarker.task")
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
hand_landmarker = vision.HandLandmarker.create_from_options(options)

# Camera setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Higher resolution for better accuracy
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)
cv2.namedWindow("Hand Sign Capture", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Hand Sign Capture", 1280, 720)

# Current letter to capture
letter_index = 0
current_letter = LETTERS[letter_index]
capture_count = get_saved_count(current_letter)

print(f"Hand Sign Capture Tool - Capturing letter: {current_letter}")
print("Press 'S' to save current hand image")
print("Press 'Q' to quit")
print("Press 'C' to change letter")

while True:
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Failed to read frame from webcam.")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    x1, y1 = int((w-BOX_SIZE)/2), int((h-BOX_SIZE)/2)
    x2, y2 = x1+BOX_SIZE, y1+BOX_SIZE

    # Check for hand in the full frame
    roi = frame[y1:y2, x1:x2]
    mp_image = create_mp_image(mp, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    result = hand_landmarker.detect(mp_image)
    hand_detected = bool(result.hand_landmarks)

    # Draw capture area
    color = (0, 255, 0) if hand_detected else (0, 0, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.line(frame, (x1 + BOX_SIZE//3, y1), (x1 + BOX_SIZE//3, y2), (200, 200, 200), 1)
    cv2.line(frame, (x1 + 2*BOX_SIZE//3, y1), (x1 + 2*BOX_SIZE//3, y2), (200, 200, 200), 1)
    cv2.line(frame, (x1, y1 + BOX_SIZE//3), (x2, y1 + BOX_SIZE//3), (200, 200, 200), 1)
    cv2.line(frame, (x1, y1 + 2*BOX_SIZE//3), (x2, y1 + 2*BOX_SIZE//3), (200, 200, 200), 1)

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            draw_hand_joints(frame, hand_landmarks)

    # Instructions
    status = "HAND DETECTED" if hand_detected else "NO HAND"
    
    cv2.putText(frame, f"Capture: {current_letter}", (x1, y1-40), cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Saved: {capture_count}", (x1, y1-10), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, status, (x1, y2-10), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "S: Save | C: Next letter | A-Z: Select letter | Q: Quit", (x1, y2+25), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Hand Sign Capture", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == ord('Q'):
        break
    elif (key == ord('s') or key == ord('S')) and hand_detected:
        # Save current ROI only if hand is detected
        letter_dir = os.path.join(BASE_DIR, current_letter)
        os.makedirs(letter_dir, exist_ok=True)
        capture_count += 1
        filename = f"capture_{capture_count}.jpg"
        filepath = os.path.join(letter_dir, filename)

        cv2.imwrite(filepath, roi)
        print(f"Saved: {filepath}")
        cleanup_old_images(letter_dir, MAX_IMAGES_PER_LETTER)
        capture_count = get_saved_count(current_letter)
    elif (key == ord('s') or key == ord('S')) and not hand_detected:
        print("No hand detected - cannot save")
    elif key == ord('c') or key == ord('C'):
        letter_index = (letter_index + 1) % len(LETTERS)
        current_letter = LETTERS[letter_index]
        capture_count = get_saved_count(current_letter)
        print(f"Changed to letter: {current_letter}")
    elif ord('A') <= key <= ord('Z') or ord('a') <= key <= ord('z'):
        normalized = key if key <= ord('Z') else key - 32
        letter_index = normalized - ord('A')
        current_letter = LETTERS[letter_index]
        capture_count = get_saved_count(current_letter)
        print(f"Changed to letter: {current_letter}")

cap.release()
cv2.destroyAllWindows()
print("Capture session ended.")

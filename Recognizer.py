import cv2
import numpy as np
import os
import string
from typing import Tuple, Optional, Dict, List

class SignLanguageRecognizer:
    MAX_SIGNATURES_PER_LETTER = 20
    REFERENCE_IMAGE_NAME = "asl_reference.png"

    def __init__(self, base_dir: str = "hand_signatures_pro", box_size: int = 240):
        self.base_dir = base_dir
        self.box_size = box_size
        self.alphabet = list(string.ascii_uppercase)
        self.signature_size = (40, 40)
        self.signature_vectors: np.ndarray = np.zeros((0, self.signature_size[0] * self.signature_size[1]), dtype=np.float32)
        self.labels: List[int] = []
        self.label_map: Dict[int, str] = {}
        self.is_initialized = False
        self.detect_size = (320, 240)
        self.hand_landmarker = None
        self.mp = None
        self.reference_image = None
        self._init_hand_landmarker()
        self._load_reference_image()

    def load_model(self) -> bool:
        signatures = []
        labels = []

        for i, letter in enumerate(self.alphabet):
            path = os.path.join(self.base_dir, letter)
            if not os.path.isdir(path):
                continue

            files = sorted(
                [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
            )
            files = files[: self.MAX_SIGNATURES_PER_LETTER]

            for img_name in files:
                img = cv2.imread(os.path.join(path, img_name), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, self.signature_size)
                vector = img.astype(np.float32).flatten()
                norm = np.linalg.norm(vector)
                if norm < 1e-6:
                    continue
                signatures.append(vector / norm)
                labels.append(i)

            if files:
                self.label_map[i] = letter

        if not signatures:
            return False

        self.signature_vectors = np.stack(signatures)
        self.labels = labels
        self.is_initialized = True
        return True

    def _init_hand_landmarker(self) -> None:
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            self.mp = mp
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
            self.hand_landmarker = vision.HandLandmarker.create_from_options(options)
        except Exception:
            self.hand_landmarker = None
            self.mp = None

    def _load_reference_image(self) -> None:
        reference_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.REFERENCE_IMAGE_NAME)
        if os.path.isfile(reference_path):
            image = cv2.imread(reference_path)
            if image is not None:
                self.reference_image = image

    def _draw_hand_joints(self, frame: np.ndarray, hand_landmarks) -> None:
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (17, 18), (18, 19), (19, 20),
            (0, 17)
        ]
        h, w, _ = frame.shape
        points = []
        landmarks = hand_landmarks.landmarks if hasattr(hand_landmarks, 'landmarks') else hand_landmarks
        for lm in landmarks:
            x = int(lm.x * w)
            y = int(lm.y * h)
            points.append((x, y))

        for a, b in connections:
            if a < len(points) and b < len(points):
                cv2.line(frame, points[a], points[b], (0, 255, 255), 2)

        for x, y in points:
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

    
    def recognize(self, frame: np.ndarray) -> Tuple[str, int]:
        if not self.is_initialized:
            return "Model not initialized", 0

        h, w, _ = frame.shape
        x1 = max(0, int((w - self.box_size) / 2))
        y1 = max(0, int((h - self.box_size) / 2))
        x2 = min(w, x1 + self.box_size)
        y2 = min(h, y1 + self.box_size)

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return "ALIGN HAND", 0

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, self.signature_size)
        vector = gray.astype(np.float32).flatten()
        norm = np.linalg.norm(vector)
        if norm < 1e-6:
            return "ALIGN HAND", 0

        vector /= norm
        scores = self.signature_vectors.dot(vector)
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])

        if best_score < 0.60:
            return "ALIGN HAND", 0

        label_id = self.labels[best_idx]
        result = self.label_map.get(label_id, "Unknown")
        confidence = int(min(100, max(0, best_score * 100)))
        return result, confidence
    
    def process_video(self, video_source: str = "0") -> None:
        if not self.is_initialized and not self.load_model():
            print("Failed to initialize model.")
            return

        cap = cv2.VideoCapture(int(video_source) if video_source.isdigit() else video_source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not cap.isOpened():
            print(f"Error: Cannot open video source {video_source}")
            return

        cv2.namedWindow("Hand Recognizer", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Hand Recognizer", 800, 600)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            gesture, confidence = self.recognize(frame)

            if self.hand_landmarker is not None and self.mp is not None:
                small = cv2.resize(frame, self.detect_size)
                mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
                result = self.hand_landmarker.detect(mp_image)
                if result.hand_landmarks:
                    for hand_landmarks in result.hand_landmarks:
                        self._draw_hand_joints(frame, hand_landmarks)

            h, w, _ = frame.shape
            x1 = max(0, int((w - self.box_size) / 2))
            y1 = max(0, int((h - self.box_size) / 2))
            x2 = min(w, x1 + self.box_size)
            y2 = min(h, y1 + self.box_size)

            color = (0, 255, 0) if gesture != "ALIGN HAND" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            display_text = gesture if gesture == "ALIGN HAND" else f"SIGN: {gesture} {confidence}%"
            size_text = f"Frame: {w}x{h} | Box: {x2-x1}x{y2-y1}"
            cv2.putText(frame, display_text, (x1, max(20, y1 - 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, size_text, (x1, max(45, y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            cv2.imshow("Hand Recognizer", frame)

            if self.reference_image is not None:
                cv2.imshow("ASL Reference", self.reference_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    recognizer = SignLanguageRecognizer()
    recognizer.process_video()

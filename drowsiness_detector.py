
"""
============================================================
DRIVER DROWSINESS DETECTION SYSTEM (SECURE MEDIAPIPE VERSION)
============================================================

DESCRIPTION:
------------
This system monitors the driver's eyes and mouth using real-time
computer vision. It detects if the driver is drowsy (eyes closed
for too long) or yawning (mouth wide open). When detected, it
displays an on-screen alert and plays a beep sound.

This version uses Google Mediapipe Face Mesh instead of dlib and
does NOT require downloading the "shape_predictor_68_face_landmarks.dat"
file, making it more secure and easier to set up.

------------------------------------------------------------
FEATURES:
------------
1. Real-time face, eye, and mouth detection using Mediapipe.
2. Calculates Eye Aspect Ratio (EAR) to detect drowsiness.
3. Calculates Mouth Aspect Ratio (MAR) to detect yawns.
4. Displays EAR & MAR on screen for debugging.
5. Shows contours around eyes and mouth for visual reference.
6. Alerts with a warning message + sound alarm if drowsy.
7. Works with webcam or pre-recorded video file.
8. No external model file download required (secure).

------------------------------------------------------------
HOW IT WORKS:
-------------
1. Captures frames from webcam/video.
2. Detects facial landmarks using Mediapipe Face Mesh.
3. Calculates EAR (Eye Aspect Ratio):
   EAR = (distance between vertical eye points) / (horizontal eye width)
   When eyes are closed, EAR drops below a threshold.
4. Calculates MAR (Mouth Aspect Ratio):
   MAR = vertical mouth opening / horizontal mouth width
   When yawning, MAR rises above a threshold.
5. If EAR stays low for a set number of frames, triggers drowsiness alert.
6. If MAR stays high for a set number of frames, shows yawn alert.

------------------------------------------------------------
USAGE:
------
# Install dependencies:
pip install opencv-python mediapipe pygame numpy scipy

# Run with webcam (default):
python drowsiness_detector.py

# Run with video file:
python drowsiness_detector.py -v path/to/video.mp4

# Adjust EAR threshold and frame count:
python drowsiness_detector.py -t 0.3 -f 15

Controls:
---------
- Press 'q' to quit
- Press 'r' to reset alerts

============================================================
"""



# """
# Driver Drowsiness Detection System (Mediapipe Version)
# This system detects drowsiness by monitoring eye aspect ratio (EAR) and alerting when eyes are closed for too long.
# No external .dat file required (Mediapipe is used instead of dlib).
# """

import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import distance as dist
import pygame
import time
import argparse

class DrowsinessDetector:
    def __init__(self):
        # Initialize constants
        self.EYE_AR_THRESH = 0.25
        self.EYE_AR_CONSEC_FRAMES = 20
        self.YAWN_THRESH = 0.6
        self.YAWN_CONSEC_FRAMES = 10

        self.COUNTER = 0
        self.YAWN_COUNTER = 0
        self.ALARM_ON = False

        pygame.mixer.init()

        print("[INFO] Loading Mediapipe Face Mesh model...")
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(refine_landmarks=True)

        # Eye landmark indexes (mediapipe face mesh)
        self.LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
        # Mouth landmark indexes (mediapipe face mesh)
        self.MOUTH_IDX = [61, 291, 78, 308, 13, 14]

    def eye_aspect_ratio(self, eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def mouth_aspect_ratio(self, mouth):
        A = dist.euclidean(mouth[0], mouth[1])  # left-right corner
        B = dist.euclidean(mouth[2], mouth[3])  # upper lip - lower lip
        mar = B / A
        return mar

    def sound_alarm(self, message="Wake up!"):
        if not self.ALARM_ON:
            self.ALARM_ON = True
            pygame.mixer.Sound(self.create_beep()).play()
            print(f"[ALERT] {message}")

    def create_beep(self):
        sample_rate = 22050
        duration = 0.5
        frequency = 440

        frames = int(duration * sample_rate)
        arr = np.zeros(frames)
        for x in range(frames):
            arr[x] = np.sin(2 * np.pi * frequency * x / sample_rate)

        arr = (arr * 32767).astype(np.int16)
        arr = np.repeat(arr.reshape(frames, 1), 2, axis=1)
        sound = pygame.sndarray.make_sound(arr)
        return sound

    def process_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if results.multi_face_landmarks:
            for landmarks in results.multi_face_landmarks:
                h, w, _ = frame.shape

                # Get eye and mouth points
                left_eye = np.array([[int(landmarks.landmark[i].x * w),
                                      int(landmarks.landmark[i].y * h)]
                                     for i in self.LEFT_EYE_IDX])
                right_eye = np.array([[int(landmarks.landmark[i].x * w),
                                       int(landmarks.landmark[i].y * h)]
                                      for i in self.RIGHT_EYE_IDX])
                mouth = np.array([[int(landmarks.landmark[i].x * w),
                                   int(landmarks.landmark[i].y * h)]
                                  for i in self.MOUTH_IDX])

                leftEAR = self.eye_aspect_ratio(left_eye)
                rightEAR = self.eye_aspect_ratio(right_eye)
                ear = (leftEAR + rightEAR) / 2.0
                mar = self.mouth_aspect_ratio([mouth[0], mouth[1], mouth[4], mouth[5]])

                # Draw contours
                cv2.polylines(frame, [left_eye], True, (0, 255, 0), 1)
                cv2.polylines(frame, [right_eye], True, (0, 255, 0), 1)
                cv2.polylines(frame, [mouth], True, (0, 255, 0), 1)

                # EAR check
                if ear < self.EYE_AR_THRESH:
                    self.COUNTER += 1
                    if self.COUNTER >= self.EYE_AR_CONSEC_FRAMES:
                        cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        self.sound_alarm("Driver is drowsy!")
                else:
                    self.COUNTER = 0
                    self.ALARM_ON = False

                # MAR check (yawn)
                if mar > self.YAWN_THRESH:
                    self.YAWN_COUNTER += 1
                    if self.YAWN_COUNTER >= self.YAWN_CONSEC_FRAMES:
                        cv2.putText(frame, "YAWN DETECTED!", (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    self.YAWN_COUNTER = 0

                # Display EAR & MAR
                cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, f"MAR: {mar:.2f}", (300, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return frame

    def run(self, video_source=0):
        print("[INFO] Starting video stream...")
        cap = cv2.VideoCapture(video_source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to grab frame")
                break

            frame = self.process_frame(frame)
            cv2.imshow("Driver Drowsiness Detection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                self.COUNTER = 0
                self.YAWN_COUNTER = 0
                self.ALARM_ON = False

        cap.release()
        cv2.destroyAllWindows()
        pygame.mixer.quit()

def main():
    parser = argparse.ArgumentParser(description='Driver Drowsiness Detection System')
    parser.add_argument('-v', '--video', type=str, default='0',
                        help='Path to input video file or camera index (default: 0 for webcam)')
    parser.add_argument('-t', '--threshold', type=float, default=0.25,
                        help='Eye aspect ratio threshold (default: 0.25)')
    parser.add_argument('-f', '--frames', type=int, default=20,
                        help='Consecutive frames for triggering alert (default: 20)')

    args = parser.parse_args()

    video_source = args.video
    if video_source.isdigit():
        video_source = int(video_source)

    detector = DrowsinessDetector()
    detector.EYE_AR_THRESH = args.threshold
    detector.EYE_AR_CONSEC_FRAMES = args.frames

    print("=" * 50)
    print("DRIVER DROWSINESS DETECTION SYSTEM (Mediapipe)")
    print("=" * 50)
    print("Controls:")
    print("  Press 'q' to quit")
    print("  Press 'r' to reset alerts")
    print(f"Settings:")
    print(f"  Eye AR Threshold: {detector.EYE_AR_THRESH}")
    print(f"  Alert after {detector.EYE_AR_CONSEC_FRAMES} consecutive frames")
    print("=" * 50)

    detector.run(video_source)

if __name__ == "__main__":
    main()


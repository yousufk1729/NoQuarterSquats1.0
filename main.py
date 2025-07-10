import argparse
import logging

import cv2
import mediapipe as mp
import numpy as np

logging.getLogger('mediapipe').setLevel(logging.ERROR)

class SquatDepthEstimator:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.target_landmarks = [
            self.mp_pose.PoseLandmark.LEFT_HIP,
            self.mp_pose.PoseLandmark.RIGHT_HIP,
            self.mp_pose.PoseLandmark.LEFT_KNEE,
            self.mp_pose.PoseLandmark.RIGHT_KNEE,
            self.mp_pose.PoseLandmark.LEFT_ANKLE,
            self.mp_pose.PoseLandmark.RIGHT_ANKLE
        ]
        
        self.target_connections = [
            (self.mp_pose.PoseLandmark.LEFT_HIP, self.mp_pose.PoseLandmark.LEFT_KNEE),
            (self.mp_pose.PoseLandmark.LEFT_KNEE, self.mp_pose.PoseLandmark.LEFT_ANKLE),
            (self.mp_pose.PoseLandmark.RIGHT_HIP, self.mp_pose.PoseLandmark.RIGHT_KNEE),
            (self.mp_pose.PoseLandmark.RIGHT_KNEE, self.mp_pose.PoseLandmark.RIGHT_ANKLE)
        ]
        
    def calculate_angle(self, point1, point2, point3):
        try:
            a = np.array([point1.x, point1.y])
            b = np.array([point2.x, point2.y])
            c = np.array([point3.x, point3.y])
            ba = a - b
            bc = c - b
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
            return np.degrees(angle)
        except:
            return 0
    
    def get_squat_depth_category(self, angle):
        if angle >= 120:
            return "Quarter Squat", (0, 255, 0)  
        elif angle >= 100:
            return "Half Squat", (0, 255, 255)  
        elif angle >= 80:
            return "Parallel Squat", (0, 165, 255) 
        else:
            return "Deep Squat", (0, 0, 255)  
    
    def draw_specific_landmarks(self, frame, landmarks):
        height, width = frame.shape[:2]
        
        for connection in self.target_connections:
            start_landmark = landmarks[connection[0].value]
            end_landmark = landmarks[connection[1].value]
            start_point = (int(start_landmark.x * width), int(start_landmark.y * height))
            end_point = (int(end_landmark.x * width), int(end_landmark.y * height))
            cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
        
        for landmark_type in self.target_landmarks:
            landmark = landmarks[landmark_type.value]
            point = (int(landmark.x * width), int(landmark.y * height))
            cv2.circle(frame, point, 5, (0, 0, 255), -1)
    
    def process_frame(self, frame):
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            self.draw_specific_landmarks(frame, landmarks)
            
            left_knee_angle = self.calculate_angle(
                landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value],
                landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value],
                landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
            )
            
            right_knee_angle = self.calculate_angle(
                landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value],
                landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value],
                landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
            )
            
            avg_knee_angle = (left_knee_angle + right_knee_angle) / 2
            depth_category, color = self.get_squat_depth_category(avg_knee_angle)
            cv2.putText(frame, f'Knee Angle: {avg_knee_angle:.1f} degrees', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f'Depth: {depth_category}', 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
        return frame

def main():
    parser = argparse.ArgumentParser(description='Squat Depth Estimator')
    parser.add_argument('--video', help='Path to video file')
    parser.add_argument('--output', help='Output video path')
    parser.add_argument('--display', action='store_true', help='Display video while processing')
    
    args = parser.parse_args()
    estimator = SquatDepthEstimator()
    cap = cv2.VideoCapture(args.video)
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Processing video: {args.video}")
    print(f"Resolution: {width}x{height}, FPS: {fps}, Total frames: {total_frames}")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            processed_frame = estimator.process_frame(frame)
            if out:
                out.write(processed_frame)
            if args.display:
                cv2.imshow('Squat Depth Analysis', processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            frame_count += 1
            if frame_count % 30 == 0:  
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}%")
    finally:
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        print("\nProcessing completed.")

if __name__ == "__main__":
    main()
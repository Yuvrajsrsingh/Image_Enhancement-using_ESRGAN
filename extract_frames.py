



import cv2
import os

video_path = 'input_video/cc.mp4'
output_dir = 'extracted_frames'
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)  
frame_count = 0
saved_frame_count = 0

success, frame = cap.read()
while success:
    
    if frame_count % int(fps) == 0:
        frame_path = os.path.join(output_dir, f"frame_{saved_frame_count:04d}.jpg")
        cv2.imwrite(frame_path, frame)
        saved_frame_count += 1
    frame_count += 1
    success, frame = cap.read()

cap.release()
print(f"Extracted {saved_frame_count} frames, one per second.")
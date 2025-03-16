import cv2
import torch
import numpy as np
import pandas as pd
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from torchvision import models, transforms
from PIL import Image
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
model = YOLO("yolov8n.pt")  
tracker = DeepSort(max_age=50, nn_budget=100, max_iou_distance=0.7)

resnet50 = models.resnet50(weights="IMAGENET1K_V1")
resnet50.fc = torch.nn.Identity()  
resnet50.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

video_path = "sample1.mp4"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)  
frame_skip = 3  

sprint_threshold = 7.0 
min_sprint_duration = 3  

player_tracks = {}  
sprint_counts = {}  
ongoing_sprint = {}  
player_data = {}  

frame_id = 0 

def extract_features(image):
    """Extract deep features using ResNet50"""
    image = transform(Image.fromarray(image))
    image = image.unsqueeze(0)  
    with torch.no_grad():
        features = resnet50(image).cpu().numpy()
    return features

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1
    if frame_id % frame_skip != 0:
        continue  

    frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))

    results = model(frame)
    detections = []
    player_features = {}

    for result in results:
        for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
            x1, y1, x2, y2 = map(int, box[:4])
            if int(cls) == 0 and conf > 0.5:  
                cropped_img = frame[y1:y2, x1:x2]
                features = extract_features(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
                player_features[(x1, y1, x2, y2)] = features  
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf.item(), None))

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        bbox = track.to_ltrb()
        x1, y1, x2, y2 = map(int, bbox)
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

        if track_id not in player_data:
            player_data[track_id] = player_features.get((x1, y1, x2, y2), None)

        
        if track_id not in player_tracks:
            player_tracks[track_id] = []
        player_tracks[track_id].append((frame_id, center_x, center_y))

        speed = 0
        if len(player_tracks[track_id]) > 2:
            prev_frame, prev_x, prev_y = player_tracks[track_id][-2]
            time_diff = (frame_id - prev_frame) / fps  
            if time_diff > 0:
                speed = ((center_x - prev_x) ** 2 + (center_y - prev_y) ** 2) ** 0.5 / time_diff  

        if speed > sprint_threshold:
            ongoing_sprint[track_id] = ongoing_sprint.get(track_id, 0) + 1 
        else:
            if ongoing_sprint.get(track_id, 0) >= min_sprint_duration:
                sprint_counts[track_id] = sprint_counts.get(track_id, 0) + 1  
            ongoing_sprint[track_id] = 0  

        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {track_id} Sprints {sprint_counts.get(track_id, 0)}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

df = pd.DataFrame([(k, round(frame_id / (fps * 60), 2), sprint_counts.get(k, 0)) for k in player_tracks.keys()],
                  columns=["Player_ID", "Minutes_Played", "Sprint_Count"])
df.to_csv("player_sprint_data.csv", index=False)


print(f"Total players detected: {len(player_tracks)}")
print(f"Sprint counts per player: {sprint_counts}")
print("Data saved to player_sprint_data.csv")


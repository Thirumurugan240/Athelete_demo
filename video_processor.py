import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from torchvision import models, transforms
from PIL import Image
import torch

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

def extract_features(image):
    """Extract deep features using ResNet50"""
    image = transform(Image.fromarray(image))
    image = image.unsqueeze(0)
    with torch.no_grad():
        features = resnet50(image).cpu().numpy()
    return features

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0.0:
        fps = 30  


    global tracker
    tracker = DeepSort(max_age=50, nn_budget=100, max_iou_distance=0.7)
    player_stats = {}  
    ongoing_sprint = {}

    frame_id = 0
    sprint_threshold = 7.0
    min_sprint_duration = 3

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        if frame_id % 3 != 0:
            continue  

        results = model(frame)
        detections = []

        for result in results:
            for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                x1, y1, x2, y2 = map(int, box[:4])
                if int(cls) == 0 and conf > 0.5:  
                    detections.append(([x1, y1, x2 - x1, y2 - y1], conf.item(), None))

        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            bbox = track.to_ltrb()  
            x1, y1, x2, y2 = map(int, bbox)
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

            if track_id not in player_stats:
                player_stats[track_id] = {
                    "Player_ID": track_id,
                    "Minutes_Played": 0,
                    "Sprint_Count": 0,
                    "Fatigue_Index": 0,
                    "Sprint_Intensity": 0,
                    "Composite_Load_Score": 0,
                    "Injury_History": 0
                }

            player_stats[track_id]["Minutes_Played"] = round(frame_id / (fps * 60), 2)
            
            if len(player_stats[track_id].get("track_data", [])) > 2:
                prev_frame, prev_x, prev_y = player_stats[track_id]["track_data"][-2]
                time_diff = (frame_id - prev_frame) / fps
                if time_diff > 0:
                    speed = ((center_x - prev_x) ** 2 + (center_y - prev_y) ** 2) ** 0.5 / time_diff
                    if speed > sprint_threshold:
                        ongoing_sprint[track_id] = ongoing_sprint.get(track_id, 0) + 1
                    else:
                        if ongoing_sprint.get(track_id, 0) >= min_sprint_duration:
                            player_stats[track_id]["Sprint_Count"] += 1
                        ongoing_sprint[track_id] = 0  

            player_stats[track_id]["track_data"] = player_stats[track_id].get("track_data", []) + [(frame_id, center_x, center_y)]

    cap.release()

    for player_id, data in player_stats.items():
        data["Fatigue_Index"] = data["Sprint_Count"] * 0.2
        data["Sprint_Intensity"] = data["Sprint_Count"] / data["Minutes_Played"] if data["Minutes_Played"] > 0 else 0
        data["Composite_Load_Score"] = data["Sprint_Count"] * 1.5

    return list(player_stats.values())  

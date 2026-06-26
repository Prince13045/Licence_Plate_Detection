#FOR VIDEO DETECTION
import pandas as pd
import cv2
import numpy as np
from ultralytics import YOLO
import re
from collections import defaultdict, deque
import easyocr

# Initialize YOLO and EasyOCR
model = YOLO(r"pt_models\best.pt")
reader = easyocr.Reader(['en'], gpu=True)

plate_pattern = re.compile(r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$')

def correct_plate_format(ocr_text):
    mapping_num_to_text = {"0": "O", "1": "I", "5": "S", "8": "B"}
    mapping_text_to_num = {"O": "0", "I": "1", "S": "5", "Z": "2", "8": "B"}
    
    ocr_text = ocr_text.upper().replace(" ", "")
    if len(ocr_text) < 8 or len(ocr_text) > 10:
        return ""
        
    corrected = []
    for i, ch in enumerate(ocr_text):
        if i < 2 or i >= 4: 
            if ch.isdigit() and ch in mapping_num_to_text:
                corrected.append(mapping_num_to_text[ch])
            elif ch.isalpha():
                corrected.append(ch)
            else:
                return ""
        else: 
            if ch.isalpha() and ch in mapping_text_to_num:
                corrected.append(mapping_text_to_num[ch])
            elif ch.isdigit():
                corrected.append(ch)
            else:
                return ""  
                
    return "".join(corrected)  

# Preprocessing license plate before OCR
def recognize_plate(plate_crop):
    if plate_crop.size == 0:
        return ""
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    plate_resized = cv2.resize(threshold, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    try:
        ocr_results = reader.readtext(
            plate_resized, detail=0, 
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        )
        if len(ocr_results) > 0:
            candidate = correct_plate_format(ocr_results[0])
            if candidate and plate_pattern.match(candidate):
                return candidate
    except Exception as e:
        pass
    return ""

# Number plate stabilization buffer
plate_history = defaultdict(lambda: deque(maxlen=10))
plate_final = {}

def get_stable_plate(track_id, new_text):
    if new_text:
        plate_history[track_id].append(new_text)
        # Find the most frequent correct reading for this tracking ID
        most_common = max(set(plate_history[track_id]), key=plate_history[track_id].count)
        plate_final[track_id] = most_common  
    return plate_final.get(track_id, "")


# --- VIDEO FOR INFERENCING SETUP ---
input_video = r"C:\Liscenceplatedetection\data\inputvideo.mp4"
output_video = r"C:\Liscenceplatedetection\data\output1.avi"  

cap = cv2.VideoCapture(input_video)

# Get precise dimensions from the source video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Fallbacks if properties fail to read
if frame_width == 0 or frame_height == 0:
    frame_width, frame_height = 1920, 1080
if fps == 0:
    fps = 30.0

# Using XVID codec which is highly robust and built into OpenCV
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

CONF_THRESH = 0.3 

print("Processing frames... Please wait.")

# --- CAPTURING VIDEO ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    results = model.track(frame, persist=True, verbose=False)
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            conf = float(box.conf.cpu().numpy()[0]) if box.conf is not None else 0
            if conf < CONF_THRESH:
                continue
                
            track_id = int(box.id.cpu().numpy()[0]) if box.id is not None else 0
            x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])
            plate_crop = frame[y1:y2, x1:x2]
            
            text = recognize_plate(plate_crop)
            stable_text = get_stable_plate(track_id, text)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            if plate_crop.size > 0:
                overlay_w, overlay_h = 400, 150
                plate_resized = cv2.resize(plate_crop, (overlay_w, overlay_h))
            
                oy1 = max(0, y1 - overlay_h - 40)
                ox1 = x1
                oy2, ox2 = oy1 + overlay_h, ox1 + overlay_w
                
                if oy2 <= frame.shape[0] and ox2 <= frame.shape[1]:
                    frame[oy1:oy2, ox1:ox2] = plate_resized
                    
                    if stable_text:
                        cv2.putText(frame, stable_text, (ox1, oy1 - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 6)
                        cv2.putText(frame, stable_text, (ox1, oy1 - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    

    final_frame = cv2.resize(frame, (frame_width, frame_height))
    out.write(final_frame)
    
    print(".", end="", flush=True)

cap.release()
out.release()

print("\nSuccess! Annotated video saved as:", output_video)
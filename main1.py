#FOR PHOTO DETECTION
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

# Regex pattern for Indian standard plates
plate_pattern = re.compile(r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$')

def correct_plate_format(ocr_text):
    mapping_num_to_text = {"0": "O", "1": "I", "5": "S", "8": "B"}
    mapping_text_to_num = {"O": "0", "I": "1", "S": "5", "Z": "2", "8": "B"}
    
    ocr_text = ocr_text.upper().replace(" ", "")
    if len(ocr_text) < 8 or len(ocr_text) > 10:
        return ""
        
    corrected = []
    for i, ch in enumerate(ocr_text):
        if i < 2 or i >= 4:  # Letters
            if ch.isdigit() and ch in mapping_num_to_text:
                corrected.append(mapping_num_to_text[ch])
            elif ch.isalpha():
                corrected.append(ch)
            else:
                return ""
        else:  # Numbers
            if ch.isalpha() and ch in mapping_text_to_num:
                corrected.append(mapping_text_to_num[ch])
            elif ch.isdigit():
                corrected.append(ch)
            else:
                return ""  
                
    return "".join(corrected)

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

# --- PHOTO INFERENCE CONFIGURATION ---
input_image_path = r"C:\Liscenceplatedetection\data\carphoto.jpg"      
output_image_path = r"data\output_photo.jpg"     

# Load image
frame = cv2.imread(input_image_path)

if frame is None:
    print(f"Error: Could not load image from {input_image_path}. Please check the file path.")
else:
    CONF_THRESH = 0.3 
    
    # Run prediction (No tracking needed for single images)
    results = model(frame, verbose=False)
    
    for r in results:
        boxes = r.boxes
        for i, box in enumerate(boxes):
            conf = float(box.conf.cpu().numpy()[0]) if box.conf is not None else 0
            if conf < CONF_THRESH:
                continue
                
            x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])
            plate_crop = frame[y1:y2, x1:x2]
            
            # Since it's a static photo, the text read directly is our final stable text
            stable_text = recognize_plate(plate_crop)
            
            # Draw bounding box around the plate
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            if plate_crop.size > 0:
                overlay_w, overlay_h = 400, 150  # Width, Height
                plate_resized = cv2.resize(plate_crop, (overlay_w, overlay_h))
            
                # Calculate position for the visual crop popup overlay
                oy1 = max(0, y1 - overlay_h - 40)
                ox1 = x1
                oy2, ox2 = oy1 + overlay_h, ox1 + overlay_w
                
                if oy2 <= frame.shape[0] and ox2 <= frame.shape[1]:
                    frame[oy1:oy2, ox1:ox2] = plate_resized
                    
                    if stable_text:
                        # Draw black outline text
                        cv2.putText(frame, stable_text, (ox1, oy1 - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 6)
                        # Draw white inner text
                        cv2.putText(frame, stable_text, (ox1, oy1 - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

    # Save the processed image directly to your folder (Bypassing GUI)
    cv2.imwrite(output_image_path, frame)
    print(f"Success! Processed photo saved as: {output_image_path}")
import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import re
import easyocr
import os
import tempfile

# --- STREAMLIT CONFIG ---
st.set_page_config(page_title="AI Number Plate Recognition", layout="wide")
st.title("🚗 Automated Number Plate Recognition (ANPR)")
st.write("Upload an image or a video to detect and recognize license plates in real-time.")

# --- CACHE MODELS FOR SPEED ---
@st.cache_resource
def load_models():
    # Adjust path if your model is elsewhere
    model = YOLO(r"pt_models\best.pt")
    reader = easyocr.Reader(['en'], gpu=False) # Set to True if CUDA is working
    return model, reader

try:
    model, reader = load_models()
except Exception as e:
    st.error(f"Error loading models. Check if 'pt_models/best.pt' exists. Details: {e}")
    st.stop()

# Regex pattern for Indian standard plates
plate_pattern = re.compile(r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$')


def correct_plate_format(ocr_text):
    mapping_num_to_text = {"0": "O", "1": "I", "5": "S", "8": "B"}
    mapping_text_to_num = {"O": "0", "I": "1", "S": "5", "Z": "2", "8": "B"}
    
    ocr_text = ocr_text.upper().replace(" ", "")
    length = len(ocr_text)
    
    if length < 8 or length > 10:
        return ""
        
    corrected = []
    for i, ch in enumerate(ocr_text):
        # Determine if this specific position should be a Letter or a Number
        # Standard Indian format: AA NN AA NNNN or AA NN AAA NNNN
        is_letter_position = False
        
        if i < 2:  # First two characters are always state codes (Letters)
            is_letter_position = True
        elif length == 10: # e.g., MH 12 EG 8397
            if i == 4 or i == 5:
                is_letter_position = True
        elif length == 9:  # e.g., DL 3C CE 1234 (single digit district) or MH 12 G 8397
            if i == 3 or i == 4:
                is_letter_position = True
        elif length == 8:  # e.g., MH 12 A 1234
            if i == 3:
                is_letter_position = True

        # Apply corrections based on what the character is supposed to be
        if is_letter_position:
            if ch.isdigit() and ch in mapping_num_to_text:
                corrected.append(mapping_num_to_text[ch])
            elif ch.isalpha():
                corrected.append(ch)
            else:
                return ""
        else:  # It's supposed to be a number
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
    except:
        pass
    return ""

# --- APP NAVIGATION ---
mode = st.sidebar.selectbox("Choose Mode", ["Photo Processing", "Video Processing"])
CONF_THRESH = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.3, 0.05)

# ==========================================
# MODE 1: PHOTO PROCESSING
# ==========================================
if mode == "Photo Processing":
    st.header("📸 Photo ANPR Pipeline")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Read image file
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)
        
        # UI Columns
        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Original Image", use_container_width=True)
            
        with st.spinner("Analyzing Image..."):
            results = model(frame, verbose=False)
            detected_plates = []
            
            for r in results:
                for box in r.boxes:
                    conf = float(box.conf.cpu().numpy()[0]) if box.conf is not None else 0
                    if conf < CONF_THRESH:
                        continue
                        
                    x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])
                    plate_crop = frame[y1:y2, x1:x2]
                    stable_text = recognize_plate(plate_crop)
                    
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
                                cv2.putText(frame, stable_text, (ox1, oy1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 6)
                                cv2.putText(frame, stable_text, (ox1, oy1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
                    
                    if stable_text:
                        detected_plates.append(stable_text)

        with col2:
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Processed Image", use_container_width=True)
            if detected_plates:
                st.success(f"**Detected Plates:** {', '.join(detected_plates)}")
            else:
                st.warning("No valid formatted plates found matching the pattern.")

# ==========================================
# MODE 2: VIDEO PROCESSING
# ==========================================
elif mode == "Video Processing":
    st.header("🎥 Video ANPR Pipeline")
    uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
    
    if uploaded_video is not None:
        # Save uploaded video to a temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        
        cap = cv2.VideoCapture(tfile.name)
        
        # Display placeholders
        frame_placeholder = st.empty()
        results_placeholder = st.sidebar.empty()
        
        st.info("Processing stream... Tracks are updated frame-by-frame below.")
        
        # Use an ephemeral dictionary for localized session stabilization
        session_history = {}
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Run tracker
            results = model.track(frame, persist=True, verbose=False)
            
            for r in results:
                for box in r.boxes:
                    conf = float(box.conf.cpu().numpy()[0]) if box.conf is not None else 0
                    if conf < CONF_THRESH:
                        continue
                        
                    track_id = int(box.id.cpu().numpy()[0]) if box.id is not None else 0
                    x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])
                    plate_crop = frame[y1:y2, x1:x2]
                    
                    text = recognize_plate(plate_crop)
                    if text:
                        session_history[track_id] = text
                    
                    stable_text = session_history.get(track_id, "")
                    
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
                                cv2.putText(frame, stable_text, (ox1, oy1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 5)
                                cv2.putText(frame, stable_text, (ox1, oy1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            
            # Stream directly to browser window frame placeholder
            frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            
            # Update live sidebar dashboard values 
            if session_history:
                results_placeholder.write("### Live Detected Log\n" + "\n".join([f"- **ID {k}**: `{v}`" for k, v in session_history.items()]))
                
        cap.release()
        os.unlink(tfile.name) # Clean temporary file path
        st.success("Video processing stream reached its end!")
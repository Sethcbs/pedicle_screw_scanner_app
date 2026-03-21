from google.cloud import vision
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from ultralytics import YOLO
import io
import cv2
import numpy as np
import base64
import sqlite3
from datetime import datetime 
import torch

torch.set_num_threads(1)

app = Flask(__name__)
CORS(app)

model = YOLO('v3-best.pt') 

#define a simple sqlite3 database for current testing and MVP
def init_db():
    conn = sqlite3.connect('inventory.db')
    c = conn.cursor()
    # create the table if it doesn't exist yet
    c.execute('''
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            brand TEXT,
            system TEXT,
            diameter TEXT,
            features TEXT
        )
    ''')
    conn.commit()
    conn.close()

# The logic of our program that determines company based on 
# colors and sections of the screw returned from the ML model
def apply_screw_logic(detected_list):
    detected = set(detected_list) 
    raw_output = ", ".join(detected_list)

    if "head_green" in detected:
        return {"brand": "OIC", "system": "Standard", "diameter": "5.5mm", "feature": raw_output}
    
    if "head_blue" in detected and "shaft_lightblue" in detected:
            return {"brand": "Orthomed", "system": "Modular", "diameter": "6.5mm", "feature": raw_output}
    if "head_darkblue" in detected:
        if "setscrew_gold" in detected:
            return {"brand": "Orthomed", "system": "Standard (Non-Modular)", "diameter": "6.5mm", "feature": raw_output}
        elif "shaft_grey" in detected:
            return {"brand": "Orthomed", "system": "Standard (Non-Modular)", "diameter": "6.5mm", "feature": raw_output}
        elif "shaft_lightblue" in detected:
            return {"brand": "Orthomed", "system": "Modular", "diameter": "6.5mm", "feature": raw_output}
            
    if "shaft_purple" in detected:
        if "head_silver" in detected:
            return {"brand": "NuVasive", "system": "Non-Modular", "diameter": "Purple Sizing", "feature": raw_output}
        elif "head_grey" in detected:
            return {"brand": "NuVasive", "system": "Modular", "diameter": "Purple Sizing", "feature": raw_output}
        return {"brand": "NuVasive", "system": "Unknown", "diameter": "Purple Sizing", "feature": raw_output}

    if "head_grey" in detected:
        if "shaft_blue" in detected:
            return {"brand": "Depuy", "system": "Standard", "diameter": "Varies", "feature": raw_output}
        elif "shaft_silver" in detected or "shaft_grey" in detected:
            return {"brand": "Mindray", "system": "Standard", "diameter": "6.5mm", "feature": raw_output}
        elif "shaft_purple" in detected:
            return {"brand": "NuVasive", "system": "Modular", "diameter": "Purple Sizing", "feature": raw_output}
            
    return {"brand": "Unknown Hardware", "system": "No Logic Match", "diameter": "--", "feature": raw_output}

# make sure our API is running as intended
@app.route('/')
def health_check():
    return jsonify({"status": "PediScan API is live and running!"})

def extract_text_from_image(cv2_image):
    success, encoded_image = cv2.imencode('.jpg', cv2_image)
    content = encoded_image.tobytes()

    # initialize Google Vision
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations

    if response.error.message:
        print(f"Google Vision Error: {response.error.message}")
        return ""

    # first item in the list is the entire block of detected text
    if texts:
        return texts[0].description.replace('\n', ' ').strip()

    return ""

@app.route('/scan', methods=['POST'])
def scan_image():
    try:
        data = request.json
        base64_string = data['image'].split(',')[1] 
        
        img_data = base64.b64decode(base64_string)
        np_arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        h, w = img.shape[:2]
        if max(h, w) > 640:
            scale = 640 / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))

        results = model(img, conf=0.25) 
        
        img_h, img_w, _ = img.shape
        
        detected_classes = []
        cropped_img = None
        
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            detected_classes.append(class_name)
            
            if class_name in ["tulip_head", "setscrew"]:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                margin = 15
                
                crop_y1 = max(0, y1 - margin)
                crop_y2 = min(img_h, y2 + margin)
                crop_x1 = max(0, x1 - margin)
                crop_x2 = min(img_w, x2 + margin)
                
                cropped_img = img[crop_y1:crop_y2, crop_x1:crop_x2]
                
        detected_text = ""
        if cropped_img is not None:
            detected_text = extract_text_from_image(cropped_img)
            print(f"OCR Found on Crop: {detected_text}")
        else:
            print("No valid hardware found to crop.")

        brand = "Unknown"
        system = "Unknown"

        if "161/111 JP23143" in detected_text.lower():
            brand = "Nuvasive"
            system = "Modular"

        else:
            fallback = apply_screw_logic(detected_classes)
            brand = fallback["brand"]
            system = fallback["system"]

        annotated_img = results[0].plot()
        _, buffer = cv2.imencode('.jpg', annotated_img)
        annotated_base64 = base64.b64encode(buffer).decode('utf-8')
        annotated_data_url = f"data:image/jpeg;base64,{annotated_base64}"

        raw_output = ", ".join(detected_classes)
        final_result = {
            "brand": brand,
            "system": system,
            "diameter": detected_text if detected_text else "No text read",
            "feature": raw_output
        }

        return jsonify({
            'result': final_result,
            'annotated_image': annotated_data_url
        })

    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({'error': str(e)}), 500

#prompt the user to confirm that this should be saved to the database
@app.route('/save', methods=['POST'])
def save_scan():
    try:
        init_db()

        data = request.json # the hardware details sent from the phone
        conn = sqlite3.connect('inventory.db')
        c = conn.cursor()
        timestamp = datetime.now().strftime("%Y-%m-%d %I:%M %p")
        
        c.execute("INSERT INTO scans (timestamp, brand, system, diameter, features) VALUES (?, ?, ?, ?, ?)",
                  (timestamp, data['brand'], data['system'], data['diameter'], data['feature']))
        conn.commit()
        conn.close()
        
        print(f" User Confirmed & Saved to DB: {data['brand']}")
        return jsonify({'status': 'success'})
        
    except Exception as e:
        print(f"Database error: {e}")
        return jsonify({'error': str(e)}), 500

# get the history of saved queries
@app.route('/history', methods=['GET'])
def get_history():
    try:
        init_db()

        conn = sqlite3.connect('inventory.db')
        c = conn.cursor()
        # grab the 50 most recent scans
        c.execute("SELECT timestamp, brand, system, diameter, features FROM scans ORDER BY id DESC LIMIT 50")
        rows = c.fetchall()
        conn.close()
        
        history_list = []
        for row in rows:
            history_list.append({
                "timestamp": row[0],
                "brand": row[1],
                "system": row[2],
                "diameter": row[3],
                "features": row[4]
            })
        return jsonify(history_list)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # build the database file before starting the server
    init_db() 
    app.run(host='0.0.0.0', port=8080, debug=True)

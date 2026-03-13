from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import sqlite3
from datetime import datetime 

app = Flask(__name__)
CORS(app)

model = YOLO('best.pt') 

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

# The logic of our program that determines company based on colors returned from the model
def apply_screw_logic(detected_list):
    detected = set(detected_list) 
    raw_output = ", ".join(detected_list)

    if "head_green" in detected:
        return {"brand": "OIC", "system": "Standard", "diameter": "5.5mm", "feature": raw_output}
    
    if "head_blue" in detected or "head_darkblue" in detected:
        if "setscrew_gold" in detected:
            return {"brand": "Orthomed", "system": "Standard (Non-Modular)", "diameter": "6.5mm", "feature": raw_output}
        elif "setscrew_star_blue" in detected:
            return {"brand": "Orthomed", "system": "Modular", "diameter": "6.5mm", "feature": raw_output}
        elif "shaft_grey" in detected:
            return {"brand": "Orthomed", "system": "Standard (Non-Modular)", "diameter": "6.5mm", "feature": raw_output}
        elif "shaft_lightblue" in detected or "shaft_silver" in detected or "shaft_blue" in detected:
            return {"brand": "Orthomed", "system": "Modular", "diameter": "6.5mm", "feature": raw_output}
            
    if "head_silver" in detected and "shaft_purple" in detected:
        return {"brand": "NuVasive", "system": "Non-Modular", "diameter": "Purple Sizing", "feature": raw_output}
        
    if "head_grey" in detected:
        if "shaft_purple" in detected and "neck_seam" in detected:
            return {"brand": "NuVasive", "system": "Modular", "diameter": "Purple Sizing", "feature": raw_output}
        elif "shaft_blue" in detected or ("shaft_purple" in detected and "setscrew_silver" in detected):
            return {"brand": "Depuy", "system": "Standard", "diameter": "Varies", "feature": raw_output}
        elif "shaft_silver" in detected or "shaft_grey" in detected:
            return {"brand": "Mindray", "system": "Standard", "diameter": "6.5mm", "feature": raw_output}
            
    return {"brand": "Unknown Hardware", "system": "No Logic Match", "diameter": "--", "feature": raw_output}

# send our front end file
@app.route('/')
def serve_frontend():
    return send_file('index.html')


@app.route('/scan', methods=['POST'])
def scan_image():
    try:
        data = request.json
        base64_string = data['image'].split(',')[1] 
        
        img_data = base64.b64decode(base64_string)
        np_arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        results = model(img, conf=0.25) 
        
        annotated_img = results[0].plot()
        
        _, buffer = cv2.imencode('.jpg', annotated_img)
        annotated_base64 = base64.b64encode(buffer).decode('utf-8')
        annotated_data_url = f"data:image/jpeg;base64,{annotated_base64}"

        names_dict = model.names
        detected_classes = []
        for r in results:
            for c in r.boxes.cls:
                detected_classes.append(names_dict[int(c)])

        final_result = apply_screw_logic(detected_classes)

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
        data = request.json # the hardware details sent from the phone
        
        conn = sqlite3.connect('inventory.db')
        c = conn.cursor()
        timestamp = datetime.now().strftime("%Y-%m-%d %I:%M %p")
        
        c.execute("INSERT INTO scans (timestamp, brand, system, diameter, features) VALUES (?, ?, ?, ?, ?)",
                  (timestamp, data['brand'], data['system'], data['diameter'], data['feature']))
        conn.commit()
        conn.close()
        
        print(f"✅ User Confirmed & Saved to DB: {data['brand']}")
        return jsonify({'status': 'success'})
        
    except Exception as e:
        print(f"Database error: {e}")
        return jsonify({'error': str(e)}), 500

# get the history of saved queries
@app.route('/history', methods=['GET'])
def get_history():
    try:
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

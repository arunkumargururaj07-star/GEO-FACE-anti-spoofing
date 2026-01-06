from flask import Flask, render_template_string, jsonify, request, send_file
import cv2
import numpy as np
import os
import csv
import base64
import math
import time
from datetime import datetime

app = Flask(__name__)

# ------------------- CONFIGURATION -------------------
KNOWN_DIR = "known_faces"
ATT_FILE = "attendance.csv"
UIDS_FILE = "uids.csv"

# 📍 LOCATION SETTINGS (Example: Gogte Institute)
# Change these to your actual classroom coordinates for the demo!
CLASS_LAT = 15.696969
CLASS_LON = 74.696060
ALLOWED_RADIUS = 50  # Meters

# ------------------- SETUP -------------------
# Load Face Detection Model (Fast)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load Face Recognition Model (Accurate)
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
    print("✅ DeepFace AI Loaded!")
except ImportError:
    DEEPFACE_AVAILABLE = False
    print("❌ DeepFace not found. Please install it.")

known_faces = {}
uid_map = {}

# ------------------- HELPER FUNCTIONS -------------------
def load_data():
    """Loads student images and USNs into memory on startup."""
    global known_faces, uid_map
    known_faces.clear()
    uid_map.clear()

    os.makedirs(KNOWN_DIR, exist_ok=True)
    
    # Create files if they don't exist
    if not os.path.exists(UIDS_FILE):
        with open(UIDS_FILE, "w") as f: f.write("Name,USN\n")
    if not os.path.exists(ATT_FILE):
        with open(ATT_FILE, "w") as f: f.write("Name,USN,Date,Time\n")

    # Load USNs (Name -> USN)
    print("📂 Loading USNs...")
    with open(UIDS_FILE, "r") as f:
        reader = csv.reader(f)
        next(reader, None) # Skip header
        for row in reader:
            if len(row) >= 2:
                uid_map[row[0].lower()] = row[1].upper()

    # Load Images (Name -> [List of Image Paths])
    print("📂 Loading Photos...")
    for filename in os.listdir(KNOWN_DIR):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            name = filename.split('.')[0].split('_')[0].lower()
            if name not in known_faces: known_faces[name] = []
            known_faces[name].append(os.path.join(KNOWN_DIR, filename))

    print(f"🚀 System Ready: {len(known_faces)} Students Loaded")

def calculate_distance(lat1, lon1, lat2, lon2):
    """Haversine Formula to calculate distance between two GPS points in meters."""
    R = 6371000 # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def mark_attendance(name):
    """Writes the attendance to the CSV file if not already marked today."""
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    uid = uid_map.get(name, "Unknown")

    # Check if already marked
    if os.path.exists(ATT_FILE):
        with open(ATT_FILE, "r") as f:
            if any(name in line and date_str in line for line in f):
                return False, "Already Marked"

    # Append to file
    with open(ATT_FILE, "a", newline="") as f:
        f.write(f"{name},{uid},{date_str},{time_str}\n")
    print(f"📝 SAVED: {name} ({uid})")
    return True, "Marked"

# ------------------- CORE LOGIC (AI + GPS) -------------------
def process_frame_logic(data):
    try:
        image_data = data.get('image')
        user_lat = data.get('lat')
        user_lon = data.get('lon')

        if user_lat is None: return {"status": "error", "message": "📡 Waiting for GPS..."}

        # 1. Geofence Check
        dist = calculate_distance(user_lat, user_lon, CLASS_LAT, CLASS_LON)
        if dist > ALLOWED_RADIUS:
            return {"status": "error", "message": f"🚫 You are {int(dist)}m away! (Max {ALLOWED_RADIUS}m)", "geofence_fail": True}

        # 2. Decode Image (Base64 -> OpenCV)
        if ',' in image_data:
            header, encoded = image_data.split(",", 1)
        else:
            encoded = image_data
        nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 3. Detect Face
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))

        if len(faces) == 0:
            return {"status": "success", "total_faces": 0, "message": "No Face Detected"}

        # 4. Recognize Face (DeepFace)
        recognized_name = "Unknown"
        if DEEPFACE_AVAILABLE:
            best_score = 0.50 # Threshold (Lower is stricter)
            for name, image_paths in known_faces.items():
                for img_path in image_paths:
                    try:
                        # Compare live frame with stored image
                        res = DeepFace.verify(frame, img_path, model_name="VGG-Face", enforce_detection=False, align=True)
                        if res['distance'] < best_score:
                            best_score = res['distance']
                            recognized_name = name
                            break
                    except:
                        pass
                if recognized_name != "Unknown": break

        # 5. Mark Attendance
        attendance_marked = False
        uid_display = ""
        if recognized_name != "Unknown":
            success, _ = mark_attendance(recognized_name)
            attendance_marked = success
            uid_display = uid_map.get(recognized_name, "PENDING")

        return {
            "status": "success",
            "name": recognized_name,
            "uid": uid_display,
            "attendance_marked": attendance_marked,
            "total_faces": 1
        }
    except Exception as e:
        print("Error:", e)
        return {"status": "error", "message": "Server Error"}

# ------------------- ROUTE: REGISTRATION (SAVE PHOTO) -------------------
@app.route('/save_new_student', methods=['POST'])
def save_new_student():
    try:
        data = request.json
        name = data.get('name').strip().lower()
        usn = data.get('usn', 'PENDING').strip().upper()
        image_data = data.get('image')

        # Decode Image
        if ',' in image_data: header, encoded = image_data.split(",", 1)
        else: encoded = image_data

        # Save File
        filename = f"{name}_{int(time.time())}.jpg"
        file_path = os.path.join(KNOWN_DIR, filename)
        with open(file_path, "wb") as fh:
            fh.write(base64.b64decode(encoded))

        # Update Memory
        if name not in known_faces: known_faces[name] = []
        known_faces[name].append(file_path)
        uid_map[name] = usn

        # Save to CSV
        with open(UIDS_FILE, "a", newline="") as f:
            f.write(f"{name},{usn}\n")

        return jsonify({"status": "success", "message": f"Saved {name.upper()}"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

# ------------------- ROUTE: MAIN ATTENDANCE PAGE -------------------
@app.route('/')
def index():
    return render_template_string('''
<!DOCTYPE html>
<html>
<head>
    <title>Geo-Face Attendance</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body { font-family: 'Segoe UI', sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); text-align: center; padding: 20px; color: #333; min-height: 100vh;}
        .box { background: rgba(255, 255, 255, 0.95); padding: 20px; border-radius: 20px; box-shadow: 0 8px 32px rgba(0,0,0,0.1); max-width: 500px; margin: auto; }
        h2 { color: #5b4bc4; }
        video { width: 100%; border-radius: 10px; transform: scaleX(-1); background: #000; }
        .status { padding: 15px; margin-top: 15px; border-radius: 12px; font-weight: bold; }
        .success { background: #d4edda; color: #155724; }
        .error { background: #f8d7da; color: #721c24; }
        .info { background: #e2e3e5; color: #383d41; }
        .btn { padding: 15px; width: 100%; border: none; border-radius: 10px; font-size: 16px; margin-top: 10px; cursor: pointer; color: white; font-weight: bold; }
        .btn-start { background: #667eea; } .btn-reg { background: #84fab0; color: #005c4b; } .btn-down { background: #f093fb; }
    </style>
</head>
<body>
    <div class="box">
        <h2>📍 Face Attendance</h2>
        <video id="video" autoplay playsinline></video>
        <canvas id="canvas" style="display:none;"></canvas>
        <div id="status" class="status info">Waiting for GPS...</div>
        <button class="btn btn-start" onclick="startSystem()">📡 Start Camera</button>
        <a href="/register"><button class="btn btn-reg">➕ Register Student</button></a>
        <a href="/download_csv"><button class="btn btn-down">📥 Download Report</button></a>
    </div>

    <script>
        const video = document.getElementById('video');
        const statusDiv = document.getElementById('status');
        const canvas = document.getElementById('canvas');
        let currentLat = null, currentLon = null;
        let isLooping = false;

        function startSystem() {
            if (!navigator.geolocation) return statusDiv.innerText = "❌ No GPS Support";
            statusDiv.innerText = "🛰️ Locating...";
            
            // Get Continuous GPS Updates
            navigator.geolocation.watchPosition(
                (pos) => {
                    currentLat = pos.coords.latitude; currentLon = pos.coords.longitude;
                    if(!video.srcObject) startCamera();
                },
                (err) => statusDiv.innerText = "❌ Allow GPS Access!", 
                { enableHighAccuracy: true }
            );
        }

        async function startCamera() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' } });
                video.srcObject = stream;
                statusDiv.innerText = "🚀 Ready! Scanning...";
                if(!isLooping) { isLooping = true; sendFrameLoop(); }
            } catch(e) { statusDiv.innerText = "Camera Permission Denied"; }
        }

        function sendFrameLoop() {
            if (currentLat === null) { setTimeout(sendFrameLoop, 1000); return; }

            // Capture Frame
            canvas.width = 240; canvas.height = 180;
            canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);

            // Send to Python
            fetch('/process_frame', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: canvas.toDataURL('image/jpeg', 0.5), lat: currentLat, lon: currentLon })
            })
            .then(res => res.json())
            .then(data => {
                if(data.status === 'success' && data.name !== "Unknown") {
                    statusDiv.innerHTML = `✅ Attendance Marked<br>${data.name.toUpperCase()} (${data.uid})`;
                    statusDiv.className = "status success";
                } else if(data.status === 'error') {
                    statusDiv.innerText = data.message;
                    statusDiv.className = "status error";
                } else {
                    statusDiv.innerText = "📷 Scanning...";
                    statusDiv.className = "status info";
                }
                setTimeout(sendFrameLoop, 200); // Loop every 200ms
            })
            .catch(e => { setTimeout(sendFrameLoop, 1000); });
        }
    </script>
</body>
</html>
    ''')

# ------------------- ROUTE: REGISTER PAGE -------------------
@app.route('/register')
def register_page():
    return render_template_string('''
<!DOCTYPE html>
<html>
<head><title>Register</title><meta name="viewport" content="width=device-width, initial-scale=1.0"><style>body{font-family:'Segoe UI';background:#333;text-align:center;padding:20px;color:white;}input,button{width:100%;padding:10px;margin:5px 0;border-radius:5px;}video{width:100%;border-radius:10px;transform:scaleX(-1);}</style></head>
<body>
    <h2>➕ Register New Student</h2>
    <video id="video" autoplay playsinline></video><canvas id="canvas" style="display:none;"></canvas>
    <input id="name" placeholder="Name (e.g. Arun)"><input id="usn" placeholder="USN">
    <button onclick="save()" style="background:#84fab0;font-weight:bold;">📸 Save Face</button>
    <a href="/"><button style="background:#555;color:white;">Back to Home</button></a>
    <script>
        const v=document.getElementById('video'), c=document.getElementById('canvas');
        navigator.mediaDevices.getUserMedia({video:{facingMode:'user'}}).then(s=>v.srcObject=s);
        function save(){
            const n = document.getElementById('name').value;
            const u = document.getElementById('usn').value;
            if(!n || !u) return alert("Enter Name and USN!");
            c.width=320;c.height=240;c.getContext('2d').drawImage(v,0,0,320,240);
            fetch('/save_new_student',{method:'POST',headers:{'Content-Type':'application/json'},
            body:JSON.stringify({image:c.toDataURL('image/jpeg',0.8),name:n,usn:u})})
            .then(r=>r.json()).then(d=>alert(d.message));
        }
    </script>
</body></html>''')

# ------------------- ROUTE: PROCESSING API -------------------
@app.route('/process_frame', methods=['POST'])
def process(): 
    return jsonify(process_frame_logic(request.get_json()))

# ------------------- ROUTE: DOWNLOAD CSV -------------------
@app.route('/download_csv')
def download_csv(): 
    return send_file(ATT_FILE, as_attachment=True, download_name="attendance.csv") if os.path.exists(ATT_FILE) else "No records"

if __name__ == '__main__':
    load_data()
    print("🚀 SERVER STARTED on http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000)
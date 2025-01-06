from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import time

app = Flask(__name__)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
print("UPLOAD_FOLDER: ", UPLOAD_FOLDER)

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Erstellt das Upload-Verzeichnis, falls es nicht existiert
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Variablen
button_status = (-1, -1) # das Label
button_set = False # wurde für diese Segmente schon ein Button gedrückt (=> kann neues Label von labeling.py abgefragt werden)?
video_paths = [] # aktuelle Videos

print(f"Video-Dateien in uploads: {os.listdir(app.config['UPLOAD_FOLDER'])}", flush=True)

@app.route('/')
def index():
    print("index")
    return render_template('index.html')

# wenn Button gedrückt
@app.route('/button_action', methods=['POST'])
def button_action():
    print("button_action", flush=True)
    global button_status
    global button_set
    action = request.json.get('action')
    # Label erstellen
    if action == 'left':
        button_status = (1, 0)
        button_set = True # Button wurde gedrückt
        print("links gedrückt und gespeichert", flush=True)
    elif action == 'right':
        button_status = (0, 1)
        button_set = True
    elif action == 'equal':
        button_status = (0.5,0.5)
        button_set = True
    elif action == 'none':
        button_status = (0,0)
        button_set = True
    elif action == 'start': # Knopf, damit die ersten zwei Videos geladen werden, da dafür noch kein Labeling-Button gedrückt wird (kann man bestimmt noch eleganter machen: erste Videos automatisch geladen oder start-Button verschwindet nach anfänglichem Drücken)
        button_status = button_status
        button_set = button_set
    else:
        return jsonify({'error': 'Ungültige Aktion'}), 400

    print("Aktueller Button-Set: ", button_set, flush=True)

    # neue Videos laden
    video_files = os.listdir(app.config['UPLOAD_FOLDER']) # hier gibt es ein Problem: in der Liste video_files (und deswegen auch in videos und video_paths) sind immer nur die ersten zwei Videos (segment 0 und 1), obwohl im Ordner schon die neuen sind, wie kann das sein?
    print('Video_files: ', video_files, flush=True)
    videos = [f for f in video_files if f.endswith('.mp4')]
    print(videos, flush=True)
    global video_paths
    video_paths.clear()

    # sind schon zwei neue Videos in Ordner? Wenn nicht, darauf warten
    while (len(videos) <= 1):
        time.sleep(2)

    # aktuelle Videos
    video_paths.append(f"/uploads/{videos[0]}")
    video_paths.append(f"/uploads/{videos[1]}")
    print("videos in video_paths", flush=True)
    print(video_paths, flush=True)

    print('in button_action: ', button_set, flush=True)

    return jsonify({'status': button_status, 'set': button_set, 'videos': video_paths})

# Videos an frontend senden
@app.route('/get-videos', methods=['GET'])
def get_videos():
    print("get_videos", flush=True)
    global video_paths
    
    # Wenn `video_paths` leer ist, lade Videos aus dem Upload-Ordner
    if not video_paths:
        video_files = os.listdir(app.config['UPLOAD_FOLDER'])
        videos = [f"/uploads/{f}" for f in video_files if f.endswith('.mp4')]
        video_paths.extend(videos)
        print("Video-Pfade aus Upload-Ordner geladen:", video_paths, flush=True)
    
    print("Video-Pfade, die an das Template gesendet werden:", video_paths, flush=True)
    return jsonify({'videos': video_paths})

# Label an labeling.py
@app.route('/status', methods=['GET'])
def get_status():
    print("status", flush=True)
    global button_status
    print('von app: ', button_status, flush=True)
    return jsonify({'status': button_status})

# Wurde Button gedrückt an labeling.py
@app.route('/set', methods=['GET'])
def get_set():
    print("set (GET)", flush=True)
    global button_set
    print('von app: ', button_set, flush=True)
    return jsonify({'set': button_set})

# von labeling.py, button_set nach Verarbeitung der Label wieder auf False setzen
@app.route('/set', methods=['POST'])
def set_set():
    print("set (POST)", flush=True)
    global button_set
    data = request.json  # Erwartet JSON-Daten
    if "new_value" in data:
        button_set = data["new_value"]
        return jsonify({"message": "Variable aktualisiert!", "variable": button_set})
    else:
        return jsonify({"error": "Kein neuer Wert übergeben!"}), 400

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    print("uploaded_file", flush=True)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # Sende die Datei mit Cache-Control Header für kein Caching
    response = send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    response.cache_control.no_cache = True
    response.cache_control.no_store = True
    response.cache_control.must_revalidate = True
    return response

# Flask Thread starten
def start_flask():
    """Startet die Flask-App."""
    app.run(debug=True, use_reloader=False)

if __name__ == '__main__':
    start_flask()

from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import time

app = Flask(__name__)

# Verzeichnis für hochgeladene Dateien
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Erstellt das Upload-Verzeichnis, falls es nicht existiert
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Variablen
button_status = (-1, -1) # das Label
button_set = False # wurde für diese Segmente schon ein Button gedrückt (=> kann neues Label von labeling.py abgefragt werden)?
video_paths = [] # aktuelle Videos

@app.route('/')
def index():
    print("index")
    return render_template('index.html')

# wenn Button gedrückt
@app.route('/button_action', methods=['POST'])
def button_action():
    print("button_action")
    global button_status
    global button_set
    action = request.json.get('action')
    # Label erstellen
    if action == 'left':
        button_status = (1, 0)
        button_set = True # Button wurde gedrückt
        print("links gedrückt und gespeichert")
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

    # neue Videos laden
    video_files = os.listdir(app.config['UPLOAD_FOLDER']) # hier gibt es ein Problem: in der Liste video_files (und deswegen auch in videos und video_paths) sind immer nur die ersten zwei Videos (segment 0 und 1), obwohl im Ordner schon die neuen sind, wie kann das sein?
    print('Video_files: ', video_files)
    videos = [f for f in video_files if f.endswith('.mp4')]
    print(videos)
    global video_paths
    video_paths.clear()

    # sind schon zwei neue Videos in Ordner? Wenn nicht, darauf warten
    while (len(videos) <= 1):
        time.sleep(2)

    # aktuelle Videos
    video_paths.append(f"/{app.config['UPLOAD_FOLDER']}/{videos[0]}")
    video_paths.append(f"/{app.config['UPLOAD_FOLDER']}/{videos[1]}")
    print("videos in video_paths")
    print(video_paths)

    print('in button_action: ', button_set)

    return jsonify({'status': button_status, 'set': button_set, 'videos': video_paths})

# Videos an frontend senden
@app.route('/get-videos', methods=['GET'])
def get_videos():
    print("get_videos")
    global video_paths
    print("Video-Pfade, die an das Template gesendet werden:", video_paths)
    return jsonify({'videos': video_paths})

# Label an labeling.py
@app.route('/status', methods=['GET'])
def get_status():
    print("status")
    global button_status
    print('von app: ', button_status)
    return jsonify({'status': button_status})

# Wurde Button gedrückt an labeling.py
@app.route('/set', methods=['GET'])
def get_set():
    print("set (GET)")
    global button_set
    print('von app: ', button_set)
    return jsonify({'set': button_set})

# von labeling.py, button_set nach Verarbeitung der Label wieder auf False setzen
@app.route('/set', methods=['POST'])
def set_set():
    print("set (POST)")
    global button_set
    data = request.json  # Erwartet JSON-Daten
    if "new_value" in data:
        button_set = data["new_value"]
        return jsonify({"message": "Variable aktualisiert!", "variable": button_set})
    else:
        return jsonify({"error": "Kein neuer Wert übergeben!"}), 400

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    print("uploaded_file")
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, cache_timeout=0)

# Flask Thread starten
def start_flask():
    """Startet die Flask-App."""
    app.run(debug=False, use_reloader=False)

if __name__ == '__main__':
    start_flask()

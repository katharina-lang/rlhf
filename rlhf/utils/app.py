from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import time
import threading
import signal
import sys
import socket
from rlhf.configs.arguments import Args

app_should_stop = False
flask_port = None
preferences_labeled = 0

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

@app.route('/stop', methods=['POST'])
def stop_app():
    """Beendet die Flask-App."""
    global app_should_stop
    app_should_stop = True
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Server kann nicht gestoppt werden.')
    func()
    return jsonify({"message": "App wird beendet."})

# Überwachung in einem separaten Thread
def monitor_app():
    global app_should_stop, preferences_labeled
    while not app_should_stop:
        if preferences_labeled >= Args.amount_preferences:
            print("\nErfolgreich alle Videos gelabelt! Der Status wurde aktualisiert.", flush=True)
            # Wartezeit, damit das Frontend die Erfolgsmeldung anzeigen kann
            time.sleep(5)
            
            # Beende den Server
            os.kill(os.getpid(), signal.SIGINT)
            break  # Beende die Schleife, sobald das Labeln abgeschlossen ist
        time.sleep(1)


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
    global preferences_labeled
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
    else:
        return jsonify({'error': 'Ungültige Aktion'}), 400
    if action in ['left', 'right', 'equal', 'none']:
        preferences_labeled += 1  # Präferenz wurde erfolgreich gelabelt
        button_set = True
        print(f"Gelabelte Präferenzen: {preferences_labeled}/{Args.amount_preferences}", flush=True)
    print("Aktueller Button-Set: ", button_set, flush=True)

    # neue Videos laden
    video_files = os.listdir(app.config['UPLOAD_FOLDER']) # hier gibt es ein Problem: in der Liste video_files (und deswegen auch in videos und video_paths) sind immer nur die ersten zwei Videos (segment 0 und 1), obwohl im Ordner schon die neuen sind, wie kann das sein?
    print('Video_files: ', video_files, flush=True)
    videos = [f for f in video_files if f.endswith('.mp4')]
    print("Videos in richtiger Reihenfolge?", videos, flush=True)
    global video_paths
    video_paths.clear()

    # sind schon zwei neue Videos in Ordner? Wenn nicht, darauf warten
    while (len(videos) <= 1):
        time.sleep(5)
        videos = [f for f in video_files if f.endswith('.mp4')]

    videos.sort(key=lambda x: int(x.split('_')[1]))

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

@app.route('/is-labeling-complete', methods=['GET'])
def is_labeling_complete():
    """Prüft, ob das Labeln abgeschlossen ist."""
    global preferences_labeled
    if preferences_labeled >= Args.amount_preferences:
        return jsonify({'complete': True})
    return jsonify({'complete': False})

def find_free_port():
    """Findet einen freien Port auf dem System."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def start_flask():
    """Startet die Flask-App auf einem dynamischen Port und speichert den Port global."""
    global flask_port
    if flask_port is None:  # Nur, wenn der Port noch nicht gesetzt wurde
        flask_port = find_free_port()  # Dynamischen Port ermitteln
        print(f"Starte den Server auf Port {flask_port}...")
        flask_thread = threading.Thread(
            target=app.run, 
            kwargs={'host': '127.0.0.1', 'port': flask_port, 'debug': True, 'use_reloader': False}, 
            daemon=True
        )
        flask_thread.start()
        monitor_thread = threading.Thread(target=monitor_app, daemon=True)
        monitor_thread.start()

        time.sleep(1)  # Warten, bis Flask vollständig gestartet ist
    else:
        print(f"Flask läuft bereits auf Port {flask_port}.")
    return flask_port

from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import time
import threading
import signal
import socket
from rlhf.configs.arguments import Args

app_should_stop = False
flask_port = None
preferences_labeled = 0

app = Flask(__name__)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Erstellt das Upload-Verzeichnis, falls es nicht existiert
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Variablen
button_status = (-1, -1) # das Label
button_set = False # wurde für diese Segmente schon ein Button gedrückt (=> kann neues Label von labeling.py abgefragt werden)?
video_paths = [] # aktuelle Videos
videos_ready = False

@app.route('/stop', methods=['POST'])
def stop_app():
    """Beendet die Flask-App und bereinigt den uploads-Ordner."""
    global app_should_stop
    app_should_stop = True
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Server kann nicht gestoppt werden.')
    func()
    return jsonify({"message": "App wird beendet und uploads-Ordner bereinigt."})


# Überwachung in einem separaten Thread
def monitor_app():
    global app_should_stop, preferences_labeled
    while not app_should_stop:
        if preferences_labeled >= Args.num_queries:
            # Wartezeit, damit das Frontend die Erfolgsmeldung anzeigen kann
            time.sleep(1)
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
    global button_status
    global button_set
    global preferences_labeled
    action = request.json.get('action')
    # Label erstellen
    if action == 'left':
        button_status = (1, 0)
        button_set = True # Button wurde gedrückt
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

    # neue Videos laden
    global videos_ready
    while (videos_ready == False):
        time.sleep(0.1)
        print("in while button")
    print("aus while button")
    videos_ready = False
    video_files = os.listdir(app.config['UPLOAD_FOLDER'])
    videos = [f for f in video_files if f.endswith('.mp4')]
    global video_paths
    video_paths.clear()

    videos.sort(key=lambda x: int(x.split('_')[1]))

    # aktuelle Videos
    video_paths.append(f"/uploads/{videos[0]}")
    video_paths.append(f"/uploads/{videos[1]}")
    return jsonify({'status': button_status, 'set': button_set, 'videos': video_paths})

# Videos an frontend senden
@app.route('/get-videos', methods=['GET'])
def get_videos():
    global video_paths
    global videos_ready
    print("Video Paths:", video_paths)
    if not video_paths:
        while (videos_ready == False):
            time.sleep(0.1)
            print("in while")
        print("aus while")
        videos_ready = False
        video_files = os.listdir(app.config['UPLOAD_FOLDER'])
        print("Video Files in Uploads:", video_files)
        videos = [f"/uploads/{f}" for f in video_files if f.endswith('.mp4')]
        video_paths.extend(videos)    
    return jsonify({'videos': video_paths})

# Label an labeling.py
@app.route('/status', methods=['GET'])
def get_status():
    global button_status
    return jsonify({'status': button_status})

# Wurde Button gedrückt an labeling.py
@app.route('/set', methods=['GET'])
def get_set():
    global button_set
    return jsonify({'set': button_set})

# von labeling.py, button_set nach Verarbeitung der Label wieder auf False setzen
@app.route('/set', methods=['POST'])
def set_set():
    global button_set
    data = request.json  # Erwartet JSON-Daten
    if "new_value" in data:
        button_set = data["new_value"]
        return jsonify({"message": "Variable aktualisiert!", "variable": button_set})
    else:
        return jsonify({"error": "Kein neuer Wert übergeben!"}), 400

# von labeling.py, videos_ready nach auf True setzen, wenn Videos komplett fertig aufgenommen -> können dann angezeigt werden
@app.route('/setVideosReady', methods=['POST'])
def set_videos_ready():
    global videos_ready
    data = request.json  # Erwartet JSON-Daten
    if "new_value" in data:
        videos_ready = data["new_value"]
        return jsonify({"message": "Variable aktualisiert!", "variable": videos_ready})
    else:
        return jsonify({"error": "Kein neuer Wert übergeben!"}), 400

@app.route('/uploads/<filename>')
def uploaded_file(filename):
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
    if preferences_labeled >= Args.num_queries:
        return jsonify({'complete': True})
    return jsonify({'complete': False})

# Caching verhindern
@app.after_request
def add_no_cache_headers(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

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
        flask_thread = threading.Thread(
            target=app.run, 
            kwargs={'host': '127.0.0.1', 'port': flask_port, 'debug': True, 'use_reloader': False}, 
            daemon=True
        )
        flask_thread.start()
        monitor_thread = threading.Thread(target=monitor_app, daemon=True)
        monitor_thread.start()

        time.sleep(1)  # Warten, bis Flask vollständig gestartet ist
    return flask_port

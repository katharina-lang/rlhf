from flask import Flask, render_template, request, jsonify, send_from_directory
import os

app = Flask(__name__)

# Verzeichnis für hochgeladene Dateien
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Standardvideo definieren
DEFAULT_VIDEO_1 = 'static/uploads/rl-video1-episode-0.mp4'
DEFAULT_VIDEO_2 = 'static/uploads/rl-video-episode-0.mp4'

# Erstellt das Upload-Verzeichnis, falls es nicht existiert
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Variablen für den Button-Status
button_status = (-1, -1)

@app.route('/', methods=['GET', 'POST'])
def index():
    video1_path = DEFAULT_VIDEO_1
    video2_path = DEFAULT_VIDEO_2

    if request.method == 'POST':
        # Prüfen, ob Dateien hochgeladen wurden
        if 'file1' in request.files and 'file2' in request.files:
            file1 = request.files['file1']
            file2 = request.files['file2']

            # Dateien speichern
            if file1.filename != '':
                file1_path = os.path.join(app.config['UPLOAD_FOLDER'], 'video1.mp4')
                file1.save(file1_path)
                video1_path = file1_path

            if file2.filename != '':
                file2_path = os.path.join(app.config['UPLOAD_FOLDER'], 'video2.mp4')
                file2.save(file2_path)
                video2_path = file2_path

    return render_template('index.html', video1_path=video1_path, video2_path=video2_path)

@app.route('/button', methods=['POST'])
def button():
    global button_status
    action = request.json.get('action')
    if action == 'left':
        button_status = (1, 0)
    elif action == 'right':
        button_status = (0, 1)
    elif action == 'equal':
        button_status = (0.5,0.5)
    elif action == 'none':
        button_status = (0,0)
    else:
        return jsonify({'error': 'Ungültige Aktion'}), 400
    return jsonify({'status': button_status})

@app.route('/status', methods=['GET'])
def get_status():
    global button_status
    return jsonify({'status': button_status})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
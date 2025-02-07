from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import time
import threading
import socket
from rlhf.configs.arguments import Args
import tyro
import logging


app_should_stop = False
flask_port = None
preferences_labeled = 0
args = tyro.cli(Args)
num_queries = args.num_queries

app = Flask(__name__)

log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

button_status = (-1, -1)
button_set = False # Tells labeling.py if a new label is available
video_paths = []
videos_ready = False # From labeling.py: Are the new videos fully recorded


@app.route("/stop", methods=["POST"])
def stop_app():
    global app_should_stop
    app_should_stop = True
    func = request.environ.get("werkzeug.server.shutdown")
    if func is None:
        raise RuntimeError("Server can't be stopped.")
    func()
    return jsonify({"message": "App is stopped and uploads folder is cleaned."})


def monitor_app():
    """Monitors progress in a seperate thread."""
    global app_should_stop, preferences_labeled
    while not app_should_stop:
        if preferences_labeled >= num_queries:
            time.sleep(1)
            app_should_stop = True
            break
        time.sleep(0.1)


@app.route("/")
def index():
    print("index")
    return render_template("index.html")


@app.route("/button_action", methods=["POST"])
def button_action():
    """Is called when a button is clicked.
    Sets the label according to the human feedback.
    Loads new videos and sends them to the frontend as soon as they've been recorded in labeling.py"""
    global button_status
    global button_set
    global preferences_labeled

    # Set labels
    action = request.json.get("action")
    if action == "left":
        button_status = (1, 0)
        button_set = True
    elif action == "right":
        button_status = (0, 1)
        button_set = True
    elif action == "equal":
        button_status = (0.5, 0.5)
        button_set = True
    elif action == "none":
        button_status = (0, 0)
        button_set = True
    else:
        return jsonify({"error": "Invalid action"}), 400
    if action in ["left", "right", "equal", "none"]:
        preferences_labeled += 1
        button_set = True

    # Load new videos
    global videos_ready
    while videos_ready == False:
        time.sleep(0.1)
    videos_ready = False
    video_files = os.listdir(app.config["UPLOAD_FOLDER"])
    videos = [f for f in video_files if f.endswith(".mp4")]
    global video_paths
    video_paths.clear()

    videos.sort(key=lambda x: int(x.split("_")[1]))

    video_paths.append(f"/uploads/{videos[0]}")
    video_paths.append(f"/uploads/{videos[1]}")
    return jsonify({"button_status": button_status, "set": button_set, "videos": video_paths, "status": (preferences_labeled)})


@app.route("/get-videos", methods=["GET"])
def get_videos():
    global video_paths
    global videos_ready
    if not video_paths:
        while videos_ready == False:
            time.sleep(0.1)
        videos_ready = False
        video_files = os.listdir(app.config["UPLOAD_FOLDER"])
        videos = [f"/uploads/{f}" for f in video_files if f.endswith(".mp4")]
        video_paths.extend(videos)
    return jsonify({"videos": video_paths})


@app.route("/status", methods=["GET"])
def get_status():
    global button_status
    return jsonify({"status": button_status})


@app.route("/set", methods=["GET"])
def get_set():
    global button_set
    return jsonify({"set": button_set})


@app.route("/set", methods=["POST"])
def set_set():
    global button_set
    data = request.json
    if "new_value" in data:
        button_set = data["new_value"]
        return jsonify({"message": "Variable updated!", "variable": button_set})
    else:
        return jsonify({"error": "No new value transferred!"}), 400


@app.route("/setVideosReady", methods=["POST"])
def set_videos_ready():
    global videos_ready
    data = request.json
    if "new_value" in data:
        videos_ready = data["new_value"]
        return jsonify({"message": "Variable updated!", "variable": videos_ready})
    else:
        return jsonify({"error": "No new value transferred!"}), 400


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    response = send_from_directory(app.config["UPLOAD_FOLDER"], filename)
    response.cache_control.no_cache = True
    response.cache_control.no_store = True
    response.cache_control.must_revalidate = True
    return response


@app.route("/is-labeling-complete", methods=["GET"])
def is_labeling_complete():
    """Checks if labeling is complete."""
    global preferences_labeled
    if preferences_labeled >= num_queries:
        return jsonify({"complete": True})
    return jsonify({"complete": False})


@app.after_request
def add_no_cache_headers(response):
    """Prevent caching."""
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


def find_free_port():
    """Finds free port in the system."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def start_flask():
    """Starts Flask app on a dynamical port and saves the port globally."""
    global flask_port
    if flask_port is None:
        flask_port = find_free_port()
        flask_thread = threading.Thread(
            target=app.run,
            kwargs={
                "host": "127.0.0.1",
                "port": flask_port,
                "debug": True,
                "use_reloader": False,
            },
            daemon=True,
        )
        flask_thread.start()
        monitor_thread = threading.Thread(target=monitor_app, daemon=True)
        monitor_thread.start()

        time.sleep(1)
    return flask_port

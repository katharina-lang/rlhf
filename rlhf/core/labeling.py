import numpy as np
import torch
import os
import shutil
import requests
import time
from threading import Thread
from rlhf.utils.app import start_flask
from rlhf.core.record_segments import record_video_for_segment

class Labeling:

    counter = 0

    def __init__(self, segment_size=60, test=False, flask_port=None):
        self.segment_size = segment_size
        self.test = test
        self.flask_port = flask_port

    def preference_elicitation(self, segment_one, segment_two, env_id, iteration):
        """
        Vergleicht zwei Segmente und erstellt Labels für die Belohnungen.
        """

        print("Labeling: Flask Port ist: ", self.flask_port)
        # Verzeichnisse flexibel erstellen
        # Basispfad wird dynamisch über abspath bestimmt, Ordner werden mit makedirs erstellt, falls sie nicht existieren
        # Dateien und Pfade werden mit path.join zusammengesetzt
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        upload_dir = os.path.join(base_dir, 'uploads')
        base_url = f"http://127.0.0.1:{self.flask_port}"

        # Verzeichnisse erstellen, falls sie nicht existieren
        os.makedirs(upload_dir, exist_ok=True)

        # Verzeichnisse leeren
        def clear_directory(directory):
            if os.path.exists(directory):
                for file in os.listdir(directory):
                    file_path = os.path.join(directory, file)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)  # Dateien löschen
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)  # Unterverzeichnisse löschen
                    except Exception as e:
                        print(f"Fehler beim Löschen von {file_path}: {e}")

        clear_directory(upload_dir)


        # erstmal nicht nebenläufig: nimmt zwei Videos auf, verschiebt sie in den Ordner für Flask, labelt sie und löscht sie dann
        record_video_for_segment(env_id, segment_one, upload_dir, self.counter, iteration)
        self.counter += 1
        record_video_for_segment(env_id, segment_two, upload_dir, self.counter, iteration)
        self.counter += 1

        # Äußere Schleife sorgt für eine Wiederholung, falls es beim Abruf der Serverdaten zu einem Fehler kommt
        while True:
                try:
                    # innere Schleife wartet explizit auf einen Button-Status, um Interaktion zu erkennen
                    while True:
                        # Button-Drücke bekommen
                        # requests.get sendet eine HTTP-GET-Anfrage an den lokalen Flask-Server
                        # Antwort: Der Server antwortet mit einem JSON-Objekt, das den akt. Status enthält, z.B. {"status":[0,1]}
                        response = requests.get(f"{base_url}/status")
                        state = response.json()
                        button_status = state['status']
                        # wieder HTTP-GET-Anfrage, diesmal zur Abfrage, ob ein Button gedrückt wurde (Antwort z.B. {"set": true})
                        response2 = requests.get(f"{base_url}/set")
                        state2 = response2.json()
                        button_set = state2['set']
                        # falls Button gedrückt wurde, weitergehen, sonst darauf warten
                        if button_set:
                            break
                        # Server wird nur alle 2 Sekunden abgefragt, um Überlastungen zu vermeiden
                        # Dafür ist die Reaktionszeit nicht gut --> könnte man vielleicht auf 0.5 setzen
                        time.sleep(2)

                    if button_set:
                        # labeln
                        segment_obs_actionOne, _, predicted_rewardOne = segment_one
                        segment_obs_actionTwo, _, predicted_rewardTwo = segment_two
                        labelOne, labelTwo = button_status

                        # Status des Buttons wird auf False gesetzt
                        # --> verhindert, dass alte Statuswerte aus vorherigen Iterationen weiterverwendet werden
                        # erstmal nur lokal
                        button_set = False
                        # hier wird jetzt an Flask das Signal gesendet, den Button-Set wieder auf False zu setzen
                        response = requests.post(f"{base_url}/set", json={"new_value": button_set})


                        # fertig verarbeitete Videos aus Ordner löschen
                        # dafür Pfade der beiden Videos aus dem uploads-Ordner speichern und dann endgültig
                        # aus dem Ordner löschen, damit der wieder clean für die neuen Videos ist
                        upload_files = [f for f in os.listdir(upload_dir) if f.endswith('.mp4')]
                        upload_paths = [os.path.join(upload_dir, file) for file in upload_files]

                        for upload_file in upload_paths:
                            os.remove(upload_file)

                        # jetzt auch aus der äußeren Schleife rausgehen, weil Benutzerabfrage erfolgreich stattgefunden hat
                        break
                    time.sleep(1)

                # Falls die Verbindung zum Flask-Server fehlschlägt (z.B. Server nicht gestartet), wird der Fehler abgefangen.
                # Statt eines Absturzes wird 1 Sekunde gewartet und dann erneut versucht, sich zu verbinden.
                except requests.exceptions.ConnectionError as e:
                    print(f"Fehler bei der Verbindung zum Flask-Server: {e}")
                    time.sleep(1)

        return (
            segment_obs_actionOne,
            segment_obs_actionTwo,
            (labelOne, labelTwo),
            (predicted_rewardOne, predicted_rewardTwo),
        )

    def select_segments(
        self, obs_action_pair_buffer, env_reward_buffer, predicted_rewards_buffer, amount_preferences
    ):
        """
        Wählt zufällige Segmente aus den Buffern aus und berechnet deren Belohnungen.
        """
        obs_action_pair_buffer = np.array(obs_action_pair_buffer)
        env_reward_buffer = np.array(env_reward_buffer)
        predicted_rewards_buffer = np.array(predicted_rewards_buffer)

        data_points = len(env_reward_buffer)
        segment_amount = amount_preferences*2

        segments = []
        for _ in range(segment_amount):
            start_idx = np.random.randint(0, data_points - self.segment_size)
            end_idx = start_idx + self.segment_size
            segment_obs_action = obs_action_pair_buffer[start_idx:end_idx]
            true_reward = sum(env_reward_buffer[start_idx:end_idx])
            predicted_reward = predicted_rewards_buffer[start_idx:end_idx]
            segment = (segment_obs_action, true_reward, predicted_reward)
            segments.append(segment)

        return segments

    def get_labeled_data(
        self, obs_action_pair_buffer, env_reward_buffer, predicted_rewards_buffer, env_id, iteration, amount_preferences
    ):
        """
        Vergleicht Segmente paarweise und erstellt die gelabelten Daten.
        """
        labeled_data = []
        segments = self.select_segments(
            obs_action_pair_buffer, env_reward_buffer, predicted_rewards_buffer, amount_preferences
        )
        while len(segments) > 1:
            segment_one = segments.pop()
            segment_two = segments.pop()
            segments_label_reward = self.preference_elicitation(
                segment_one, segment_two, env_id, iteration
            )
            # Ich hätte gedacht, man muss hier auf Flask warten
            labeled_data.append(segments_label_reward)

        return labeled_data

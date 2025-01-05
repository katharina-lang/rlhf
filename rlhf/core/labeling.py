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

    def __init__(self, segment_size=60, test=False):
        self.segment_size = segment_size
        self.test = test

    def preference_elicitation(self, segment_one, segment_two, env_id, iteration):
        """
        Vergleicht zwei Segmente und erstellt Labels für die Belohnungen.
        """

        # Verzeichnisse flexibel erstellen
        # Basispfad wird dynamisch über abspath bestimmt, Ordner werden mit makedirs erstellt, falls sie nicht existieren
        # Dateien und Pfade werden mit path.join zusammengesetzt
        base_dir = os.path.dirname(os.path.abspath(__file__))
        segment_dir = os.path.join(base_dir, 'segment_videos')
        upload_dir = os.path.join(base_dir, 'rlhf/utils/static/uploads')
        # erstmal nicht nebenläufig: nimmt zwei Videos auf, verschiebt sie in den Ordner für Flask, labelt sie und löscht sie dann
        record_video_for_segment(env_id, segment_one, f"segment_videos", self.counter)
        self.counter += 1
        record_video_for_segment(env_id, segment_two, f"segment_videos", self.counter)
        self.counter += 1


        # in video_files werden alle Dateien im angegebenen Pfad aufgelistet (['video1.mp4','video2.mp4'])
        video_files = [f for f in os.listdir(segment_dir) if f.endswith('.mp4')]
        # Dateinamen aller enthaltenen Videos (Endung .mp4) in einer Liste videos speichern (sollten genau 2 sein)
        video_paths = [os.path.join(segment_dir, video) for video in video_files[:2]]

        # shutil.move nimmt den Dateipfad von der Datei, die verschoben werden soll und den Pfad zu der Stelle, wo
        # die Datei hinverschoben werden soll und verschiebt die betreffende Datei dann
        # Verbesserungsvorschlag von ChatGPT:
        # target_folder = 'C:/users/hanna/rlhf/rlhf/utils/static/uploads'
        # os.makedirs(target_folder, exist_ok=True)
        # for video in video_paths:
        #   shutil.move(video, os.path.join(target_folder, os.path.basename(video)))
        for video in video_paths:
            shutil.move(video, upload_dir)

        # Äußere Schleife sorgt für eine Wiederholung, falls es beim Abruf der Serverdaten zu einem Fehler kommt
        while True:
                try:
                    # innere Schleife wartet explizit auf einen Button-Status, um Interaktion zu erkennen
                    while True:
                        # Button-Drücke bekommen
                        # requests.get sendet eine HTTP-GET-Anfrage an den lokalen Flask-Server
                        # Antwort: Der Server antwortet mit einem JSON-Objekt, das den akt. Status enthält, z.B. {"status":[0,1]}
                        response = requests.get('http://127.0.0.1:5000/status')
                        state = response.json()
                        button_status = state['status']
                        # wieder HTTP-GET-Anfrage, diesmal zur Abfrage, ob ein Button gedrückt wurde (Antwort z.B. {"set": true})
                        response2 = requests.get('http://127.0.0.1:5000/set')
                        state2 = response2.json()
                        button_set = state2['set']
                        print(button_status)
                        print(button_set)
                        # falls Button gedrückt wurde, weitergehen, sonst darauf warten
                        if (button_set == True):
                            break
                        # Server wird nur alle 2 Sekunden abgefragt, um Überlastungen zu vermeiden
                        # Dafür ist die Reaktionszeit nicht gut --> könnte man vielleicht auf 0.5 setzen
                        time.sleep(2)

                    if (button_set == True):
                        # labeln
                        segment_obs_actionOne, _, predicted_rewardOne = segment_one
                        segment_obs_actionTwo, _, predicted_rewardTwo = segment_two
                        labelOne, labelTwo = button_status

                        # Status des Buttons wird auf False gesetzt
                        # --> verhindert, dass alte Statuswerte aus vorherigen Iterationen weiterverwendet werden
                        # erstmal nur lokal
                        button_set = False
                        # hier wird jetzt an Flask das Signal gesendet, den Button-Set wieder auf False zu setzen
                        response = requests.post('http://127.0.0.1:5000/set', json={"new_value": button_set})

                        print('label gesetzt', button_status)

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
        self, obs_action_pair_buffer, env_reward_buffer, predicted_rewards_buffer
    ):
        """
        Wählt zufällige Segmente aus den Buffern aus und berechnet deren Belohnungen.
        """
        obs_action_pair_buffer = np.array(obs_action_pair_buffer)
        env_reward_buffer = np.array(env_reward_buffer)
        predicted_rewards_buffer = np.array(predicted_rewards_buffer)

        data_points = len(env_reward_buffer)
        segment_amount = data_points // self.segment_size

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
        self, obs_action_pair_buffer, env_reward_buffer, predicted_rewards_buffer, env_id, iteration
    ):
        """
        Vergleicht Segmente paarweise und erstellt die gelabelten Daten.
        """
        labeled_data = []
        segments = self.select_segments(
            obs_action_pair_buffer, env_reward_buffer, predicted_rewards_buffer
        )

        # Flask als Thread starten
        # Funktion start_flask() wird als Ziel angegeben --> nach dem Starten des Threads wird diese Funktion ausgeführt
        flask_thread = Thread(target=start_flask, daemon = True)
        flask_thread.start()
        # Frage: Welche Funktion führt Flask eigentlich aus??
        # Flask verwendet eine Schleife, die kontinuierlich auf HTTP-Anfragen wartet
        # Wird Flask automatisch wieder beendet?

        while len(segments) > 1:
            segment_one = segments.pop()
            segment_two = segments.pop()
            segments_label_reward = self.preference_elicitation(
                segment_one, segment_two, env_id, iteration
            )
            # Ich hätte gedacht, man muss hier auf Flask warten
            labeled_data.append(segments_label_reward)
        
        # warum wartet man hier auf Flask?
        # flask_thread.join --> kann glaube ich weg

        # Flask ganz beenden. Ist die Frage, ob wir das nach jeder Episode wollen oder ob Flask lieber die ganze
        # Zeit laufen soll --> hängt wahrscheinlich auch davon ab, wie wir das mit der Anzahl der Abfragen machen
        # requests.post('http://127.0.0.1:5000/shutdown', json={})

        return labeled_data

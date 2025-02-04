# User Interface Documentation

- [Structure](#structure)

## Structure

Our UI includes the frontend index.html and the backend app.py, which communicates with labeling.py. <br>

The frontend is responsible for displaying the user interface; the videos are passed from the backend to the frontend and get displayed by it. <br>
When a button for labeling is clicked, the backend is informed about the choice. <br>
The frontend is also responsible for disabling the buttons and displaying a loading message while the next videos are being recorded and the human has nothing to evaluate.

The backend is responsible for the logic behind the user interface. <br>
It processes the button clicks from the frontend and sends the resulting label to labeling.py. <br>
After that, it fetches the new videos from the designated uploads folder. <br>
It also checks whether labeling is complete and, if so, terminates the Flask thread.

If human feedback is desired, labeling.py creates a folder for uploading videos in preference_elicitation(). It checks to make sure that the folder is empty and, if not, empties it. <br>
After that, the videos for the two current segments are recorded in record_segments.py. To do this, the segments are split into their individual observations and actions, which are then executed step by step and recorded. When this is complete and the two videos are in the uploads folder, app.py is notified. <br>
As soon as the frontend displays the videos, a human can label them. The label is stored in a tuple in the backend and a boolean variable is set when a new label becomes available. As soon as this boolean variable is True, labeling.py fetches the label from app.py and get_labeled_data() appends it to the labeled_data list so that the reward models can be trained with it.

Incomplete sequence diagram of the process:

![Incomplete sequence diagram of the process](rlhf/readme_images/SequenzUI.png)
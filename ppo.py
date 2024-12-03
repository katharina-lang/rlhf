'''
Dieses Modul ermöglicht es Befehlszeilenargumente (also Eingabewerte, die beim Starten des Programms über die Kommandozeile übergeben werden) zu analysieren und zu verarbeiten. Das Modul ist sehr nützlich, wenn du ein Python-Skript schreibst, das von der Kommandozeile aus aufgerufen werden soll und benutzerdefinierte Eingabewerte benötigt.
'''
import argparse 
'''
- Modul, das Zugriff auf Betriebssystemfunktionen bietet. Mit os kannst du z.B. mit Dateien und Verzeichnissen arbeiten oder Umgebungsvariablen abfragen.
- verwendet, um den Namen der Datei zu bekommen, die gerade ausgeführt wird.
'''
import os 
'''
 Funktion wird verwendet, um einen String (Text) in einen booleschen Wert umzuwandeln, also True oder False. Zum Beispiel würde die Eingabe von 'yes' oder 'no' in einen entsprechenden booleschen Wert konvertiert werden. Diese Funktion ist besonders nützlich, wenn du Eingaben von Benutzern verarbeiten musst, die True oder False sein sollten, aber als Text (z.B. 'yes'/'no') vorliegen
'''
from distutils.util import strtobool 

'''
Kontrollmechanismus, um sicherzustellen, dass der folgende Code nur ausgeführt wird, wenn das Skript direkt gestartet wird (nicht, wenn es von einem anderen Skript importiert wird).
'''

from torch.utils.tensorboard import SummaryWriter 


import time


'''
Um Eingabewerte (Argumente) des Programms zu analysieren, die über die Kommandozeile eingegeben werden
'''
def parse_args():
    '''
    Ermöglicht uns zu definieren welche Eingabewerte unser Program erwartet
    '''
    parser = argparse.ArgumentParser() 
    '''
    Name, Type, Default Value, Help Text for documentation; Experiment name takes by default name of the file
    '''
    parser.add_argument('--exp name', type=str, default=os.path.basename(__file__).rstrip(".py"), 
                        help='the name of this experiment') #Standartwert, falls der Benutzer Argument nicht übergibt
    '''
    Als nächstes Gym ID
    '''
    parser.add_argument('--gym-id', type=str, default= "CarloPole-v1",
                        help='the id of the gym environment')
    '''
    Learning rate
    '''
    parser.add_argument('--learning-rate', type=float, default=2.5e-4,
                        help='the learning rate of the optimizer') 
    
    '''
    Random seed by default set it to 1
    Seed sorgt dafür, dass bei der Ausführung eines Programms, das Zufallszahlen verwendet, die gleichen Zufallszahlen erzeugt werden 
    seed ist dann immer eine bestimmte serie von zufallszahlen
    '''
    parser.add_argument('--seed', type=int, default=1, 
                        help='seed of the experiment') 
    
    '''
    Total timesteps= number of environment stevs
    '''

    parser.add_argument('--total-timesteps', type=int, default=25000,
                        help='total timesteps of the experiment') 
    
    '''
    two gpu related variables
    - torch deterministic equals true will help us reproduce experiments 
    - the cuda euqals true will help us utilize gpu whenever possible-> verkürzt Berechnungszeit, denn GPUs sind schneller als CPUs

    torch-deterministic:
    Diese Variable bezieht sich auf eine Einstellung von PyTorch, einer beliebten Bibliothek für maschinelles Lernen.
    PyTorch verwendet für bestimmte Operationen auf der GPU nicht-deterministische Algorithmen, die dazu führen können, dass die Ergebnisse bei verschiedenen Durchläufen leicht variieren. Das bedeutet, dass selbst bei Verwendung des gleichen Seeds unterschiedliche Ergebnisse auftreten können.
    Wenn torch-deterministic=True gesetzt ist, sorgt das dafür, dass PyTorch deterministische Berechnungen verwendet, was zu einer besseren Reproduzierbarkeit führt. Der Algorithmus wird also immer gleich ablaufen und das Ergebnis wird bei jedem Durchlauf gleich sein.
    Es wird durch die Zeile torch.backends.cudnn.deterministic=True in PyTorch gesteuert.

    Zusammenfassung: Diese Variable sorgt dafür, dass deine Experimente bei jedem Durchlauf exakt gleich sind, wenn du denselben Seed verwendest.

    ''' 

    parser.add_argument('--torch-determiistic', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, `torch.backends.cudnn.deterministic=False`') 
    
    # toggled bedeutet es wir dentweder aktiviert oder eben nicht aktiviert 
    parser.add_argument('--cuda', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True, 
                        help='if toggled, cuda will not be enabled by default')

    '''
    track: um anzugeben, ob das Experiment mit einem Tool namens "Weights & Biases" (kurz wandb) verfolgt werden soll

    "Weights & Biases" (wandb) ist ein Tool, das von Entwicklern und Forschern im Bereich maschinelles Lernen verwendet wird, um ihre Experimente zu verfolgen, zu organisieren und zu visualisieren. Es ist besonders nützlich, wenn du viele Experimente mit verschiedenen Hyperparametern, Modellen oder Datensätzen durchführst und eine Möglichkeit benötigst, um die Ergebnisse zu speichern, zu vergleichen und zu analysieren.

    Ein Experiment der Prozess, bei dem ein Modell mit einem bestimmten Datensatz trainiert wird. Bei jedem Experiment kannst du verschiedene Hyperparameter wie Lernrate, Anzahl der Epochen oder Architekturen des Modells ändern, um zu sehen, wie sich das Modellverhalten verändert. Jedes Mal, wenn du diese Parameter änderst und das Modell erneut trainierst, startest du ein neues Experiment. Du möchtest die Ergebnisse dieser Experimente aufzeichnen, um die beste Konfiguration zu finden.


    Was bedeutet "verfolgen"?
    Wenn wir sagen, dass das Experiment verfolgt wird, bedeutet das, dass alle wichtigen Informationen zum Experiment – wie die Hyperparameter, die während des Trainings verwendet wurden, und die Ergebnisse, die erzielt wurden (z. B. die Genauigkeit des Modells, der Verlust) – aufgezeichnet und gespeichert werden. Dies ist besonders wichtig, wenn du viele Experimente durchführst und den Überblick behalten möchtest.

    Verfolgen bedeutet also:

    - Alle Eingabewerte (z. B. Hyperparameter wie Lernrate, Batchgröße).
    - Alle Ergebnisse des Experiments (z. B. die Leistung des Modells während des Trainings).
    - Alle Grafiken, die im Verlauf des Trainings generiert werden (z. B. Verlust- und Genauigkeitskurven).

    Wie hilft Weights & Biases dabei?
    Weights & Biases ermöglicht es, all diese Informationen in einer zentralen Plattform zu speichern. Du kannst dann die Ergebnisse deiner Experimente leicht vergleichen und analysieren. Die wichtigsten Funktionen von wandb sind:

    - Experiment-Tracking: Du kannst Experimente mit verschiedenen Hyperparametern starten und wandb wird alle Details und Ergebnisse speichern.
    - Visualisierung: Du kannst Verlaufsdaten visualisieren, z. B. wie sich der Verlust während des Trainings verändert hat.
    - Vergleich von Experimenten: Wenn du viele Experimente durchführst, kannst du diese leicht miteinander vergleichen, um herauszufinden, welches Modell oder welche Hyperparameter die besten Ergebnisse liefern.

    Was passiert also, wenn du --track aktivierst?
    
    Wenn du den Schalter --track in deinem Programm aktivierst, wird das Experiment automatisch mit Weights & Biases verbunden. Das bedeutet:

    - Das Programm wird automatisch alle wichtigen Daten (wie Hyperparameter und Metriken) an wandb senden.
    - Du kannst deine Experimente und deren Ergebnisse später in einem Web-Interface einsehen und auswerten.
    '''
    parser.add_argument('--track', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True, 
        help='if toggled, this experiment will be tacked with weights and biases')

    # Namen des Projekt definieren, das mit Weights&Biases verbunden ist
    parser.add_argument('--wand-project-name', type=str,default="cleanRL", 
    help="the wand's project name") 

    parser.add_argument('--wandb-entity', type=str, default=None, 
    help="the entity (team) of wandb's project")   
    '''
    Der parser verarbeitet und analysiert die übergebenen Kommandozeilenargumente.

    Verarbeiten (Parsing):
    - Programm nimmt die Kommandozeilenargumente (--exp-name, MyExperiment, --learning-rate, 0.01) und erkennt sie als die definierten Optionen (Argumente).
    - Es überprüft, ob die angegebenen Argumente gültig sind (z. B., ob --learning-rate tatsächlich erwartet wird und ob der Wert 0.01 im richtigen Format vorliegt, in diesem Fall als float).

    Auswerten: 
    Die verarbeiteten Eingaben werden in eine strukturierte Form gebracht, üblicherweise ein Python-Objekt (z. B. Namespace), das die Werte enthält:

    '''
    args = parser.parse_args() #sammelt Eingaben, die über die Kommandozeile an das Programm übergeben wurde und speichert sie in der Variable args

    '''
    Ergebnis der Funktion wird der Variable args zugewiesen
    '''
    args = parse_args() 
    print(args) 


'''
Kontrollmechanismus, um sicherzustellen, dass der folgende Code nur ausgeführt wird, wenn das Skript direkt gestartet wird (nicht, wenn es von einem anderen Skript importiert wird).
'''
if __name__ == "__main__" :
    args = parse_args() 
    print(args)
#a unique run name for our experiment
    run_name = f"{args.gym.id}_{args.exp_name}_{args.seed}_{int(time.time())}"
# save the metrics of our experiment to a folder with that run name 
    writer = SummaryWriter(f"runs/{run_name}") 

#to test that out we are going to encode our args variable as a text data  and we are also going to add some scalar data 

    writer.add_text (
    "hyperparameters"
    "|param|value|\n%s" % ("\n".join([f"|{key}|{value}" for key, value in vars(args).items()]))
    ) 

    for i in range(100) :
    writer.add_scalar("test_loss", i*2, global__step=i) 


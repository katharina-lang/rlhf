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
    
    parser.add_argument('--cuda', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True, 
                        help='if toggled, cuda will not be enabled by default')
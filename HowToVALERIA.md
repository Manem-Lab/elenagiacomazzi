# How to VALERIA

## 1. Get access to Valeria:
- supervisor has to ask for that (Venkata wrote a mail for getting me access)
- I then could sign into the Valeria browser and ssh to the server
- `ssh <your-IDUL>@login.valeria.science` + Valeria pw
- Was not allowed to use any of the tools in Valeria -> Venkata had to send a second mail to ask that I get access for them (including my Laval mail)
## 2. Folder structure of the account:
- Home -> personal area
- Project -> to work in a team
- Scatch -> working area
- Public -> tutorials etc
## 3. Bring data to server
- git-repository:
    - clone the repository into home, project or scatch (in the documentation there are more details about the best practice)
- Use Globus Connect Personal to create a node on your personal computer and being able to transfer the data to other points in the system
    - Download globus Connect Personal
    - create collection and say what folders the collection can access on my laptop
    - go to the webapp and look for my local collection (IUCPQ-Ocotech) and select the files I want to sync
    - find the VALERIA laval account and connect with credentials to it
    - start the transaction
## 4. Start JupyterHub/JupyterLab
- In Valeria (when you got all the permissions) you can start a JupyterHub Server (I just used the default settings for it) 
- so far no python libraries are available in there but you can look at your code, open a Terminal
- How to run your code including all the libraries will be explained later

## 5. The way to submitting my first job to the server
- Jobs are submitted using a bash script which uses SLURM parameters
1. Slurm
    - [Parameter desciptions](https://doc.s3.valeria.science/fr/calcul/slurm.html)
2. Load Module and Python Environment
    - To run python on Valeria you need to load modules `module load StdEnv/2020 python/3.8 scipy-stack/2022a`
    - this module has to be loaded each time when you exit and reenter the server
    - Then You need a virtural python environment
        - create environnement virtuel 
            - `virtualenv --no-download ~/venvs/mon-projet` 
        - activate environnement virtuel
            - `source ~/venvs/mon-projet/bin/activate`
        - always upgrade pip 
            - `pip install --no-index --upgrade pip`
        - install other Python libraries via wheels
            - search for available wheels:
                - `avail_wheels pandas`
                - `avail_wheels --name "*cdf*"` (looks for library which include "cdf")
        - this virtual environment stays on your account and to activate it on a relogin you need to
            1. reload the modules
                - `module load StdEnv/2020 python/3.8 scipy-stack/2022a`
            2. activate the environemnt
                - `source ~/venvs/mon-projet/bin/activate`
        - all this will be put into a bash script in which the modules are loaded an the environment is created
        - build-env1.sh :
         ```bash
         #!/bin/bash
        module load StdEnv/2020 python/3.8 scipy-stack/2022a
        module save mymodules-venv1

        Create virtual env
        virtualenv --no-download ~/venvs/venv1
        source ~/venvs/venv1/bin/activate

        pip install --no-index --upgrade pip
        pip install --no-index scikit_learn lifelines boto3 s3fs sqlalchemy psycopg2 pgpasslib xlrd openpyxl
        pip install pymrmre
        ```
    - To start a job a second bash script is needed:
        - declare all sbatch parameters, reload earlier saved module and activate virtual environment and finally start the python script
        - run-job.sh:
        ```bash
        #!/bin/bash
        #SBATCH --nodes=1
        #SBATCH --partition=bigmem
        #SBATCH --nodelist=ul-val-pr-cpu90
        #SBATCH --cpus-per-task=32
        #SBATCH --mem=200G
        #SBATCH --job-name=my-batch
        #SBATCH --output=%x-%j.out

        echo "Restoring modules"
        module restore mymodules-venv1

        # Create virtual env
        source ~/venvs/venv1/bin/activate
        echo "venv activated"

        echo "call python script next"
        python script_to_train.py
        ```
    - The python script
        - I have to load and save data but the server is not able to get it from my local file paths. The following steps need to be performed:
            1. Go to the Valeria login browser -> My Stockage
            2. Start the S3 Navigateur with *lancer*
            3. Create a bucket (compartiment) with *ajouter compartiment*, name it *bucketname* (no upper letters and spaces) 
            4. Upload the needed files directly to the bucket or create an extra folder
        - Tell the python code where it can find that bucket
            - boto3 helps with identifying you with your credentials (I do not know exacly how it works but it does with this lines of code so I don't complain. :D)
        ```python 
        s3 = boto3.resource('s3', aws_access_key_id= 'YOUR_ACCESS_KEY_ID', aws_secret_access_key='YOUR_SECRET_ACCESS_KEY')
        ENDPOINT_URL = 'https://s3.valeria.science'
        bucket = 'bucketname'
        # Import
        df = pd.read_csv(f"s3://{bucket}/folder_name_if_you_created_one/file_name.csv", storage_options={"client_kwargs": {'endpoint_url': ENDPOINT_URL}})
        # Export
        df.to_csv(f"s3://{bucket}/folder_name_if_you_created_one/file_name.csv", storage_options={"client_kwargs": {'endpoint_url': ENDPOINT_URL}})
        ```
        - You can also try the working of this in a JupyterHub if you loaded the modules, virtual environment and made it available for jupyterHub (command at the bottom of this page or explanation [here in french](https://doc.s3.valeria.science/fr/calcul/python/python-jupyterlab.html) )
    - To start the job:
        - `sbatch build-env1.sh`
        - `sbatch run-job.sh`
    - To check if it is running and which jobs are running in general:
        - `sq`
    - To see the output file
        - `cat my-batch-JOBID.out`
        
### Make a python environment available for JupyterLab:
```bash
export NAME=mon-projet
module reset
module load python/3.9 scipy-stack/2022a
module save $NAME

virtualenv --no-download ~/venvs/$NAME
source ~/venvs/$NAME/bin/activate
pip install --no-index --upgrade pip
# install other libraries that you need

python -m ipykernel install --name ${NAME} --user
```
In JupyterLab:
- Go to the upper right corner and select the kernel -Y choose the name you named your project and you can execute code with all the libraries installed in this environment

1. I need data 
    - Download globus Connect Personal
    - create collection and say what folders the collection can access on my laptop
    - go to the webapp and look for my local collection (IUCPQ-Ocotech) and select the files I want to sync
    - find the VALERIA laval account and connect with credentials to it
    - start the transaction
2. ssh works into that account and the same account is also shown in the jupyterHub/lab window
- but it does not have scikit installed




# Valeria

## Documentation
1. Log in stuff and generel access what for whom
2. Navigateur S3 -> Objektspeicher
- Valeria -> Mon stockage -> lacer le navigateur 
- (Kommandozeile: rclone, da genauer erklärt)
    1. Fach erstellen (compartiment)
    - Click AJOUTER UN COMPARTIMENT -> Name eingeben (Wichtig zum Projekt bezeichnung)
    - _Ajouter_ = hinzufügen
    2. Fach löschen (supprimer)
    - Drei Punkte rechts daneben -> supprimer
    - Name des Fachs in Feld eingeben, um **ALLES** zu löschen
    - Geht nur mit Rechten dazu
    3. Leuten Upload und löschen erlauben (téléversement et supprimer)
    - auf Fach klicken, -> _Autoriser_  (S3 browser kann mit Objektspeicher S3 kommunizieren (beides Valeria))

    1. Verwalten von daten (télecharger et supprimer) des fichiers
    - Alle Dateiformate
    - Details für téléchargement
3. Politique d'access aux compartiments S3
- Details über wer wann zu was Zugriff hat
- Wer was machen darf und wo anfragen
4. Obtenir ses clés S3
- Wie auf S3 Speicherzuzugreifen -> wird Navigateur S3 empfohlen (Browser S3)
- Brauche ich Zugang ? -> Wenn ja Schlüssel anfordern!
5. Procédure d'utilisation du logiciel S3 - CyperDuck
- Möglich für S3 nutzung -> software dafür 
- Erklärt wie das alles abläutf mit Installation,  zugang etc.
6. rclone
- wie man das benutzt
7. s3cmd
- Kommandozeilenprogramm, das schon zur Verfügung steht 
- Über jupyterLab zugriff auf Dateisystem
- Sieht am besten aus, wenn ich das irgendwie brauche

8. Utilisation de Lustre (Speicher)
- In JupyterHub UND Globus integriert
- 1 TB
    - Zugriff von jupyterLab aus:
        - Mes outiles -> Lancer Jupyter -> partir un server(start)
    - oder über ssh und ls
- Ordner:
    - Home -> persönlicher Bereich
    - Project -> Projektbereich zum Mitarbeiter teilen
    - Scatch -> arbeitsbereich
    - Public -> tutorials etc
    
9. Passage à l'environnement standard 2020
- StdEnv/2020 umgebung
    1. Liste modules activités -> `module list` oder unter *Softwares*
- Umstellung von alt nach neu
- Python Environments (!!!) in [Python Doc](https://doc.s3.valeria.science/fr/calcul/python/env_virt_python.html) angucken!

10. Python - Utilisation de base en mode interactif
- jupyter und notebooks erklärt 
- mit links zu Calcul Canada modules, python & jupyterLab

11. Création et gestion des environnements virtuels Python
- meisten Module als wheels installiert in Wheelhouse
- Reihnfolge zum Erstellen von venv
    - charger modules 
        - `module load StdEnv/2020 python/3.8 scipy-stack/2022a`
    - créer environnement virtuel 
        - `virtualenv --no-download ~/venvs/mon-projet` 
    - activer environnement virtuel
        - `source ~/venvs/mon-projet/bin/activate`
    - mise-à-jour de pip 
        - `pip install --no-index --upgrade pip`
    - installation des paquets Python
        - search for available wheels:
            - `avail_wheels pandas`
            - `avail_wheels --name "*cdf*"`
- zum Nutzen von venv
    - charger modules 
        - `module load StdEnv/2020 python/3.8 scipy-stack/2022a`
    - activer environnement virtuel
        - `source ~/venvs/mon-projet/bin/activate`

## Info from Louis-Jacques
- bash_job.sh script need command at tip
```
#!/bin/bash
#SBATCH --partition=bigmem
#SBATCH --nodelist=ul-val-pr-cpu90
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --job-name=my-batch
#SBATCH --output=%x-%j.out
echo ‘Test for MIXCR processing of 1 sample!’ 

# load module
module load StdEnv/2020 python/3.8 scipy-stack/2022a
# Create virtual env
virtualenv --no-download ~/venvs/venv1
source ~/venvs/venv1/bin/activate

pip install --no-index --upgrade pip
pip install --no-index scikit_learn lifelines
pip install pymrmre

python script.py
```
- in command line: `sbatch bash_job.sh` to start the above script

- continue reading here: https://doc.s3.valeria.science/fr/calcul/hpc-s3-interactif-batch.html X
- check for all the steps that need to be performed before python script can be called X
- translate my scripts into python scripts X
- with large loops also through the different feature selection methods X
- Understand how it accesses files and saves files [here guuut](https://vsoch.github.io/lessons/sherlock-jobs/)
- Try this: [this](https://doc.s3.valeria.science/fr/calcul/hpc-s3-interactif-batch.html)

- Compartment erstellen -> Mon stockage -> lancer -> name und go
- for commandline tools activate: -> go to public/exemples/700 und 'python val-...'
- Jetzt habe ich immer noch keinen Zugang um die Datei im notebook zu laden, aber sie wird mir mit rclone angezeigt

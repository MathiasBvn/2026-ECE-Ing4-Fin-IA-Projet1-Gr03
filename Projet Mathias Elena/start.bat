@echo off
echo === INITIALISATION DU PROJET MATHIAS ELENA ===

REM Définition du chemin Python
set PYTHON_PATH=C:\Users\elena\AppData\Local\Programs\Python\Python312\python.exe

REM 1. Vérification et création du venv si absent
if not exist venv (
    echo [INFO] Création de l'environnement virtuel...
    "%PYTHON_PATH%" -m venv venv
) else (
    echo [INFO] Environnement virtuel détecté.
)

REM 2. Activation de l'environnement
echo [INFO] Activation de venv...
call venv\Scripts\activate

REM 3. Installation des dépendances
echo [INFO] Installation des librairies (cela peut prendre du temps)...
pip install -r requirements.txt

REM 4. Lancement de l'application
echo [INFO] Lancement de Streamlit...
set STREAMLIT_SERVER_HEADLESS=true
streamlit run dashboard.py

pause

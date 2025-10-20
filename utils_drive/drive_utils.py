import os
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

# 🔹 Chemins des fichiers
BASE_DIR = os.path.dirname(__file__)
CLIENT_SECRETS_FILE = os.path.join(BASE_DIR, "client_secrets.json")
CREDENTIALS_FILE = os.path.join(BASE_DIR, "credentials.json")

# 🔹 Authentification Google Drive
gauth = GoogleAuth()
gauth.LoadClientConfigFile(CLIENT_SECRETS_FILE)  # charge client_secrets.json
gauth.LoadCredentialsFile(CREDENTIALS_FILE)      # charge credentials.json si existant

if gauth.credentials is None:
    gauth.LocalWebserverAuth()  # première fois : ouvre le navigateur pour autoriser
elif gauth.access_token_expired:
    gauth.Refresh()
else:
    gauth.Authorize()

gauth.SaveCredentialsFile(CREDENTIALS_FILE)  # sauvegarde pour les prochaines exécutions

# 🔹 Création de l'objet GoogleDrive
drive = GoogleDrive(gauth)

# 🔹 IDs des dossiers Drive
ORIGINALS_FOLDER_ID = "1PXmoyabvfMnr0yxTJb2USxnaTgbZyADv"
TEXTS_FOLDER_ID = "1_y_bbWfpOaZXXSMbPuSrbBBS3lSn3165"

# 🔹 Fonction d'upload
def upload_to_drive(local_path, folder_id=None):
    """Upload un fichier sur Google Drive et retourne le lien partageable"""
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Le fichier {local_path} n'existe pas !")

    file_name = os.path.basename(local_path)
    file_drive = drive.CreateFile({
        'title': file_name,
        'parents':[{'id': folder_id}] if folder_id else []
    })
    file_drive.SetContentFile(local_path)
    file_drive.Upload()
    file_drive.InsertPermission({'type': 'anyone', 'value': 'anyone', 'role': 'reader'})
    
    print(f"✅ Fichier '{file_name}' envoyé sur Drive : {file_drive['alternateLink']}")
    return file_drive['alternateLink']

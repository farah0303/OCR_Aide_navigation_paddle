from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
import os

# 🔹 Paths for client secrets and credentials
BASE_DIR = os.path.dirname(__file__)
CLIENT_SECRETS_FILE = os.path.join(BASE_DIR, "client_secrets.json")
CREDENTIALS_FILE = os.path.join(BASE_DIR, "credentials.json")

# 🔹 Authenticate with Google Drive
gauth = GoogleAuth()
gauth.LoadClientConfigFile(CLIENT_SECRETS_FILE)
gauth.LoadCredentialsFile(CREDENTIALS_FILE)

gauth.settings['client_config_file'] = CLIENT_SECRETS_FILE
gauth.settings['get_refresh_token'] = True  # offline mode

if gauth.credentials is None:
    gauth.LocalWebserverAuth()  # first-time authentication
elif gauth.access_token_expired:
    gauth.Refresh()
else:
    gauth.Authorize()

gauth.SaveCredentialsFile(CREDENTIALS_FILE)

# 🔹 Google Drive object
drive = GoogleDrive(gauth)

# 🔹 Function to upload file
def upload_to_drive(local_path, folder_id=None):
    """Upload a file to Google Drive and return the shareable link."""
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Le fichier {local_path} n'existe pas !")

    file_name = os.path.basename(local_path)
    file_drive = drive.CreateFile({
        'title': file_name,
        'parents': [{'id': folder_id}] if folder_id else []
    })
    file_drive.SetContentFile(local_path)
    file_drive.Upload()
    file_drive.InsertPermission({'type': 'anyone', 'value': 'anyone', 'role': 'reader'})

    print(f"✅ Fichier '{file_name}' envoyé sur Drive : {file_drive['alternateLink']}")
    return file_drive['alternateLink']

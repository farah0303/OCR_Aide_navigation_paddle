from boxsdk import OAuth2, Client
import json, os

# ==============================
# ⚙️ Lecture et sauvegarde automatiques des tokens
# ==============================
BASE_DIR = os.path.dirname(__file__)
TOKEN_FILE = os.path.join(BASE_DIR, "config.json")


def store_tokens_callback(access_token, refresh_token):
    """Sauvegarde automatique des tokens mis à jour."""
    with open(TOKEN_FILE, "w") as f:
        json.dump({
            "access_token": access_token,
            "refresh_token": refresh_token
        }, f)

# Charger les tokens actuels
with open(TOKEN_FILE, "r") as f:
    tokens = json.load(f)

oauth = OAuth2(
    client_id="0gcd5tewkuobhqgngz9ghgo37po1vdvy",
    client_secret="f3LItDHnGTrTCusjuBtdDqzOcZpkQ91H",
    access_token=tokens["access_token"],
    refresh_token=tokens["refresh_token"],
    store_tokens=store_tokens_callback
)

client = Client(oauth)
# ==============================
# 📤 Fonction d’upload sur Box
# ==============================
def upload_to_box(local_path, folder_id):
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Le fichier {local_path} n'existe pas !")

    file_name = os.path.basename(local_path)
    folder = client.folder(folder_id)
    print(f"⬆️ Upload de '{file_name}' vers Box (folder {folder_id}) ...")

    # Supprimer l’ancien fichier s’il existe déjà
    for item in folder.get_items(limit=500):
        if item.name == file_name:
            print("♻️ Fichier existant trouvé, suppression avant upload...")
            item.delete()
            break

    uploaded_file = folder.upload(local_path)
    shared_link = uploaded_file.get_shared_link(access='open')
    print(f"✅ Fichier '{file_name}' envoyé sur Box : {shared_link}")
    return shared_link


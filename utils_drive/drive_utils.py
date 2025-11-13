from boxsdk import OAuth2, Client
import json
import os
import requests
import webbrowser

# ==============================
# ⚙️ Configuration générale
# ==============================
BASE_DIR = os.path.dirname(__file__)
TOKEN_FILE = os.path.join(BASE_DIR, "config.json")

CLIENT_ID = "0x9yy63d0la4qn9rwyqad2axk0581ysg"
CLIENT_SECRET = "UY3YNEBXNIV1XI8ZdHzERiuKuTzke25q"
REDIRECT_URI = "https://localhost.com"


# ==============================
# 💾 Gestion automatique des tokens
# ==============================
def store_tokens_callback(access_token, refresh_token):
    """Sauvegarde automatique des tokens mis à jour."""
    with open(TOKEN_FILE, "w") as f:
        json.dump({
            "access_token": access_token,
            "refresh_token": refresh_token
        }, f)
    print("💾 Tokens mis à jour dans config.json")


def get_tokens():
    """Charge les tokens existants ou en crée de nouveaux via OAuth2."""
    if os.path.exists(TOKEN_FILE):
        try:
            with open(TOKEN_FILE, "r") as f:
                data = json.load(f)
                if "access_token" in data and "refresh_token" in data:
                    print("✅ Tokens chargés depuis config.json")
                    return data["access_token"], data["refresh_token"]
        except Exception:
            pass

    # Aucun token valide -> démarrage du flux OAuth2
    print("\n🚀 Aucun token valide trouvé. Lancement de l'authentification OAuth2...")

    auth_url = (
        f"https://account.box.com/api/oauth2/authorize?"
        f"response_type=code&client_id={CLIENT_ID}&redirect_uri={REDIRECT_URI}"
    )

    print("\n👉 Ouvre cette URL dans ton navigateur et connecte-toi :")
    print(auth_url)
    webbrowser.open(auth_url)

    code = input("\n➡️ Copie ici le code renvoyé dans l'URL (paramètre 'code=...') : ").strip()

    token_url = "https://api.box.com/oauth2/token"
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
    }

    r = requests.post(token_url, data=data)
    if r.status_code != 200:
        raise Exception(f"Erreur OAuth2 ({r.status_code}): {r.text}")

    tokens = r.json()
    access_token = tokens["access_token"]
    refresh_token = tokens["refresh_token"]

    with open(TOKEN_FILE, "w") as f:
        json.dump({
            "access_token": access_token,
            "refresh_token": refresh_token
        }, f)

    print("\n✅ Nouveaux tokens sauvegardés dans config.json")
    return access_token, refresh_token


# ==============================
# 🔐 Initialisation du client Box
# ==============================
access_token, refresh_token = get_tokens()

oauth = OAuth2(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    access_token=access_token,
    refresh_token=refresh_token,
    store_tokens=store_tokens_callback
)

client = Client(oauth)


# ==============================
# 📤 Fonction d’upload sur Box
# ==============================
def upload_to_box(local_path, folder_id):
    """Upload un fichier dans un dossier Box (remplace s’il existe déjà)."""
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Le fichier {local_path} n'existe pas !")

    file_name = os.path.basename(local_path)
    folder = client.folder(folder_id)
    print(f"\n⬆️ Upload de '{file_name}' vers Box (folder {folder_id}) ...")

    # Supprimer l’ancien fichier s’il existe déjà
    for item in folder.get_items(limit=500):
        if item.name == file_name:
            print("♻️ Fichier existant trouvé, suppression avant upload...")
            item.delete()
            break

    # Upload du fichier
    uploaded_file = folder.upload(local_path)
    shared_link = uploaded_file.get_shared_link(access='open')
    print(f"✅ Fichier '{file_name}' envoyé sur Box : {shared_link}\n")
    return shared_link


# ==============================
# 🧪 Exemple d’utilisation directe
# ==============================
if __name__ == "__main__":
    test_file = os.path.join(BASE_DIR, "test_upload.txt")

    # Création d’un petit fichier test
    if not os.path.exists(test_file):
        with open(test_file, "w") as f:
            f.write("Ceci est un test d’upload automatique vers Box.")

    # Remplace par ton dossier Box cible
    folder_id = "349750293522"
    upload_to_box(test_file, folder_id)

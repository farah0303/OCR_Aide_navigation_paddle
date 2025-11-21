from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def corriger_texte_ai(texte):
    prompt = f"Corrige ce texte OCR en français.  Ne change rien si tout est correct et surtout n'écrit pas 'voici votre texte corrigé' juste répond par le texte corrigé directement:\n\n{texte}"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    print('correction AI effectuée')
    return response.choices[0].message.content
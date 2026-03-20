import requests
import os
from dotenv import load_dotenv

load_dotenv()

CLIENT_ID = os.getenv("ZOHO_CLIENT_ID")
CLIENT_SECRET = os.getenv("ZOHO_CLIENT_SECRET")
REFRESH_TOKEN = os.getenv("ZOHO_REFRESH_TOKEN")


def get_access_token():

    url = "https://accounts.zoho.com/oauth/v2/token"

    params = {
        "refresh_token": REFRESH_TOKEN,
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "grant_type": "refresh_token"
    }

    r = requests.post(url, params=params, timeout=30)
    r.raise_for_status()

    data = r.json()

    access_token = data.get("access_token")
    if not access_token:
        raise RuntimeError(f"No fue posible obtener access token de Zoho: {data}")

    return access_token

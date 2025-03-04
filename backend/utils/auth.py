from google.oauth2 import id_token
from google.auth.transport import requests

def verify_google_token(token, client_id):
    try:
        idinfo = id_token.verify_oauth2_token(token, requests.Request(), client_id)
        return idinfo
    except ValueError as e:
        raise ValueError(f"Invalid token: {e}")
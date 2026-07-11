import os
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pickle
import requests

SCOPES = ["https://www.googleapis.com/auth/photoslibrary.readonly"]

def authenticate():
    """
    Handles the OAuth flow: opens a browser for login the first time,
    then reuses saved credentials on future runs.
    """
    creds = None

    # token.pickle stores the access/refresh token after first login
    if os.path.exists("token.pickle"):
        with open("token.pickle", "rb") as token_file:
            creds = pickle.load(token_file)

    # If no valid credentials, log in fresh
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())  # reuse refresh token instead of full re-login
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)  # opens browser for login

        # save for next time
        with open("token.pickle", "wb") as token_file:
            pickle.dump(creds, token_file)

    return creds


def list_google_photos(creds, page_size=50):
    """
    Calls the Google Photos API to list media items (photos).
    Returns a list of dicts with photo URLs and metadata.
    """
    headers = {"Authorization": f"Bearer {creds.token}"}
    url = "https://photoslibrary.googleapis.com/v1/mediaItems"
    params = {"pageSize": page_size}

    response = requests.get(url, headers=headers, params=params)
    data = response.json()

    return data.get("mediaItems", [])


if __name__ == "__main__":
    creds = authenticate()
    photos = list_google_photos(creds)
    print(f"Found {len(photos)} photos")
    print(photos[0] if photos else "No photos found") 
from .local_folder import scan_local_folder
from .google_photos import authenticate, list_google_photos

def ingest_photos(source: str, path_or_config=None) -> list[dict]:
    """
    Unified entry point. source = 'local' or 'google'.
    Returns a consistent list of dicts: [{"id": ..., "path_or_url": ..., "source": ...}]
    """
    if source == "local":
        paths = scan_local_folder(path_or_config)
        return [{"id": p, "path_or_url": p, "source": "local"} for p in paths]

    elif source == "google":
        creds = authenticate()
        photos = list_google_photos(creds)
        return [
            {"id": p["id"], "path_or_url": p["baseUrl"], "source": "google"}
            for p in photos
        ]

    else:
        raise ValueError("source must be 'local' or 'google'")
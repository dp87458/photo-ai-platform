from pathlib import Path

# Which file types count as images
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".webp"}

def scan_local_folder(folder_path: str) -> list[str]:
    """
    Walks through a folder (and all subfolders) and returns
    the full path of every image file found.
    """
    folder = Path(folder_path)
    image_paths = []

    for file_path in folder.rglob("*"):  # rglob = recursive glob, searches subfolders too
        if file_path.suffix.lower() in VALID_EXTENSIONS:
            image_paths.append(str(file_path))

    return image_paths


if __name__ == "__main__":
    # quick manual test
    paths = scan_local_folder("./sample_photos")
    print(f"Found {len(paths)} images")
    print(paths[:5])  # show first 5
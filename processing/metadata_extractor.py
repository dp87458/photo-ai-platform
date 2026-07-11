from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from datetime import datetime


def extract_metadata(image_path: str) -> dict:
    """
    Opens an image, reads its EXIF data, and returns a clean dict
    with timestamp, GPS coordinates, and camera info.
    Missing fields default to None instead of crashing.
    """
    metadata = {
        "path": image_path,
        "timestamp": None,
        "latitude": None,
        "longitude": None,
        "camera_make": None,
        "camera_model": None,
    }

    try:
        image = Image.open(image_path)
        exif_data = image._getexif()  # returns a dict of raw EXIF tags, or None

        if not exif_data:
            return metadata  # no EXIF at all — return defaults

        # Convert numeric EXIF tag IDs into human-readable names
        readable_exif = {}
        for tag_id, value in exif_data.items():
            tag_name = TAGS.get(tag_id, tag_id)
            readable_exif[tag_name] = value

        # Timestamp
        if "DateTimeOriginal" in readable_exif:
            raw_time = readable_exif["DateTimeOriginal"]  # format: "2024:07:15 14:23:01"
            metadata["timestamp"] = datetime.strptime(raw_time, "%Y:%m:%d %H:%M:%S")

        # Camera info
        metadata["camera_make"] = readable_exif.get("Make")
        metadata["camera_model"] = readable_exif.get("Model")

        # GPS data (nested inside its own sub-dictionary)
        if "GPSInfo" in readable_exif:
            gps_data = {}
            for key, value in readable_exif["GPSInfo"].items():
                gps_tag_name = GPSTAGS.get(key, key)
                gps_data[gps_tag_name] = value

            lat = _convert_gps_to_decimal(gps_data, "GPSLatitude", "GPSLatitudeRef")
            lon = _convert_gps_to_decimal(gps_data, "GPSLongitude", "GPSLongitudeRef")
            metadata["latitude"] = lat
            metadata["longitude"] = lon

    except Exception as e:
        print(f"Could not read EXIF for {image_path}: {e}")

    return metadata


def _convert_gps_to_decimal(gps_data, coord_key, ref_key):
    """
    GPS coordinates in EXIF are stored as (degrees, minutes, seconds) tuples,
    not plain decimal numbers. This converts them into a normal decimal
    latitude/longitude value usable on a map (e.g., 19.0760° instead of 19°4'33.6").
    """
    if coord_key not in gps_data:
        return None

    degrees, minutes, seconds = gps_data[coord_key]
    decimal = float(degrees) + float(minutes) / 60 + float(seconds) / 3600

    # South and West are negative values on a standard map
    if gps_data.get(ref_key) in ["S", "W"]:
        decimal = -decimal

    return round(decimal, 6)


if __name__ == "__main__":
    # quick manual test
    result = extract_metadata("./sample_photos/download (1).jpeg")
    print(result)
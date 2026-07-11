from ingestion import ingest_photos

local_photos = ingest_photos("local", "./sample_photos")
print(local_photos)
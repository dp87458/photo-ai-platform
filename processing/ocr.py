import pytesseract
from PIL import Image

# Point pytesseract to the installed Tesseract engine.
# Adjust this path if your installer used a different location.
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def extract_text(image_path: str) -> str:
    """
    Runs OCR on an image and returns the extracted text as a plain string.
    Returns an empty string if no text is found or the image can't be read.
    """
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        print(f"OCR failed for {image_path}: {e}")
        return ""


def extract_text_with_confidence(image_path: str) -> dict:
    """
    More detailed version — also returns Tesseract's confidence score
    per detected word, so we can filter out garbage/low-confidence OCR
    results (e.g., OCR attempting to read text on a photo that isn't
    actually a document, and hallucinating random characters from noise).
    """
    try:
        image = Image.open(image_path)
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

        words = []
        confidences = []
        for i, word in enumerate(data["text"]):
            conf = int(data["conf"][i])
            if word.strip() and conf > 0:  # skip empty detections and conf=-1 (no text)
                words.append(word)
                confidences.append(conf)

        full_text = " ".join(words)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0

        return {
            "path": image_path,
            "text": full_text,
            "avg_confidence": round(avg_confidence, 2),
            "word_count": len(words),
        }
    except Exception as e:
        print(f"OCR failed for {image_path}: {e}")
        return {"path": image_path, "text": "", "avg_confidence": 0, "word_count": 0}


if __name__ == "__main__":
    from ingestion import ingest_photos
    from processing.categorization import build_category_embeddings, categorize_photo

    photos = ingest_photos("local", "./sample_photos")
    paths = [p["path_or_url"] for p in photos]

    print("--- Step 1: Categorize each photo (reusing Step 6) ---")
    category_embeddings = build_category_embeddings()

    document_like_paths = []
    for path in paths:
        result = categorize_photo(path, category_embeddings)
        if result["category"] in ("document", "receipt", "prescription"):
            document_like_paths.append(path)
            print(f"{path} → {result['category']} (will run OCR)")

    print(f"\n--- Step 2: Running OCR on {len(document_like_paths)} document-like photo(s) ---")
    for path in document_like_paths:
        result = extract_text_with_confidence(path)
        print(f"\n{result['path']}")
        print(f"  Confidence: {result['avg_confidence']}%  |  Words found: {result['word_count']}")
        print(f"  Extracted text: {result['text'][:300]}")  # first 300 chars for readability
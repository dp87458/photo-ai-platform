import torch
import open_clip
from PIL import Image
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai"
)
tokenizer = open_clip.get_tokenizer("ViT-B-32")
model.to(device)
model.eval()


def get_image_embedding(image_path: str) -> list[float]:
    """Converts an image into a 512-number vector representing its meaning."""
    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model.encode_image(image_tensor)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)

    return embedding.squeeze(0).cpu().tolist()


def get_text_embedding(text: str) -> list[float]:
    """Converts a text string into a vector in the SAME space as images."""
    tokens = tokenizer([text]).to(device)

    with torch.no_grad():
        embedding = model.encode_text(tokens)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)

    return embedding.squeeze(0).cpu().tolist()


if __name__ == "__main__":
    img_vec = get_image_embedding("./sample_photos/download (1).jpeg")
    print(f"Image embedding length: {len(img_vec)}")
    print(f"First 5 values: {img_vec[:5]}")

    text_vec = get_text_embedding("a photo of a person outdoors")
    print(f"Text embedding length: {len(text_vec)}")

def cosine_similarity(vec1, vec2):
    v1, v2 = np.array(vec1), np.array(vec2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# test: does a relevant text description score higher than an irrelevant one?
img_vec = get_image_embedding("./sample_photos/download (1).jpeg")
relevant_text = get_text_embedding("a photo of a person")
irrelevant_text = get_text_embedding("a photo of a pizza")

print("Similarity to 'a photo of a person':", cosine_similarity(img_vec, relevant_text))
print("Similarity to 'a photo of a pizza':", cosine_similarity(img_vec, irrelevant_text))
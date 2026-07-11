import os
from dotenv import load_dotenv
from groq import Groq
from rag.search import search_photos, build_photo_index

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def build_context_from_results(search_results: list[dict]) -> str:
    if not search_results:
        return "No matching photos were found."
    lines = []
    for i, result in enumerate(search_results, 1):
        filename = result["path"].split("\\")[-1].split("/")[-1]
        lines.append(f"{i}. Filename: {filename}, relevance score: {result['score']}")
    return "\n".join(lines)


def generate_rag_response(query: str, photo_index: dict, top_k: int = 5) -> dict:
    search_results = search_photos(query, photo_index, top_k=top_k)
    context = build_context_from_results(search_results)

    prompt = f"""You are a helpful photo assistant. A user searched their photo library with this query: "{query}"

Here are the photos that were actually found in their library, ranked by relevance:
{context}

Write a short, friendly 1-2 sentence response summarizing what was found.
Only reference the photos listed above — do not invent or assume photos that aren't listed.
If no photos were found, say so honestly."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )

    generated_text = response.choices[0].message.content

    return {
        "query": query,
        "answer": generated_text,
        "matching_photos": search_results,
    }

if __name__ == "__main__":
    from ingestion import ingest_photos

    photos = ingest_photos("local", "./sample_photos")
    paths = [p["path_or_url"] for p in photos]

    print("--- Building photo index ---")
    photo_index = build_photo_index(paths)

    test_queries = ["show me my dog photos", "find pasta photos", "any travel pictures?"]

    for query in test_queries:
        print(f"\n=== Query: '{query}' ===")
        result = generate_rag_response(query, photo_index, top_k=3)
        print(f"LLM Answer: {result['answer']}")
        print(f"Matching photos:")
        for p in result["matching_photos"]:
            print(f"  - {p['path']} (score: {p['score']})")
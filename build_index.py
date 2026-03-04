import os
import json
import time
import requests
import numpy as np
import xml.etree.ElementTree as ET
import re
from sentence_transformers import SentenceTransformer

ROOT = 'your-base-folder'
TERMS_PATH = ROOT + "/data/terms.json"
INDEX_DIR = ROOT + "/data/medlineplus_index"
os.makedirs(INDEX_DIR, exist_ok=True)

CORPUS_PATH = os.path.join(INDEX_DIR, "corpus.json")
EMBEDDINGS_PATH = os.path.join(INDEX_DIR, "embeddings.npy")
METADATA_PATH = os.path.join(INDEX_DIR, "metadata.json")
IDS_PATH = os.path.join(INDEX_DIR, "ids.json")

MEDLINEPLUS_URL = "https://wsearch.nlm.nih.gov/ws/query"
RATE_LIMIT_SLEEP = 0.8
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

OLLAMA_URL = "where-your-ollama-is"
OLLAMA_MODEL = "llama3.2"


def clamp_to_3_sentences(text, max_sentences=3):
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    sents = [s for s in sents if s]
    return " ".join(sents[:max_sentences]).strip()


def ollama_generate(prompt, timeout=60):
    resp = requests.post(
        OLLAMA_URL,
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=timeout
    )
    resp.raise_for_status()
    return (resp.json().get("response") or "").strip()

def fallback_summary(condition_name, aliases, category):
    prompt = (
        "Write 2-3 simple sentences for a patient describing the CONDITION. Do not mention things like: 'Here are 2-3 simple sentences' \n"
        "Use general medical knowledge. Avoid jargon.\n"
        "If useful, mention common symptoms and typical treatment/management at a high level.\n\n"
        f"CONDITION: {condition_name}\n"
        f"ALIASES: {', '.join(aliases) if aliases else 'None'}\n"
        f"CATEGORY: {category}\n"
    )
    out = ollama_generate(prompt, timeout=60)
    return clamp_to_3_sentences(out or condition_name, 3)


def validate_match(raw_summary, condition_name):
    prompt = (
        "Decide if the following text is primarily about the given condition.\n"
        "Return ONLY one token: MATCH or MISMATCH.\n\n"
        f"Condition: {condition_name}\n"
        f"Text: {raw_summary}\n"
    )
    out = ollama_generate(prompt, timeout=45).upper()
    if "MISMATCH" in out:
        return False
    if "MATCH" in out:
        return True
    return False


def simplify_summary_matched(raw_summary, condition_name):
    prompt = (
        f"Summarize the following medical description of '{condition_name}' "
        f"in 2-3 simple sentences for a patient. Do not mention things like: 'Here are 2-3 simple sentences'\n\n"
        f"{raw_summary}"
    )
    out = ollama_generate(prompt, timeout=60)
    return clamp_to_3_sentences(out or raw_summary, 3)


def simplify_summary_mismatch(raw_summary, condition_name):
    prompt = (
        "Write 2-3 simple sentences about the CONDITION using general medical knowledge.\n"
        "Incorporate only clearly relevant details from the TEXT. Do not mention things like: 'Here are 2-3 simple sentences'\n\n"
        f"CONDITION: {condition_name}\n"
        f"TEXT: {raw_summary}\n"
    )
    out = ollama_generate(prompt, timeout=60)
    return clamp_to_3_sentences(out or raw_summary, 3)


def fetch_summary(query_term):
    params = {
        "db": "healthTopics",
        "term": query_term,
        "rettype": "brief"
    }

    try:
        response = requests.get(MEDLINEPLUS_URL, params=params, timeout=10)
        response.raise_for_status()
        root = ET.fromstring(response.content)

        docs = list(root.iter("document"))
        if not docs:
            return None

        for doc in docs:
            for content in doc.iter("content"):
                if content.attrib.get("name") == "FullSummary":
                    text = "".join(content.itertext()).strip()
                    if text:
                        return text

            for content in doc.iter("content"):
                if content.attrib.get("name") == "snippet":
                    text = "".join(content.itertext()).strip()
                    if text:
                        return text

    except Exception:
        print(f"Fetch error: {query_term}")

    return None


def build_index():
    with open(TERMS_PATH, "r") as f:
        data = json.load(f)

    conditions = data["conditions"]
    print(f"Conditions: {len(conditions)}")

    corpus = {}
    metadata = []

    print("Fetching summaries")
    for i, cond in enumerate(conditions):
        cond_id = cond["id"]
        condition_name = cond["condition"]

        query_terms = cond.get("query_terms") or [condition_name]
        aliases = cond.get("aliases", [])
        category = cond.get("category", "General")

        print(f"[{i+1}/{len(conditions)}] {condition_name}")

        summary = None
        matched = False
        best_raw = None

        for term in query_terms:
            raw = fetch_summary(term)
            time.sleep(RATE_LIMIT_SLEEP)

            if not raw:
                continue

            best_raw = raw

            try:
                matched = validate_match(raw, condition_name)
            except:
                matched = True

            if matched:
                summary = raw
                break

        if summary:
            print(f"Summary {len(summary)} chars")
            print("Summary intro:", summary[:100])

            try:
                simplified = simplify_summary_matched(summary, condition_name)
            except:
                simplified = clamp_to_3_sentences(summary, 3)

            print(f"Saved {len(simplified)} chars")
            print("Simplified Summary:", simplified)

        else:
            if best_raw:
                print(f"Summary {len(best_raw)} chars")
                print("Summary intro:", best_raw[:100])

                try:
                    simplified = simplify_summary_mismatch(best_raw, condition_name)
                except:
                    simplified = clamp_to_3_sentences(best_raw, 3)

                print(f"Saved {len(simplified)} chars")
                print("Simplified Summary:", simplified)
            else:
                print("Summary: None")
                print("Fallback")

                try:
                    simplified = fallback_summary(condition_name, aliases, category)
                except:
                    simplified = condition_name

                print(f"Saved {len(simplified)} chars")
                print("Simplified Summary:", simplified)

        corpus[cond_id] = {
            "condition": condition_name,
            "summary": simplified,
            "aliases": aliases,
            "category": category
        }

        metadata.append({
            "id": cond_id,
            "condition": condition_name,
            "aliases": aliases,
            "category": category
        })

        time.sleep(RATE_LIMIT_SLEEP)

    print(f"Saving {len(corpus)} entries")
    with open(CORPUS_PATH, "w") as f:
        json.dump(corpus, f, indent=2)

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print("Building embeddings")
    model = SentenceTransformer(EMBEDDING_MODEL)

    ids = list(corpus.keys())
    with open(IDS_PATH, "w") as f:
        json.dump(ids, f, indent=2)

    texts_to_embed = []
    for cond_id in ids:
        entry = corpus[cond_id]
        text = entry["condition"] + " " + " ".join(entry.get("aliases", []))
        texts_to_embed.append(text.strip())

    embeddings = model.encode(texts_to_embed, show_progress_bar=True)
    np.save(EMBEDDINGS_PATH, embeddings)
    print(f"Embeddings {embeddings.shape}")

    print("Done")




if __name__ == "__main__":
    build_index()
from __future__ import annotations

import os
import json
import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

from utilities import clean_markdown, clamp_to_1_sentence, safe_json_extract
from utilities import INDEX_DIR

OLLAMA_URL = os.getenv("OLLAMA_URL", "where-your-ollama-is")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

MW_API_KEY = os.getenv("MW_API_KEY", "your-api-key-max-1000-requests") 
MW_URL = os.getenv("MW_URL", "https://www.dictionaryapi.com/api/v3/references/medical/json")

CORPUS_PATH = os.path.join(INDEX_DIR, "corpus.json")
IDS_PATH = os.path.join(INDEX_DIR, "ids.json")
EMBEDDINGS_PATH = os.path.join(INDEX_DIR, "embeddings.npy")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
SIM_THRESHOLD = float(os.getenv("SIM_THRESHOLD", "0.80"))



# Ollama helpers


def ollama_generate(prompt: str, timeout: int = 60) -> str:
    resp = requests.post(
        OLLAMA_URL,
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=timeout,
    )
    resp.raise_for_status()
    return (resp.json().get("response") or "").strip()




def _mw_lookup(term: str) -> Optional[str]:
    """Merriam-Webster medical lookup. Requires MW_API_KEY to be set."""
    if not MW_API_KEY:
        return None
    try:
        response = requests.get(
            f"{MW_URL}/{term}",
            params={"key": MW_API_KEY},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        if not data or isinstance(data[0], str):
            return None
        entry = data[0]
        shortdefs = entry.get("shortdef", [])
        return (shortdefs[0] if shortdefs else None)
    except Exception:
        return None


def _simplify_definition(term: str, definition: str) -> str:
    prompt = f"""Simplify this medical definition of '{term}' in one simple sentence
that a patient with no medical background can understand.
Return only the simplified sentence.

Definition: {definition}
"""
    try:
        out = ollama_generate(prompt, timeout=30)
        return clamp_to_1_sentence(out or definition)
    except Exception:
        return clamp_to_1_sentence(definition)


def _fallback_definition(term: str) -> str:
    prompt = f"""Write one simple sentence defining '{term}' for a patient.
Avoid jargon. Return only the sentence.
"""
    try:
        out = ollama_generate(prompt, timeout=30)
        return clamp_to_1_sentence(out or term)
    except Exception:
        return term


def load_runtime_index() -> Optional[Tuple[Dict[str, Any], List[str], Any, Dict[str, str], Any]]:

    if not (os.path.exists(CORPUS_PATH) and os.path.exists(IDS_PATH) and os.path.exists(EMBEDDINGS_PATH)):
        return None


    with open(CORPUS_PATH) as f:
        corpus = json.load(f)
    with open(IDS_PATH) as f:
        ids = json.load(f)
    embeddings = np.load(EMBEDDINGS_PATH)

    exact_map: Dict[str, str] = {}
    for entry in corpus.values():
        cond = entry.get("condition")
        summary = entry.get("summary")
        if cond:
            exact_map[cond.lower()] = summary
        for a in entry.get("aliases", []) or []:
            exact_map[str(a).lower()] = summary

    embed_model = SentenceTransformer(EMBEDDING_MODEL)
    return corpus, ids, embeddings, exact_map, embed_model


def _cosine_sim_scores(query_vec, mat):
    
    q = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    m = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)
    return m @ q


def _semantic_lookup(term: str, corpus, ids, embeddings, embed_model) -> Optional[str]:
    
    q = embed_model.encode([term])[0]
    sims = _cosine_sim_scores(q, embeddings)
    best_i = int(np.argmax(sims))
    best_score = float(sims[best_i])
    best_id = ids[best_i]
    if best_score >= SIM_THRESHOLD and best_id in corpus:
        return corpus[best_id].get("summary")
    return None


def lookup_term(term: str, runtime_index=None) -> str:
    t = (term or "").strip()
    if not t:
        return ""

    # semantic/exact if available
    if runtime_index is not None:
        corpus, ids, embeddings, exact_map, embed_model = runtime_index
        t_lower = t.lower()
        if t_lower in exact_map:
            return exact_map[t_lower] or ""
        sem = _semantic_lookup(t, corpus, ids, embeddings, embed_model)
        if sem:
            return sem

    # MW lookup if configured
    definition = _mw_lookup(t)
    if definition:
        return _simplify_definition(t, definition)

    return _fallback_definition(t)



# LLM extractors


def summarize_note(note_text: str) -> str:
    prompt = f"""You are helping a patient understand their clinical note.
Write a 5-6 sentence plain language summary of the following clinical note.
Avoid all medical jargon. Write as if explaining to a patient with no medical background.
Return only the summary, nothing else. Explain it to the patient, and write it in second person. Make it easy to read, and courteous.
DO NOT mention that this is a summary etc in the output.
Clinical note:
{note_text}
"""
    return ollama_generate(prompt, timeout=60)


def extract_next_steps(note_text: str) -> List[str]:
    prompt = f"""Extract all action items and follow-up instructions for the patient from this clinical note.
Return ONLY a JSON array of strings in plain language a patient can understand.
Example: ["Take Keflex 500mg twice a day for 2 weeks", "Follow up in one month if symptoms don't improve"]
ONLY include steps explicitly stated in the note. If none, return []. Remember, in medicine, we need to be as accurate as possible.
Clinical note:
{note_text}
"""
    raw = ollama_generate(prompt, timeout=60)
    parsed = safe_json_extract(raw)
    return parsed if isinstance(parsed, list) else []


def extract_medical_terms(note_text: str) -> List[str]:
    prompt = f"""You are a medical NLP assistant.
Extract all medical terms, drug names, conditions, and clinical abbreviations from the following clinical note.
Ignore duplicate or identical terms. You can also ignore common terms that any layperson will know.
Only provide medical terms, drug names, conditions, and clinical abbreviations that are in the text.
Remember, in medicine, it is important to be as accurate as possible. 
Return ONLY a JSON array of strings.

Clinical note:
{note_text}
"""
    raw = ollama_generate(prompt, timeout=60)
    parsed = safe_json_extract(raw)
    if isinstance(parsed, list):
        seen = set()
        out = []
        for x in parsed:
            if not isinstance(x, str):
                continue
            k = x.strip().lower()
            if k and k not in seen:
                seen.add(k)
                out.append(x.strip())
        return out
    return []
 
def generate_doctors_reasoning(note_text: str, summary: str, next_steps: list[str], abnormal_vitals: dict, timeline: list[dict]) -> str:
    
    prompt = f"""You are explaining a clinician's reasoning to a patient.

    Rules:
    - Do NOT diagnose.
    - Do NOT add new facts. Only use what is present in the provided information.
    - If information is missing or unclear, say so plainly.
    - Explain in simple language with a calm, helpful tone. Do not accuse the patient of anything.
    - Keep it to 3-4 short bullet points max, be as concise as possible.
    Remember, in medicine, accuracy is extremely important.

    Clinical note (cleaned):
    {note_text}

    Patient-friendly summary:
    {summary}

    Abnormal vitals (if any):
    {json.dumps(abnormal_vitals, ensure_ascii=False)}

    Timeline items (if any):
    {json.dumps(timeline, ensure_ascii=False)}

    Return ONLY the bullet points (markdown list).
    """
    return ollama_generate(prompt, timeout=60)

def build_glossary(terms: List[str], runtime_index=None) -> Dict[str, str]:
    glossary: Dict[str, str] = {}
    for term in terms:
        glossary[term] = lookup_term(term, runtime_index=runtime_index)
    return glossary


def create_text_elements(note_markdown: str, runtime_index=None, context: dict | None = None) -> dict:
    note_text = clean_markdown(note_markdown)

    summary = summarize_note(note_text)
    next_steps = extract_next_steps(note_text)

    terms = extract_medical_terms(note_text)
    glossary = build_glossary(terms, runtime_index=runtime_index)

    # context from visualizations (optional)
    abnormal_vitals = (context or {}).get("abnormal_vitals", {})
    timeline = (context or {}).get("timeline", [])

    doctors_reasoning = generate_doctors_reasoning(
        note_text=note_text,
        summary=summary,
        next_steps=next_steps,
        abnormal_vitals=abnormal_vitals,
        timeline=timeline,
    )

    return {
        "note_text": note_text,
        "summary": summary,
        "next_steps": next_steps,
        "glossary": glossary,
        "terms": terms,
        "doctors_reasoning": doctors_reasoning,
    }

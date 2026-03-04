
from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Optional

import requests
import plotly.graph_objects as go

from utilities import clean_markdown, safe_json_extract, load_json_file, build_normal_ranges, check_abnormal_vitals
from utilities import DATA_DIR

OLLAMA_URL = os.getenv("OLLAMA_URL", "where-your-ollama-is")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")


NORMAL_RANGES_PATH = os.path.join(DATA_DIR, "normal_ranges.json")


def ollama_generate(prompt: str, timeout: int = 60) -> str:
    resp = requests.post(
        OLLAMA_URL,
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=timeout,
    )
    resp.raise_for_status()
    return (resp.json().get("response") or "").strip()


def extract_vitals_and_timeline(note_markdown: str) -> Dict[str, Any]:
    note_text = clean_markdown(note_markdown)

    prompt = f"""Extract any vitals and timeline events from this clinical note. 
Only use what is present in the text, remember, in medicine accuracy is vital.
For timeline, only provide events that need medical intervention, appointment follow-up or tests. 
Do not provide events from the past. Only provide events that will or need to happen.
If an event happened during the consultation, ignore it.

Return ONLY a JSON object with this exact structure, use null if not found:
{{
    "vitals": {{
        "blood_pressure": {{"value": null, "unit": "mmHg"}},
        "heart_rate": {{"value": null, "unit": "bpm"}},
        "temperature": {{"value": null, "unit": "F"}},
        "oxygen_saturation": {{"value": null, "unit": "%"}},
        "weight": {{"value": null, "unit": "lbs"}},
        "height": {{"value": null, "unit": "in"}}
    }},
    "timeline": [
        {{"event": null, "time": null}}
    ]
}}

Clinical note:
{note_text}
"""
    raw = ollama_generate(prompt, timeout=60)
    parsed = safe_json_extract(raw)
    return parsed if isinstance(parsed, dict) else {"vitals": {}, "timeline": []}


def plot_abnormal_vitals(abnormal: Dict[str, Any]) -> List[go.Figure]:
    figures: List[go.Figure] = []

    for vital_name, reading in (abnormal or {}).items():
        value = reading.get("value")
        normal_range = reading.get("normal_range") or ""
        unit = reading.get("unit") or ""
        label = reading.get("label") or vital_name

        
        range_part = (normal_range.split(" ")[0] if normal_range else "")
        try:
            low, high = map(float, range_part.split("-"))
        except Exception:
            low, high = None, None

        
        if isinstance(value, dict):
            for sub_key, sub_val in value.items():
                fig = go.Figure()
                fig.add_trace(go.Bar(name="Your Value", x=[f"{label} ({sub_key})"], y=[sub_val]))
                if high is not None:
                    fig.add_trace(go.Bar(name="Normal Max", x=[f"{label} ({sub_key})"], y=[high], opacity=0.5))
                fig.update_layout(
                    title=f"{label} ({sub_key}) — {unit}".strip(),
                    barmode="group",
                    yaxis_title=unit,
                    plot_bgcolor="white",
                )
                figures.append(fig)
        else:
            fig = go.Figure()
            fig.add_trace(go.Bar(name="Your Value", x=[label], y=[value]))
            if high is not None:
                fig.add_trace(go.Bar(name="Normal Max", x=[label], y=[high], opacity=0.5))
            fig.update_layout(
                title=f"{label} — {unit}".strip(),
                barmode="group",
                yaxis_title=unit,
                plot_bgcolor="white",
            )
            figures.append(fig)

    return figures


def build_timeline_flowchart(timeline: List[Dict[str, Any]]):
    try:
        import graphviz
    except Exception:
        return None

    dot = graphviz.Digraph(format="png")
    dot.attr(rankdir="TB", size="8,10")
    dot.attr("node", shape="box", style="filled", fillcolor="#e8f4f8", fontname="Helvetica", fontsize="11")
    dot.attr("edge", color="#555555")

    for i, item in enumerate(timeline or []):
        event = item.get("event") or "Unknown event"
        time = item.get("time")
        label = f"{event}\n({time})" if time and str(time).lower() != "null" else event
        dot.node(str(i), label)
        if i > 0:
            dot.edge(str(i - 1), str(i))
    return dot


def create_visualizations(note_markdown: str) -> Dict[str, Any]:
    data = extract_vitals_and_timeline(note_markdown)

    vitals = data.get("vitals") or {}
    timeline = data.get("timeline") or []

    normal_ranges_data = load_json_file(NORMAL_RANGES_PATH, default={"vitals": []})
    normal_ranges = build_normal_ranges(normal_ranges_data)

    abnormal = check_abnormal_vitals(vitals, normal_ranges)
    figures = plot_abnormal_vitals(abnormal)

    flowchart = build_timeline_flowchart(timeline)

    return {
        "vitals": vitals,
        "timeline": timeline,
        "abnormal_vitals": abnormal,
        "figures": figures,
        "timeline_flowchart": flowchart,  
    }

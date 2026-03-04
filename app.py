from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, Optional

import streamlit as st
import pandas as pd
import base64

from create_text_elements import create_text_elements, load_runtime_index
from create_visualizations import create_visualizations
from utilities import markdown_to_pdf_bytes, ASSETS_DIR, BASE_DIR

st.set_page_config(page_title="CiceroNote 🐕", layout="wide")

LOADING_GIF_PATH = os.getenv(
    "LOADING_GIF_PATH",
    os.path.join(ASSETS_DIR, "loading.gif")
)


st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
        }

        h1, h2, h3 {
            color: #1a1a1a;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
    </style>
    """,
    unsafe_allow_html=True
)


def show_loading_gif(path: str, width: int = 420) -> None:
    if not os.path.exists(path):
        st.spinner("Working…")
        return

    try:
        with open(path, "rb") as f:
            data = f.read()

        b64 = base64.b64encode(data).decode("utf-8")

        st.markdown(
            f"""
            <div style="display:flex; justify-content:center; align-items:center; padding: 10px 0;">
                <img src="data:image/gif;base64,{b64}" width="{width}" />
            </div>
            """,
            unsafe_allow_html=True,
        )
    except Exception:
        st.spinner("Working…")


def go(page: str) -> None:
    st.session_state.page = page
    st.rerun()



def build_report_markdown(text_out: Dict[str, Any], viz_out: Dict[str, Any]) -> str:

    summary = (text_out.get("summary") or "").strip()
    next_steps = text_out.get("next_steps") or []
    glossary = text_out.get("glossary") or {}

    abnormal = viz_out.get("abnormal_vitals") or {}
    timeline = viz_out.get("timeline") or []

    lines = []

    lines.append("# Patient Report")
    lines.append("")

    lines.append("## Summary")
    lines.append(summary or "—")
    lines.append("")

    lines.append("## Next Steps")

    if next_steps:
        for s in next_steps:
            lines.append(f"- {s}")
    else:
        lines.append("- No clear next steps found.")

    lines.append("")

    reasoning = (text_out.get("doctors_reasoning") or "").strip()

    lines.append("## Doctor’s Reasoning")
    lines.append(reasoning or "—")
    lines.append("")

    if abnormal:
        lines.append("## Abnormal Vitals")

        for k, v in abnormal.items():
            label = v.get("label") or k
            value = v.get("value")
            unit = v.get("unit") or ""
            nr = v.get("normal_range") or ""

            lines.append(f"- **{label}**: {value} {unit} (normal: {nr})")

        lines.append("")

    if timeline and any((t.get("event") or "").strip() for t in timeline):

        lines.append("## Timeline")

        for item in timeline:
            ev = item.get("event") or "—"
            tm = item.get("time") or "—"
            lines.append(f"- {ev} — {tm}")

        lines.append("")

    if glossary:

        lines.append("## Glossary")

        for term in sorted(glossary.keys(), key=lambda x: str(x).lower()):

            lines.append(f"### {term}")
            lines.append((glossary.get(term) or "—").strip())
            lines.append("")

    return "\n".join(lines).strip() + "\n"


# Session state
if "page" not in st.session_state:
    st.session_state.page = "input"

if "note_md" not in st.session_state:
    st.session_state.note_md = ""

if "text_out" not in st.session_state:
    st.session_state.text_out = None

if "viz_out" not in st.session_state:
    st.session_state.viz_out = None

if "error" not in st.session_state:
    st.session_state.error = None


# Sidebar
with st.sidebar:

    st.header("Controls")

    st.divider()

    if st.button("Reset session"):

        for k in ["page", "note_md", "text_out", "viz_out", "error"]:
            if k in st.session_state:
                del st.session_state[k]

        st.rerun()


@st.cache_resource
def _get_runtime_index():
    return load_runtime_index()


# Page: INPUT
if st.session_state.page == "input":

    st.title("CiceroNote 🐕")

    uploaded = st.file_uploader(
        "Upload a clinical note (.md or .txt)",
        type=["md", "txt"]
    )

    if uploaded is not None:
        st.session_state.note_md = uploaded.getvalue().decode(
            "utf-8",
            errors="replace"
        )

    note = st.text_area(
        "Paste clinical note (markdown)",
        height=280,
        value=st.session_state.note_md
    )

    c1, c2, c3 = st.columns([1, 1, 2])

    with c1:

        can_run = bool(note.strip())

        if st.button("Generate report", type="primary", disabled=not can_run):

            st.session_state.note_md = note
            st.session_state.text_out = None
            st.session_state.viz_out = None
            st.session_state.error = None

            go("loading")

    with c2:

        if st.session_state.text_out and st.session_state.viz_out:

            if st.button("View last report"):
                go("report")

    with c3:
        st.caption("AI-generated, use with caution.")

    if st.session_state.error:
        st.error(st.session_state.error)


# Page: LOADING
elif st.session_state.page == "loading":

    st.title("Generating report…")

    show_loading_gif(LOADING_GIF_PATH, width=420)

    status = st.status("Starting…", expanded=True)

    try:

        note_md = st.session_state.note_md

        if not note_md or not note_md.strip():
            st.session_state.error = "No note content found."
            go("input")

        status.update(label="Loading glossary index…", state="running")
        runtime_index = _get_runtime_index()

        status.update(label="Extracting vitals/timeline…", state="running")
        viz_out = create_visualizations(note_md)

        status.update(label="Creating summary + glossary + reasoning…", state="running")
        text_out = create_text_elements(
            note_md,
            runtime_index=runtime_index,
            context=viz_out
        )

        st.session_state.text_out = text_out
        st.session_state.viz_out = viz_out

        status.update(label="Done.", state="complete")

        go("report")

    except Exception as e:

        st.session_state.error = f"Run failed: {type(e).__name__}: {e}"

        status.update(label="Failed.", state="error")

        st.button("← Back", on_click=lambda: go("input"))


# Page: REPORT
elif st.session_state.page == "report":

    st.title("Report")

    top1, top2 = st.columns([1, 5])

    with top1:
        if st.button("← Back"):
            go("input")

    text_out: Optional[Dict[str, Any]] = st.session_state.text_out
    viz_out: Optional[Dict[str, Any]] = st.session_state.viz_out

    if not text_out or not viz_out:

        st.warning("No report found. Generate one first.")

        if st.button("Go to input"):
            go("input")

        st.stop()

    tab1, tab2, tab3 = st.tabs(
        ["📁 Summary & Next Steps", "📁 Glossary", "📁 Visualizations"]
    )

    with tab1:

        st.subheader("Summary")
        st.markdown(text_out.get("summary") or "—")

        st.divider()

        st.subheader("Next Steps")

        steps = text_out.get("next_steps") or []

        if steps:
            for s in steps:
                st.write("•", s)
        else:
            st.write("No clear next steps found.")

        st.divider()

        st.subheader("Doctor’s Reasoning")

        reasoning = text_out.get("doctors_reasoning") or ""
        st.markdown(reasoning if reasoning.strip() else "—")

        report_md = build_report_markdown(text_out, viz_out)

        pdf_bytes = markdown_to_pdf_bytes(report_md, title="Patient Report")

        fname = f"patient_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

        st.download_button(
            "Download PDF",
            data=pdf_bytes,
            file_name=fname,
            mime="application/pdf",
        )

        with st.expander("Report preview (markdown)"):
            st.code(report_md, language="markdown")
    
    with tab2:
        st.subheader("Glossary")
        glossary = text_out.get("glossary") or {}

        if not glossary:
            st.info("No glossary terms extracted.")
        else:
            df = pd.DataFrame(
                [{"term": k, "definition": v} for k, v in glossary.items()]
            ).sort_values("term", key=lambda s: s.str.lower())

            q = st.text_input("Search glossary", "")
            if q.strip():
                mask = df["term"].str.contains(q, case=False, na=False) | df["definition"].str.contains(q, case=False, na=False)
                df = df[mask]

            st.dataframe(df, use_container_width=True, hide_index=True)

            with st.expander("Glossary cards"):
                for term, definition in sorted(glossary.items(), key=lambda kv: kv[0].lower()):
                    with st.expander(term):
                        st.write(definition)

    with tab3:
        st.subheader("Abnormal vitals")
        abnormal = viz_out.get("abnormal_vitals") or {}

        if not abnormal:
            st.success("No abnormal vitals detected (based on available normal ranges).")
        else:
            rows = []
            for k, v in abnormal.items():
                rows.append({
                    "vital": v.get("label") or k,
                    "value": v.get("value"),
                    "unit": v.get("unit"),
                    "normal_range": v.get("normal_range"),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("Charts")
        figs = viz_out.get("figures") or []

        if not figs:
            st.info("No charts to show.")
        else:
            for fig in figs:
                st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("Timeline")
        timeline = viz_out.get("timeline") or []

        if timeline and any((t.get("event") or "").strip() for t in timeline):
            st.dataframe(pd.DataFrame(timeline), use_container_width=True, hide_index=True)
        else:
            st.info("No timeline items found.")

        dot = viz_out.get("timeline_flowchart")
        if dot is not None and timeline and len(timeline) > 1:
            st.subheader("Timeline flowchart")
            st.graphviz_chart(dot.source)
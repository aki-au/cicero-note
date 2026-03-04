# CiceroNote 

> Named after my dog Cicero :)

CiceroNote is a local, privacy-first clinical note explainer. 

It takes a SOAP-based clinical note and transforms it into a patient-friendly card: a plain-language summary, next steps, a glossary of medical terms, an abnormal vitals chart, and a clinical journey flowchart.

---

## What it does

- Extracts medical terms from clinical notes and looks them up via a local corpus that you create beforehand using MedLine and Merriam-Webster Medical Dictionary API
- Simplifies definitions into plain language 
- Summarizes the note
- Extracts next steps and patient action items
- Flags abnormal vitals and visualizes them against normal ranges
- Builds a clinical journey flowchart from the sequence of events in the note

---
 ## Why did I build this?

I built CiceroNote after seeing my grandparents come home from doctor visits, not 100% sure about what had actually happened during the appointment. I wanted to create a prototype to bridge that gap!

---
## Folder Setup
```
cicero-note/
│
├── app.py # Streamlit application (UI + pipeline orchestration)
├── create_text_elements.py # summary, next steps, glossary generation, doctor reasoning
├── create_visualizations.py # abnormal vitals detection, charts, timeline extraction
├── utilities.py # shared helpers (markdown cleaning, PDF export, paths)
├── build_index.py # builds the MedlinePlus semantic lookup index
│
├── data/
│ ├── terms.json # seed list of medical conditions
│ ├── normal_ranges.json # reference ranges for vital signs
│ ├── notes/ # example clinical notes (SOAP format)
│ └── medlineplus_index/ # local semantic glossary index
│ ├── corpus.json
│ ├── embeddings.npy
│ └── metadata.json
│
└── requirements.txt
```
---
## How to set up!
### 1. Install dependencies

```bash
pip install -r requirements.txt
brew install graphviz
```


### 2. Install and start Ollama

Download Ollama from [ollama.com](https://ollama.com), then pull the model used by the app:

```bash
ollama pull llama3.2
ollama serve
```


### 3. Add your Merriam-Webster Medical Dictionary API key

Create a free key at [dictionaryapi.com](https://dictionaryapi.com), then add it to `create_text_elements.py`:

```python
MW_API_KEY = "get your api keyy - it's free!"
```

This API is used as a fallback when glossary terms cannot be found in the local MedlinePlus index, that we create ourselves/


### 4. Build the local glossary index

Run once to create the semantic lookup index:

```bash
python build_index.py
```

This generates the embeddings used for fast local medical-term lookups.


### 5. Run the app

```bash
streamlit run app.py
```

The app will launch locally, trust ;)

---
## Stack

- **LLM** — Ollama (`llama3.2`) running locally
- **Embeddings** — `sentence-transformers` (`all-MiniLM-L6-v2`)
- **Medical Dictionary** — MedlinePlus Web Service + Merriam-Webster Medical API
- **Visualizations** — Plotly and Graphviz
- **UI** — Streamlit

---

## Notes

- Tested on MacBook Pro M4 (16GB) — runs fully offline, really comfortably after setup
  Note: you might need to make sure Graphviz is set up for Windows if you are not using a Macbook.
- No real patient data — demo only
- In production, this would integrate with MedlinePlus Connect via ICD codes and would be checked to ensure we are following HIPAA guidelines.

---

## Future Work
- I would like to try different domains (legal, financial)
- Support for radiology reports and discharge summaries would also be amazing

Let's collaborate!
---

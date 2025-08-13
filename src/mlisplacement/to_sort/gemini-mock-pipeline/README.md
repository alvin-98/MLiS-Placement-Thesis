# Gemini Markdown → JSON Pipeline

Simple CLI utility that sends a markdown document to the Google Gemini API and returns a structured JSON representation.

## Quick Start

```bash
# install deps (ideally in a venv)
pip install -r requirements.txt

# run – print JSON to stdout
do_script -m /path/to/doc.md  # or: python pipeline.py doc.md

# save to file
python pipeline.py doc.md -o doc.json
```

Set your API key via a `.env` file, environment variable, or CLI flag.

```bash
# .env file in project root:
# GEMINI_API_KEY=your_key_here

# or export manually
export GEMINI_API_KEY="your_key"

python pipeline.py doc.md
```

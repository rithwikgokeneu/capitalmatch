# CapitalMatch

CapitalMatch helps first-time founders raising a seed round find the real funding deals most like their own, so they can target the kind of investors who back companies like theirs.

- `app.py`: the Streamlit interface and the matching logic (rule-based shortlist, AI analyst, answer checks), the same steps as the n8n workflow from Assignment 4
- `data/capitalmatch_data.csv`: verified funding records from Assignment 3 (Crunchbase history, SEC Form D filings, funding news)

The Gemini API key is stored in Streamlit secrets as `GEMINI_API_KEY` and is never committed to this repository.

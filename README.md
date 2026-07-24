# Search Query Intent Classifier

A small, real, working system that looks at a search query and decides: does the person want to **go somewhere** (like typing "youtube"), or do they want an **answer** (like "compare iPhone 17 vs 16")? If it's confident the person just wants to go somewhere, it skips a costly AI-generated response and goes straight to a result.

**Live demo:** https://search-intent-classifier-5sopogvnwwunaqtknbnymz.streamlit.app

This started as a self-initiated case study on why AI-powered search products (like Google's AI Mode) sometimes run a full AI response even for simple, obvious searches — and turned into an end-to-end, tested, deployed system that demonstrates the actual fix.

---

## Table of contents
- [The problem](#the-problem)
- [The solution](#the-solution)
- [Architecture](#architecture)
- [Tech stack](#tech-stack)
- [Dataset](#dataset)
- [Model performance](#model-performance)
- [Hypotheses — tested honestly](#hypotheses--tested-honestly)
- [Limitations](#limitations)
- [Running it locally](#running-it-locally)
- [Project structure](#project-structure)
- [What I'd improve next](#what-id-improve-next)

---

## The problem

Large AI search products try to generate an AI answer for almost every query. But a lot of queries don't need one — the person already knows exactly where they want to go ("youtube", "gmail"). Running a full AI response for these costs real compute, adds latency, and doesn't help the user any more than a direct result would.

**Honest update, found while researching this:** Google's actual system already handles most of this well — reported activation rates for AI Mode on clearly navigational/branded queries are below 10%. So this project isn't "fixing a bug Google missed." It's a hands-on demonstration of the real technique large-scale AI products use to control this cost — built, tested, and evaluated from scratch.

## The solution

A hybrid system with two layers, so each part does the job it's actually good at:

1. **A known-sites lookup**, using fuzzy string matching, catches obvious cases instantly and cheaply — including typos ("goggle" → "google").
2. **A machine learning classifier** (TF-IDF + Logistic Regression) handles everything else, with a confidence cutoff so it only acts on a "Navigational" guess when it's genuinely sure — defaulting safely to "Informational" otherwise.

A Redis cache sits in front of both, so repeated queries return instantly without recomputation. Every classification — cached or fresh — is logged to a Postgres database for a complete, honest record of real usage.

## Architecture

```
Query
  │
  ▼
Redis cache ──(hit)──► Return cached result
  │ (miss)
  ▼
Known-sites lookup (fuzzy match) ──(match)──► Navigational
  │ (no match)
  ▼
TF-IDF + Logistic Regression model
  │
  ▼
Confidence ≥ 90% and predicted "Navigational"?
  ├─ Yes → Navigational
  └─ No  → Informational (safe default)
  │
  ▼
Save to Redis + log to Postgres
  │
  ▼
Return result
```

## Tech stack

| Part | Tool | Why |
|---|---|---|
| Data exploration & training | Python, Google Colab | Fast iteration, no setup |
| Data handling | pandas | Cleaning and sampling |
| Text → features | TF-IDF (scikit-learn) | Simple, fast, explainable |
| Classifier | Logistic Regression (scikit-learn) | Interpretable — every decision can be explained |
| Typo handling | rapidfuzz | Fuzzy string matching against known site names |
| Cache | Redis (Redis Cloud, free tier) | Instant repeated-query lookups, reachable from the deployed app |
| Persistent logging | PostgreSQL (Neon, free tier) | Full, honest record of every query and prediction |
| Demo app | Streamlit, deployed on Streamlit Community Cloud | Public, testable by anyone |
| Secrets | python-dotenv locally, Streamlit Secrets in production | Credentials never committed to git |

## Dataset

Built from **[ORCAS-I](https://researchdata.tuwien.ac.at/records/pp7xz-n9a06)** — real, anonymized Bing search queries, labeled by intent (Informational / Navigational / Transactional), released by TU Wien under CC-BY 4.0.

- **Training sample:** 4,000 queries per class (12,000 total), balanced by design — not proportional to real-world frequency, so the model gets a fair chance to learn all three categories rather than defaulting to the majority class.
- **Test set:** a separate, human-verified "gold" subset (~1,000 rows), held out and untouched until final evaluation.

## Model performance

On an internal 20% validation split (not the gold set):

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Informational | 0.67 | 0.83 | 0.74 |
| Navigational | 0.78 | 0.61 | 0.68 |
| Transactional | 0.99 | 0.99 | 0.99 |
| **Overall accuracy** | | | **0.81** |

The model alone is noticeably weaker at recalling Navigational queries — this is explained and addressed directly below in the hypotheses section.

## Hypotheses — tested honestly

Same principle used throughout this project (and a companion project, DarkStorePricing): write the hypothesis before testing, then report the real result — even when it doesn't confirm.

| # | Hypothesis | Result | Verdict |
|---|---|---|---|
| **H1** | Bare single-word brand queries ("youtube", "gmail") will be classified Navigational with >90% confidence. | Checked the training data directly: an exact bare-word match appeared only 0–1 times per brand in the entire 12,000-row sample. The model was never realistically given a chance to learn this pattern; confidence stayed low (0.42–0.55) for most. | **Killed** — with a proven, data-scarcity cause. This directly justified the known-sites lookup layer. |
| **H2** | Typos will hurt model accuracy, but fuzzy-matching will recover it. | Model alone on 10 generated typos: **0/10 correct (0%)**. Hybrid system (lookup + fuzzy matching): **10/10 correct (100%)**. | **Confirmed** — scoped to typos of names already in the lookup list. |
| **H3** | A high confidence threshold keeps wrong guesses near zero; genuinely ambiguous queries score lower rather than confidently wrong. | Part A: 0/7 wrong Navigational guesses passed the 90% cutoff on brand+intent queries (e.g. "netflix subscription cost"). Part B: ambiguous words (apple, target, chrome, mercury, orange) scored a mixed 0.47–0.64, vs. 0.78–0.81 for clear brands. | **Confirmed** — with one honest limitation (see below). |

## Limitations

Stated plainly, not hidden:

- **The known-sites list is small and self-curated** (~30 names) for this student project — a production system would need a much larger, maintained list.
- **The lookup matches on spelling, not meaning.** A genuinely ambiguous word like "amazon" (river, rainforest, company) still gets forced to 100% confidence if it's on the list — even though the ML model itself was appropriately uncertain about it. This is a real trade-off between speed/certainty and true understanding.
- **No real click-behavior data.** A production system would strengthen classification using what people actually click after searching — this project only uses query text, since click data isn't publicly available at this scale.
- **The typo-recovery result (H2) is scoped** to typos of names already in the lookup list, not typos of unknown or unlisted sites.

## Running it locally

```bash
git clone https://github.com/YOUR-USERNAME/search-intent-classifier.git
cd search-intent-classifier
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root (see `.env.example` for the required keys — Redis and Postgres credentials, never committed to this repo).

```bash
streamlit run app.py
```

## Project structure

```
search-intent-classifier/
├── app.py                   # Streamlit demo app
├── requirements.txt
├── .env.example              # placeholder keys — copy to .env and fill in your own
├── models/                   # trained artifacts (model, vectorizer, known-sites list)
├── notebooks/
│   └── model_training.ipynb  # full data exploration, training, and hypothesis testing
├── src/
│   └── classifier.py         # core logic: lookup, model, Redis cache, Postgres logging
└── test_classifier.py        # sanity check for classify_query()
```

## What I'd improve next

- Use real click-behavior data instead of query text alone, for a stronger, more realistic classifier.
- Grow the known-sites list significantly, and consider a maintained, larger gazetteer instead of a hand-picked one.
- Run a real user study (in progress) to measure how often the model's guess matches what a person actually wanted, beyond the offline metrics above.
- Explore whether Docker + a self-hosted deployment (instead of Streamlit Community Cloud) would be worth it if this needed to scale beyond a demo.

---

Built by Arpita as a self-initiated PM/analyst/ML case study.
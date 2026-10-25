# VoiceOps 2026 Capstone Report

## Project

ZentroMart Voice Support Agent answers customer questions from policy and help documents, retrieves product catalog rows, retains conversation context, checks identity before sharing order data, creates escalation tickets, and accepts and returns speech through Gnani.

## Build by step

**Step 1 - LLM baseline.** Groq-hosted `qwen/qwen3.8-27b` answers the supplied questions using a support-assistant prompt without retrieved company documents. Run `python run.py evaluate` to record its practice score.

**Step 2 - Retrieval augmented generation.** The index splits Markdown files at their stable section IDs and includes product catalog rows. It builds TF-IDF vectors, scores cosine similarity, ranks current policies above lower-authority sources, and penalizes superseded policies. The same questions compare prompt-only and retrieved answers; the detailed results include answer facts and source IDs.

**Step 3 - Conversation and tools.** The agent retains the last turns and verified order data for the current session. The supplied lookup tool gates order details behind a phone suffix or delivery pincode, with one retry. Ticket creation is confirmed only after a successful tool result. The agent follows the fixed 20 October 2026 date and business-day calendar.

**Step 4 - Voice.** Gnani Vachana transcribes customer audio and synthesizes replies with a named timbre model. The batch command stores transcripts, replies, audio files, and retrieved sections. No other speech service is used.

## Evaluation

| Measure | Result |
| --- | --- |
| Prompt-only fact accuracy | Run evaluator |
| RAG fact accuracy | Run evaluator |
| Retrieval evidence hit rate | Run evaluator |
| Unsupported claims in RAG answers | Run evaluator |

Run `python run.py evaluate` and `python run.py evaluate-conversations` with the API key configured. The first reports prompt-only and RAG accuracy, retrieval hit rate, and forbidden claims; the second scores required facts, source hits, and expected tool calls over the multi-turn cases. Regenerate the PDF with `python make_report.py` to fill the results table from the evaluation JSON saved under `outputs/`.

## Lessons and next improvements

The hardest cases are policy precedence, date-dependent return windows, indirect follow-ups, Hindi or Hinglish wording, and keeping unverified order data private. Section IDs make retrieval auditable, while the separate order tool prevents the language model from inventing customer records. Before submission, review low-scoring cases in the JSON results, tune retrieval using those misses, and capture the required short voice demo using a self-recorded question and a real Gnani response.

## Reproducibility

Install `requirements.txt`, copy `.env.example` to `.env`, add your Groq and Gnani API keys, and run `python run.py batch <audio-folder> <output-folder>`. Keep the dataset in the project folder or set `CAPSTONE_DATASET_DIR`. Do not include `.env`, credentials, or a copy of the dataset in the submission ZIP.

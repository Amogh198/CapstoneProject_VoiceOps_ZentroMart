# Two-minute demo recording plan

Record only your own voice. Put each spoken turn in a separate WAV, MP3, M4A, FLAC, or OGG file. Prefix filenames with `01_`, `02_`, etc. so the batch runner processes them in order.

1. Ask a policy question in English or Hindi and show the answer plus its retrieved section ID.
2. Ask a natural follow-up such as “How long will that take?” to demonstrate remembered context.
3. Mention an order number and let the assistant ask for one verification value. Provide the value from a practice case; show that the order details are only shared after the check.
4. Include a safety issue or request a human colleague to demonstrate ticket creation and handover.

Run `python run.py batch demo_audio demo_output --conversation`. Show the original audio turns, the transcript, the spoken WAV replies, and `results.jsonl`. Keep the final video under two minutes. Do not display API keys or verification values in the recording.

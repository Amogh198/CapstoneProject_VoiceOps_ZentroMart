from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "team_name_voiceops.zip"
FILES = [
    "README.md", ".env.example", "requirements.txt", "run.py", "report.pdf", "demo.mp4",
    "demo_script.md", "make_report.py", "make_submission.py", "src/__init__.py", "src/settings.py",
    "src/retrieval.py", "src/llm.py", "src/agent.py", "src/voice.py",
    "src/evaluate.py", "tools/__init__.py", "tools/tools.py",
    "sample_outputs/README.md",
]


def build():
    required_results = [ROOT / "outputs" / "evaluation.json", ROOT / "outputs" / "conversation_evaluation.json"]
    missing_results = [str(path.relative_to(ROOT)) for path in required_results if not path.is_file()]
    if missing_results:
        raise FileNotFoundError("Run both evaluators before creating the final ZIP: " + ", ".join(missing_results))
    report = ROOT / "report.pdf"
    if report.exists() and report.stat().st_mtime <= max(path.stat().st_mtime for path in required_results):
        raise RuntimeError("Regenerate report.pdf with python make_report.py after completing both evaluations.")
    sample_manifest = ROOT / "sample_outputs" / "results.jsonl"
    sample_audio = list((ROOT / "sample_outputs").glob("*_reply.wav"))
    if not sample_manifest.is_file() or not sample_audio:
        raise FileNotFoundError("Run a recorded voice example into sample_outputs/ before creating the final ZIP.")
    missing = [name for name in FILES if not (ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError("Complete the report and two-minute demo first; missing: " + ", ".join(missing))
    with ZipFile(OUTPUT, "w", ZIP_DEFLATED) as archive:
        for name in FILES:
            archive.write(ROOT / name, arcname=name)
        for result_path in required_results:
            archive.write(result_path, arcname=f"sample_outputs/{result_path.name}")
        for output in [sample_manifest, *sample_audio]:
            archive.write(output, arcname=f"sample_outputs/{output.name}")
    print(f"Created {OUTPUT}")
    print("Dataset CSVs, knowledge-base files, credentials, environments, and generated outputs were excluded.")


if __name__ == "__main__":
    build()

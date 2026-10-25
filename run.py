import argparse
import json
import sys
from pathlib import Path

from src.agent import SupportAgent
from src.evaluate import evaluate, evaluate_conversations
from src.settings import dataset_root
from src.voice import synthesize, transcribe


AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}


def process_audio_folder(input_dir, output_dir, language_code=None, conversation=False):
    source = Path(input_dir)
    target = Path(output_dir)
    if not source.is_dir():
        raise FileNotFoundError(f"Audio input folder does not exist: {source}")
    files = sorted(path for path in source.iterdir() if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES)
    if not files:
        raise FileNotFoundError(f"No supported audio files found in {source}")
    target.mkdir(parents=True, exist_ok=True)
    manifest = target / "results.jsonl"
    shared_agent = SupportAgent() if conversation else None
    with manifest.open("w", encoding="utf-8") as output:
        for audio in files:
            agent = shared_agent or SupportAgent()
            heard = transcribe(audio, language_code)
            if not heard:
                reply = "Sorry, I didn't catch that. Could you say it again?"
                sources = []
            else:
                answer = agent.respond(heard)
                reply = answer["reply"]
                sources = answer["sources"]
            speech_path = target / f"{audio.stem}_reply.wav"
            synthesize(reply, speech_path, language_code)
            record = {
                "input_audio": str(audio),
                "transcript": heard,
                "reply": reply,
                "reply_audio": str(speech_path),
                "sources": [{"section_id": chunk["section_id"], "score": chunk["score"], "path": chunk["path"]} for chunk in sources],
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"{audio.name}: {reply}")
    print(f"Saved text, audio, and retrieval evidence in {target}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="ZentroMart VoiceOps capstone agent")
    commands = parser.add_subparsers(dest="command", required=True)
    batch = commands.add_parser("batch", help="Transcribe, answer, and speak every audio file in a folder")
    batch.add_argument("input_dir", help="Folder containing customer audio")
    batch.add_argument("output_dir", help="Folder for reply audio and results.jsonl")
    batch.add_argument("--language", default=None, help="Gnani language code for both speech calls, such as en-IN or hi-IN")
    batch.add_argument("--conversation", action="store_true", help="Keep one agent session across sorted input files")
    evaluation = commands.add_parser("evaluate", help="Compare prompt-only and RAG answers on practice questions")
    evaluation.add_argument("--output", default=None, help="Path for detailed JSON evaluation results")
    evaluation.add_argument("--dataset-dir", default=None, help="Dataset folder; defaults to this project folder")
    conversation_evaluation = commands.add_parser("evaluate-conversations", help="Evaluate all provided multi-turn conversations")
    conversation_evaluation.add_argument("--output", default=None, help="Path for detailed JSON evaluation results")
    conversation_evaluation.add_argument("--dataset-dir", default=None, help="Dataset folder; defaults to this project folder")
    conversation_evaluation.add_argument("--resume", action="store_true", help="Resume from completed conversations in the output file")
    args = parser.parse_args(argv)
    if args.command == "batch":
        process_audio_folder(args.input_dir, args.output_dir, args.language, args.conversation)
    elif args.command == "evaluate":
        evaluate(args.output, args.dataset_dir or dataset_root())
    elif args.command == "evaluate-conversations":
        evaluate_conversations(args.output, args.dataset_dir or dataset_root(), args.resume)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)

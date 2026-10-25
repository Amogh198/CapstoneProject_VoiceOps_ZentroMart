from pathlib import Path
import json

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "report.pdf"


def _metric_rows():
    question_path = ROOT / "outputs" / "evaluation.json"
    conversation_path = ROOT / "outputs" / "conversation_evaluation.json"
    question = json.loads(question_path.read_text(encoding="utf-8")) if question_path.exists() else {}
    conversation = json.loads(conversation_path.read_text(encoding="utf-8")) if conversation_path.exists() else {}
    def percent(value):
        return f"{value:.1%}" if isinstance(value, (int, float)) else "Run evaluator"
    def count(value):
        return str(value) if isinstance(value, int) else "Run evaluator"
    return [
        ["Prompt-only fact accuracy", percent(question.get("baseline_fact_accuracy"))],
        ["RAG fact accuracy", percent(question.get("rag_fact_accuracy"))],
        ["Question retrieval hit rate", percent(question.get("retrieval_hit_rate"))],
        ["Conversation fact accuracy", percent(conversation.get("required_fact_turn_accuracy"))],
        ["Expected tool-call accuracy", percent(conversation.get("tool_expectation_accuracy"))],
        ["Forbidden-claim turns", count(conversation.get("forbidden_claim_turns"))],
    ]


def build_report():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="CoverTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=21, leading=25, textColor=colors.HexColor("#15324B"), alignment=TA_CENTER, spaceAfter=8))
    styles.add(ParagraphStyle(name="Lead", parent=styles["BodyText"], fontSize=10, leading=14, textColor=colors.HexColor("#405365"), alignment=TA_CENTER, spaceAfter=16))
    styles.add(ParagraphStyle(name="Section", parent=styles["Heading2"], fontSize=12, leading=15, textColor=colors.HexColor("#147D83"), spaceBefore=8, spaceAfter=4))
    styles.add(ParagraphStyle(name="BodyCompact", parent=styles["BodyText"], fontSize=9, leading=12, spaceAfter=5))
    styles.add(ParagraphStyle(name="Cell", parent=styles["BodyText"], fontSize=8.5, leading=11))
    story = [
        Paragraph("VOICEOPS 2026", styles["CoverTitle"]),
        Paragraph("ZentroMart Voice Support Agent · Capstone Project Report", styles["Lead"]),
        Paragraph("Project", styles["Section"]),
        Paragraph("The agent answers customer questions from company policies, help pages, and the product catalog. It keeps conversational context, verifies callers before exposing order details, creates human handover tickets, and handles customer audio through Gnani speech services.", styles["BodyCompact"]),
        Paragraph("Build by step", styles["Section"]),
    ]
    steps = [
        ("1 · LLM baseline", "Groq-hosted Qwen 3.8 27B answers practice questions with a support-assistant prompt and no company-document retrieval."),
        ("2 · Grounded retrieval", "Markdown is split by stable section ID and indexed as TF-IDF vectors alongside catalog rows. Similarity search applies source precedence and downranks superseded policies."),
        ("3 · Conversation and tools", "The agent retains recent turns and verified order details within a session. The supplied lookup tool checks a phone suffix or delivery pincode and allows one retry. Ticket success is reported only after confirmation."),
        ("4 · Voice", "Gnani Vachana transcribes customer audio and synthesizes replies with an explicitly named timbre model. Batch output includes transcripts, spoken WAV files, and retrieved source IDs."),
    ]
    for heading, description in steps:
        story.append(KeepTogether([Paragraph(heading, styles["BodyCompact"]), Paragraph(description, styles["BodyCompact"])]))
    story.extend([
        Paragraph("Evaluation", styles["Section"]),
        Paragraph("Run `python run.py evaluate` and `python run.py evaluate-conversations` with an API key to score prompt-only and retrieval-augmented answers, source-section hits, forbidden claims, required facts, and expected tool calls. Detailed results are saved under `outputs/`.", styles["BodyCompact"]),
    ])
    table_data = [[Paragraph("Measure", styles["Cell"]), Paragraph("Result", styles["Cell"])]]
    table_data.extend([[Paragraph(label, styles["Cell"]), Paragraph(value, styles["Cell"])] for label, value in _metric_rows()])
    table = Table(table_data, colWidths=[3.3 * inch, 2.3 * inch], hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E5F2F1")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#15324B")),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#C8D4DC")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.extend([
        table,
        Spacer(1, 8),
        Paragraph("Lessons and next improvements", styles["Section"]),
        Paragraph("The hardest cases are policy precedence, date-sensitive returns, indirect follow-ups, multilingual wording, and privacy around unverified orders. Stable section IDs make evidence auditable, and the lookup tool gates customer details behind verification. Review evaluator misses, tune retrieval, and record the required short demo using your own voice and a Gnani response.", styles["BodyCompact"]),
    ])
    doc = SimpleDocTemplate(str(OUTPUT), pagesize=letter, rightMargin=0.72 * inch, leftMargin=0.72 * inch, topMargin=0.65 * inch, bottomMargin=0.65 * inch, title="VoiceOps 2026 Capstone Report", author="ZentroMart VoiceOps Capstone")
    doc.build(story)
    print(OUTPUT)


if __name__ == "__main__":
    build_report()

import math
import re
import csv
from collections import Counter
from pathlib import Path

from .settings import dataset_root


TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)
ALIASES = {
    "refund": "रिफंड paisa paise money wapas",
    "रिफंड": "refund paisa paise money wapas",
    "return": "returns returnable nonreturnable window period",
    "वापस": "return wapas pickup reverse collection",
    "grocery": "groceries fresh produce perishables food",
    "groceries": "grocery fresh produce perishables food",
    "cancel": "रद्द cancel कैंसिल",
    "रद्द": "cancel कैंसिल cancellation",
    "delivery": "डिलीवरी parcel order shipment courier",
    "डिलीवरी": "delivery parcel order shipment courier",
    "damaged": "खराब broken defect defective wrong missing damage",
    "खराब": "damaged broken defect defective wrong missing damage",
    "pickup": "collection reverse pickup executive collect",
    "paisa": "refund money रिफंड",
    "paise": "refund money रिफंड",
    "late": "delay delayed overdue promised date",
    "deri": "late delay delayed overdue",
    "kab": "when time timeline कितने दिन",
}
PRIORITY = {"policy": 1.12, "agent_guide": 1.04, "help": 1.0, "faq": 0.96, "marketing": 0.72, "meta": 0.7, "catalog": 1.05}


def tokenize(text):
    return [token.casefold() for token in TOKEN_RE.findall(text) if len(token) > 1]


def _parse_document(path):
    raw = path.read_text(encoding="utf-8")
    header = {}
    if raw.startswith("---\n"):
        _, frontmatter, body = raw.split("---", 2)
        for line in frontmatter.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                header[key.strip()] = value.strip().strip("\"'")
    else:
        body = raw
    sections = []
    section_id = "INTRO"
    title = path.stem
    content = []
    for line in body.splitlines():
        match = re.match(r"^##\s+\[([^\]]+)\]\s*(.*)$", line)
        if match:
            if content:
                sections.append((section_id, title, "\n".join(content).strip()))
            section_id, title = match.group(1), match.group(2).strip()
            content = []
        else:
            content.append(line)
    if content:
        sections.append((section_id, title, "\n".join(content).strip()))
    return header, sections


def _chunks(root):
    for path in sorted((root / "kb").glob("*.md")):
        metadata, sections = _parse_document(path)
        for section_id, title, text in sections:
            if not text:
                continue
            yield {
                "doc_id": metadata.get("doc_id", path.stem),
                "doc_type": metadata.get("doc_type", "unknown"),
                "status": metadata.get("status", "current"),
                "effective_from": metadata.get("effective_from", ""),
                "section_id": section_id,
                "title": title,
                "text": text,
                "path": str(path.relative_to(root)),
            }
    products_path = root / "data" / "products.csv"
    if products_path.exists():
        with products_path.open(encoding="utf-8-sig", newline="") as handle:
            for product in csv.DictReader(handle):
                sku = product.get("sku", "")
                details = "; ".join(f"{key}: {value}" for key, value in product.items() if value)
                yield {
                    "doc_id": "PRODUCT-CATALOG",
                    "doc_type": "catalog",
                    "status": "current",
                    "effective_from": "catalog",
                    "section_id": f"products.csv:{sku}",
                    "title": product.get("title", sku),
                    "text": details,
                    "path": "data/products.csv",
                }


class KnowledgeIndex:
    def __init__(self, root=None):
        self.root = Path(root or dataset_root())
        self.chunks = list(_chunks(self.root))
        if not self.chunks:
            raise FileNotFoundError(f"No Markdown knowledge documents found under {self.root / 'kb'}")
        self.term_counts = [Counter(tokenize(chunk["title"] + " " + chunk["text"])) for chunk in self.chunks]
        document_frequency = Counter()
        for counts in self.term_counts:
            document_frequency.update(counts.keys())
        total = len(self.chunks)
        self.idf = {term: math.log(1 + (total - count + 0.5) / (count + 0.5)) for term, count in document_frequency.items()}
        self.norms = []
        self.vectors = []
        for counts in self.term_counts:
            vector = {term: (1 + math.log(count)) * self.idf[term] for term, count in counts.items()}
            self.vectors.append(vector)
            self.norms.append(math.sqrt(sum(value * value for value in vector.values())) or 1.0)

    @staticmethod
    def _expand(tokens):
        expanded = list(tokens)
        for token in tokens:
            expanded.extend(tokenize(ALIASES.get(token, "")))
        return expanded

    def search(self, query, top_k=5):
        terms = Counter(self._expand(tokenize(query)))
        query_vector = {term: (1 + math.log(count)) * self.idf[term] for term, count in terms.items() if term in self.idf}
        query_norm = math.sqrt(sum(value * value for value in query_vector.values())) or 1.0
        ranked = []
        for chunk, vector, norm in zip(self.chunks, self.vectors, self.norms):
            dot = sum(value * vector.get(term, 0.0) for term, value in query_vector.items())
            score = (dot / (query_norm * norm)) * PRIORITY.get(chunk["doc_type"], 1.0)
            if chunk["status"] == "superseded":
                score *= 0.55
            if score >= 0.025:
                ranked.append((score, chunk))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [{**chunk, "score": round(score, 4)} for score, chunk in ranked[:top_k]]

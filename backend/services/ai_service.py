"""
خدمة AI خفيفة للتلخيص بدون اعتماد على نماذج خارجية.
Extractive summarization using a simple frequency-based scoring (language-aware for AR/EN).

الاستخدام:
from backend.services.ai_service import AIService
ai = AIService()
result = ai.summarize(text, max_sentences=3, ratio=0.25)
print(result["summary"])
"""

from __future__ import annotations
import re
from collections import Counter
from typing import Any, Dict, List, Tuple

__all__ = ["AIService"]

_AR_CHARS_RE = re.compile(r"[\u0600-\u06FF]")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?؟…])\s+|\n+")
_WORD_RE = re.compile(r"[\dA-Za-zÀ-ÖØ-öø-ÿ\u0600-\u06FF]+", flags=re.UNICODE)

_STOPWORDS_EN = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "in", "on", "at", "of", "for", "to", "from", "with", "without", "by",
    "and", "or", "but", "if", "then", "so", "than", "as", "that", "this",
    "these", "those", "it", "its", "into", "about", "over", "after", "before",
    "under", "above", "you", "we", "they", "he", "she", "his", "her", "their",
    "our", "your", "not", "no", "do", "does", "did", "done", "can", "could",
    "should", "would", "will", "just", "also", "such", "i", "me", "my", "myself",
    "ours", "yours", "theirs", "what", "which", "who", "whom", "where", "when",
    "why", "how", "all", "any", "both", "each", "few", "more", "most", "other",
    "some", "own", "same", "s", "t", "d", "ll", "m", "o", "re", "ve", "y",
}

_STOPWORDS_AR = {
    "و", "في", "على", "من", "إلى", "الى", "عن", "هذا", "هذه", "ذلك", "تلك",
    "هناك", "هنا", "هو", "هي", "هم", "هن", "أنا", "انا", "أنت", "انت", "أنتم",
    "انتم", "أنتن", "انتن", "نحن", "كما", "مثل", "لكن", "بل", "أو", "او", "أم",
    "ام", "مع", "أكثر", "اكثر", "أقل", "اقل", "قد", "لقد", "لم", "لن", "لا",
    "ما", "ماذا", "لماذا", "كيف", "أين", "اين", "متى", "إن", "ان", "أن", "أنّ",
    "كان", "تكون", "يكون", "كانت", "كانوا", "سوف", "كل", "أي", "اي",
    "بعض", "بين", "ضمن", "خلال", "قبل", "بعد", "عند", "عندما", "حيث", "إذ",
    "اذ", "إلا", "الا", "أيضًا", "ايضًا", "ايضا", "جدًا", "جدا",
}

def _has_arabic(text: str) -> bool:
    return bool(_AR_CHARS_RE.search(text or ""))

def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())

def _split_sentences(text: str) -> List[str]:
    parts = _SENTENCE_SPLIT_RE.split(text)
    return [s.strip() for s in parts if s and s.strip()]

def _tokenize(text: str) -> List[str]:
    return _WORD_RE.findall(text.lower())

class AIService:
    def __init__(
        self,
        stopwords_ar: set[str] | None = None,
        stopwords_en: set[str] | None = None,
    ) -> None:
        self.stopwords_ar = set(stopwords_ar or _STOPWORDS_AR)
        self.stopwords_en = set(stopwords_en or _STOPWORDS_EN)

    def summarize(
        self,
        text: str,
        max_sentences: int = 3,
        ratio: float = 0.25,
        min_sentence_len: int = 30,
    ) -> Dict[str, Any]:

        cleaned = _normalize_whitespace(text or "")
        if not cleaned:
            return {
                "summary": "",
                "language": "ar" if _has_arabic(text or "") else "en",
                "selected_indices": [],
                "sentences_count": 0,
            }

        lang = "ar" if _has_arabic(cleaned) else "en"
        sentences = _split_sentences(cleaned)

        if not sentences:
            return {
                "summary": "",
                "language": lang,
                "selected_indices": [],
                "sentences_count": 0,
            }

        if len(sentences) <= 2 or len(cleaned) < 200:
            return {
                "summary": cleaned,
                "language": lang,
                "selected_indices": list(range(len(sentences))),
                "sentences_count": len(sentences),
            }

        stopwords = self.stopwords_ar if lang == "ar" else self.stopwords_en

        sentence_tokens: List[List[str]] = []
        for s in sentences:
            toks = [t for t in _tokenize(s) if t and t not in stopwords and len(t) > 2]
            sentence_tokens.append(toks)

        all_tokens = [tok for toks in sentence_tokens for tok in toks]
        if not all_tokens:
            k = max(1, min(max_sentences, int(max(1, len(sentences) * ratio))))
            summary_sents = sentences[:k]
            return {
                "summary": " ".join(summary_sents).strip(),
                "language": lang,
                "selected_indices": list(range(k)),
                "sentences_count": len(sentences),
            }

        freq = Counter(all_tokens)
        max_freq = max(freq.values()) if freq else 1
        for w in freq.keys():
            freq[w] /= max_freq

        scores: List[Tuple[int, float]] = []
        for idx, toks in enumerate(sentence_tokens):
            if len(sentences[idx]) < min_sentence_len:
                continue
            if not toks:
                continue
            score = sum(freq.get(t, 0.0) for t in toks) / len(toks)
            scores.append((idx, float(score)))

        if not scores:
            k = max(1, min(max_sentences, int(max(1, len(sentences) * ratio))))
            selected_idx = list(range(k))
        else:
            k = max(1, min(max_sentences, int(max(1, len(sentences) * ratio))))
            top = sorted(scores, key=lambda x: x[1], reverse=True)[:k]
            selected_idx = sorted([idx for idx, _ in top])

        summary_sentences = [sentences[i] for i in selected_idx]
        summary_text = " ".join(summary_sentences).strip()

        return {
            "summary": summary_text,
            "language": lang,
            "selected_indices": selected_idx,
            "sentences_count": len(sentences),
        }

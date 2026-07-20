from dataclasses import dataclass

from models.document import Document


@dataclass
class RetrievalResult:
    document: Document
    score: float

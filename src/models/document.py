from dataclasses import dataclass


@dataclass
class Document:
    id: int
    title: str
    body: str
    label: int
    text: str
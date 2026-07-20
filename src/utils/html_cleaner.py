from bs4 import BeautifulSoup


def clean_html(text: str) -> str:
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text(" ", strip=True)

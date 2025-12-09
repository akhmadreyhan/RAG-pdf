from aiohttp.log import client_logger
from langchain_core.documents import Document
from PyPDF2 import PdfReader
from google import genai
import chromadb
from sentence_transformers import SentenceTransformer
import re as rr


def clean_pdf(txt):
    # del email
    txt = rr.sub(r"\S+@\S+", "", txt)

    # del url
    txt = rr.sub(r"http\S+", "", txt)

    # del arxiv::
    txt = rr.sub(r"arXiv:\d+\.\d+", "", txt)

    # del number page
    txt = rr.sub(r"^\s*\d+\s*$", "", txt, flags=rr.MULTILINE)

    # del copyright author
    txt = rr.sub(r"©.*\n", "", txt)
    txt = rr.sub(r"(Author|Authors|Affiliation|University|Dept\.).*\n", "", txt)

    pattern_author = rr.compile(r"(.*?)(abstract)", rr.IGNORECASE | rr.DOTALL)
    match = pattern_author.search(txt)
    if not match:
        return txt

    before_abstract = match.group(1)
    after_abstract = txt[len(before_abstract) :]

    title = before_abstract.strip().split("\n")[0]
    txt = title + "\n\n" + after_abstract

    return split_chunk(txt)


def split_chunk(txt):
    chunk = []
    current_chunk = []

    sentences = rr.split(r"(?<=[.?!])\s+", txt)
    for sentence in sentences:
        if not sentence.strip():
            continue

        current_chunk.append(sentence)

        if len(current_chunk) == 5:
            chunk.append(" ".join(current_chunk))
            current_chunk = []

    if current_chunk:
        chunk.append(" ".join(current_chunk))

    return chunk


def obj_doc(txt):
    data = []
    for page in txt:
        data.append(Document(page_content=page))

    data = [page.page_content for page in data]
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embed = model.encode(data)
    return save_db(embed, data)


def save_db(embed, data):
    db = chromadb.Client()
    collection = db.get_or_create_collection(name="rag")
    collection.upsert(
        documents=data, embeddings=embed, ids=[str(i) for i in range(len(embed))]
    )

    return collection


def llm(txt):
    model = genai.Client()
    result = model.models.generate_content(
        model="gemini-2.5-flash",
        contents=[{"role": "user", "parts": [{"text": txt}]}],
    )
    return result.text


read = PdfReader("books.pdf")
txt = ""
for hal in read.pages:
    txt += hal.extract_text()

z = obj_doc(clean_pdf(txt))
print(z)

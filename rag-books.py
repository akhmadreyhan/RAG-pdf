from langchain_core.documents import Document
import pymupdf as pdf
import pymupdf.layout
from tqdm import tqdm
import pymupdf4llm
from google import genai
import chromadb
from sentence_transformers import SentenceTransformer
import re as rr


def clean_pdf(txt):
    print("Cleaning text...")
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

    # del '>' word
    txt = rr.sub(r"^>.*", "", txt, flags=rr.MULTILINE)

    # del '==>' word
    txt = rr.sub(r"^==>.*", "", txt, flags=rr.MULTILINE)

    # del ascii symbol
    txt = rr.sub(r"^(?:[+|]|-{3,}).*", "", txt, flags=rr.MULTILINE)

    # del white space
    txt = rr.sub(r"\n{3,}", "\n\n", txt)

    pattern_author = rr.compile(r"(.*?)(abstract)", rr.IGNORECASE | rr.DOTALL)
    match = pattern_author.search(txt)
    if not match:
        return txt

    before_abstract = match.group(1)
    after_abstract = txt[len(before_abstract) :]

    title = before_abstract.strip().split("\n")[0]
    txt = title + "\n\n" + after_abstract
    # txt = rr.sub(r"\[\s*\d+\s*\]", "", txt)
    # txt = rr.sub(r"\[[\d,\s]+\]", "", txt)
    txt = txt.strip()

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

    # print(data)

    data = [page.page_content for page in data]
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Encoding text...\n")
    embed = model.encode(data)

    return save_db(embed, data)


def save_db(embed, data):
    db = chromadb.Client()
    collection = db.get_or_create_collection(name="rag")
    collection.upsert(
        documents=data, embeddings=embed, ids=[str(i) for i in tqdm(range(len(embed)))]
    )
    # docs = collection.get(include=["documents"])
    print("\n")
    # a = 1
    # for i in docs["documents"][:5]:
    #     print(f"Chunk ke-{a}: \n{i}\n")
    #     a += 1

    print("Starting LLM...\n")
    return collection


def llm(txt):
    db = chromadb.Client()
    collection = db.get_or_create_collection(name="rag")
    user = input("\nHalo! ada yang bisa saya bantu? \nYour thoughts: ")
    print("\nGenerating response...\n")
    queries = collection.query(query_texts=[user], n_results=3, include=["documents"])
    prompt = f""" 
    Anda adalah seorang AI asisten yang membantu pengguna dengan pertanyaan mereka. Hal-hal seperti umur pengguna, jenis kelamin pengguna, dan sifat pengguna bermacam-macam sehingga Anda harus dapat beradaptasi dengan pengguna.
    Dalam menjawab pertanyaan, anda diberikan beberapa konteks yang relevan sehingga anda dapat memberikan jawaban yang akurat. Selain itu, berikan jawaban kepada pengguna sesuai dengan bahasa yang mereka gunakan seperti jika pengguna menanyakan pertanyaan dalam bahasa inggris, maka berikan jawaban bahasa inggris juga. 
    Konteks tersebut sebagai berikut:
    {queries}

    Pertanyaan yang user berikan: {user}
    """
    model = genai.Client(api_key="insert api here")
    result = model.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    return result.text.replace("**", "")


read = pdf.open("books.pdf")
print("Extracting text...")
body = pymupdf4llm.to_text(read, footer=False)

z = obj_doc(clean_pdf(body))
y = llm(z)
print(y)

import os
import fitz

def load_pdfs(folder="docs"):

    docs = []

    if not os.path.exists(folder):
        return docs

    for file in os.listdir(folder):

        if file.endswith(".pdf"):

            path = os.path.join(folder, file)

            pdf = fitz.open(path)

            text = ""

            for page in pdf:
                text += page.get_text()

            docs.append(text)

    return docs
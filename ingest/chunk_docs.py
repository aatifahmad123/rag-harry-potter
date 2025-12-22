from langchain_text_splitters import RecursiveCharacterTextSplitter
from load_docs import load_pdfs

def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = []

    for doc in documents:
        split_texts = splitter.split_text(doc["text"])
        for chunk in split_texts:
            chunks.append({
                "source": doc["source"],
                "text": chunk
            })

    return chunks


if __name__ == "__main__":
    documents = load_pdfs()
    chunks = chunk_documents(documents)
    print(f"Created {len(chunks)} chunks")

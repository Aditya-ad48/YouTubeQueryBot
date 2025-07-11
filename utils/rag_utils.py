from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_size_splitter(transcript):
  if len(transcript) < 3000:
    chunk_size = 800
    chunk_overlap = 100

  elif len(transcript) < 10000:
    chunk_size = 1000
    chunk_overlap = 150
  else:
    chunk_size = 1500
    chunk_overlap = 200
  return chunk_size, chunk_overlap



def process_transcript(transcript, persist_id="default"):
    chunk_size, chunk_overlap = chunk_size_splitter(transcript)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = splitter.create_documents([transcript])
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=f"./db/{persist_id}")
    return vectorstore.as_retriever(search_type="mmr",
                                    search_kwargs={"k": 3, "lambda_mult": 0.5}  # k = top results, lambda_mult = relevance-diversity balance
                                    )
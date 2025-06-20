from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
from dotenv import load_dotenv
import os
import re

# Load environment variables from .env file
load_dotenv(override=True)

# Load FAISS vector database from local folder
def load_vectordb(path="faiss_index"):
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local(
        path,
        embeddings,
        allow_dangerous_deserialization=True  # Be cautious in production
    )

# Perform similarity search
def search(query, db, k=3):
    docs = db.similarity_search(query, k=k)
    if not docs:
        return "No relevant documents found."
    return docs[0].page_content

# Clean the context text
def clean_text(text):
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Generate answer using OpenAI API
def complete_chat(query, context):
    client = OpenAI()
    system_prompt = "You are a helpful assistant who answers questions about Berwyn."
    user_prompt = f"""
Answer the question using the following information:\n
{context}\n
If the context isn't relevant to the question, just say you don't have information. Don't make anything up.
The user asks: {query}
    """.strip()

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    return completion.choices[0].message.content

# Main logic for command-line usage
if __name__ == "__main__":
    vectordb = load_vectordb()
    query = input("Enter a question: ")
    context = clean_text(search(query, vectordb))
    response = complete_chat(query, context)
    print("\nAssistant:", response)

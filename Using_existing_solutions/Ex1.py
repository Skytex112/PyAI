import chromadb
import ollama

chromadb_client = chromadb.PersistentClient(path="./chroma_db")
collection = chromadb_client.get_or_create_collection(name="local_base")

documents = [
    (
        "doc_1",
        "Architect project Opa: mobile app on React Native "
        "(Expo) with navigation, validation and work with gallery "
        "React Hook Form"
    ),
    (
        "doc_2",
        "Backend development database structure "
        "API endpoints authentication system"
    ),
    (
        "doc_3",
        "Frontend design user interface components "
        "responsive layout state management"
    ),
    (
        "doc_4",
        "Tester database structure "
        "API endpoints authentication system"
    ),
    (
        "doc_5",
        "User user interface components "
        "responsive layout state management"
    )
]

if collection.count() == 0:
    ids = [item[0] for item in documents]
    texts = [item[1] for item in documents]
    # ChromaDB сам посчитает эмбеддинги через встроенный all-MiniLM-L6-v2 (ONNX)
    collection.add(documents=texts, ids=ids)

user_query = "Which docs with backend?"
result = collection.query(query_texts=[user_query], n_results=2)

retrieved_chunks = result['documents'][0]

for i, chunk in enumerate(retrieved_chunks):
    print(f"[{i+1}] {chunk}")

context = "\n".join(retrieved_chunks)

prompt = f"""Use only the context provided below to answer the question.
If the answer is not in the context, say: "I do not have that information in my base."

Context:
{context}

User query:
{user_query}
"""

response = ollama.chat(
    model="llama3",
    messages=[
        {
            "role": "system",
            "content": "You are a clever corporate assistant who answers based on docs.",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]
)

print("=" * 40)
print("Answer:")
print("=" * 40)
print(response['message']['content'])
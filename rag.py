import os
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version=os.getenv("AZURE_OPENAI_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

# Folder holding your text dataset
DATASET_FOLDER = "data"  # (Can change this name for other datasets)

def load_all_txt_files(folder):
    """Reads and concatenates all .txt files in the dataset folder."""
    all_docs = ""
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            with open(os.path.join(folder, filename), "r", encoding="utf-8") as f:
                all_docs += f"--- {filename} ---\n{f.read().strip()}\n\n"
    return all_docs

# Load dataset at start (can be reloaded anytime if needed)
dataset_context = load_all_txt_files(DATASET_FOLDER)

print("Welcome to RAG-style QA! (Type 'exit' to quit)")

while True:
    user_input = input("\nUser: ")
    if user_input.strip().lower() == "exit":
        print("Bye!")
        break

    # Build prompt: System + user with dataset
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert assistant. "
                "Always answer using ONLY the information provided in the dataset below. "
                "If the answer cannot be found in the dataset, reply: 'Not found in provided documents.'"
            )
        },
        {
            "role": "user",
            "content": (
                f"Dataset:\n{dataset_context}\n\n"
                f"Question: {user_input}"
            )
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.01,
        top_p=0.95,
        frequency_penalty=0.05,
        presence_penalty=1.99,
        max_tokens=1000
    )

    answer = response.choices[0].message.content
    print("\nAI Assistant:", answer)

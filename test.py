import os
from google import genai

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY", ""))

response = client.models.embed_content(
    model="gemini-embedding-001",
    contents="retry logic in dispatch system",
)

embedding = response.embeddings[0].values
print(response)

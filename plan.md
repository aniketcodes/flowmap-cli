# FlowMap — Org-Level Code Intelligence Plan (Gemini)

## 🎯 Goal

Build a system that:

* Indexes multiple repositories
* Uses **Gemini embeddings (text-embedding-004)**
* Stores vectors in a local DB (Qdrant)
* Enables cross-repo search
* Supports:

  * 🐞 Debugging
  * 🛠️ Feature development across repos

---

## 🧠 High-Level Architecture

```id="n4e6sv"
Repos → Chunk → Gemini Embeddings → Qdrant → Retrieval → Claude/MCP
```

---

## ⚙️ Phase 1: MVP (1–2 days)

---

### Step 1: Setup

#### Install dependencies

```bash id="n7q9z1"
pip install google-generativeai qdrant-client tqdm
```

---

#### Run Qdrant

```bash id="t9a2ls"
docker run -p 6333:6333 qdrant/qdrant
```

---

#### Set Gemini API key

```bash id="l5r0gq"
export GOOGLE_API_KEY="your_key"
```

---

### Step 2: Gemini Embedding Function

```python id="8y2xmq"
import google.generativeai as genai

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_embedding(text):
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=text
    )
    return result["embedding"]
```

---

### Step 3: Repo Indexing CLI

Command:

```bash id="2yqf8c"
flowmap index /path/to/repo
```

---

### Responsibilities

* Walk repo files
* Filter code files
* Chunk text (~1000 chars, overlap 200)
* Generate embeddings via Gemini
* Store in Qdrant

---

### Step 4: Metadata (critical)

```json id="i6g1kq"
{
  "repo": "dispatch-service",
  "file": "dispatch.service.ts",
  "text": "...",
  "file_name": "dispatch.service.ts",
  "extension": "ts"
}
```

---

### Step 5: Store in Qdrant

Collection: `code_index`

Vector size:

* Gemini embedding ≈ **768 dims** (auto-detect once)

---

### Step 6: Query CLI

```bash id="rf4h2c"
flowmap ask "how does dispatch call geo?"
```

Flow:

1. Embed query using Gemini
2. Search Qdrant
3. Return top results

---

### Step 7: Validate MVP

Test queries:

* "where is retry logic?"
* "dispatch flow"
* "distance matrix usage"

Success criteria:

* Relevant results in top 5
* Cross-repo retrieval works

---

## 🚀 Phase 2: Improve Retrieval

---

### Add filtering

* repo filter
* extension filter

---

### Hybrid search

Combine:

```id="l0u3w9"
vector similarity + keyword match
```

---

### Add tags

```json id="z9q1m8"
"tags": ["retry", "dispatch"]
```

---

## 🧠 Phase 3: Cross-Repo Intelligence

---

### Lightweight relationships (DON’T overbuild)

Extract:

* imports
* service names

```json id="p2n7xa"
"imports": ["geo-service"]
```

---

### Query expansion

```id="j7v3xp"
"retry issue"
→ "retry dispatch driver rejection config"
```

---

### Structured results

```id="0xv8qt"
- retryDispatch() → dispatch-service
- geo-service usage
- config for retry
```

---

## 🔌 Phase 4: Claude Integration (MCP)

---

### Tool: search_codebase

Input:

```id="y3w8qt"
query
```

Output:

* top chunks from Qdrant

---

### Tool: get_file

Input:

```id="c6n1bp"
file path
```

Output:

* full file

---

### Claude Flow

```id="h5r9ls"
User → Claude → MCP → Qdrant → Claude → Answer
```

---

## ⚡ Phase 5: Optimization

---

### Batch embeddings (IMPORTANT)

```python id="u2m4xr"
genai.embed_content(
    model="models/text-embedding-004",
    content=[text1, text2, text3]
)
```

---

### Performance

* parallel file processing
* batch uploads to Qdrant

---

### Cost

* Gemini is cheap/free tier
* cache embeddings to avoid reprocessing

---

## 🧪 Evaluation

---

### Debugging queries

* "why retries increased?"
* "why cost spiked?"

---

### Feature queries

* "where to add pricing logic?"
* "how dispatch interacts with geo?"

---

### Metrics

* relevance
* latency
* cross-repo coverage

---

## ⚠️ MVP Limitations

* naive chunking
* no logs/metrics
* no deep dependency graph

---

## 🔥 Future Enhancements

---

### System intelligence

* logs + metrics integration
* anomaly detection

---

### Dependency graph

* service graph
* execution tracing

---

### TurboQuant

* compress embeddings
* reduce memory usage

---

## 🧠 Key Principles

* Start simple
* Don’t overengineer
* Retrieval > model
* Metadata > raw embeddings

---

## ✅ Final Outcome

FlowMap becomes:

👉 A cross-repo intelligence system that helps you:

* Debug faster
* Build features confidently
* Understand system flow

---


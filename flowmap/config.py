import os
from pathlib import Path

from dotenv import load_dotenv

# Project root: repo-embedd/ (parent of this package)
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "") or os.getenv("GEMINI_API_KEY", "")

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = "code_index"

# Default: Gemini Embedding 2 (multimodal; text-only still supported). Override: FLOWMAP_EMBEDDING_MODEL
# After changing model or dimensions, re-index (vectors are not compatible with a different model/dim).
EMBEDDING_MODEL = os.getenv("FLOWMAP_EMBEDDING_MODEL", "gemini-embedding-2-preview")
EMBEDDING_DIMS = int(os.getenv("FLOWMAP_EMBEDDING_DIMS", "768"))

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
# Smaller batches + delay reduce Gemini 429 rate-limit errors on free tier
BATCH_SIZE = int(os.getenv("FLOWMAP_EMBED_BATCH_SIZE", "8"))
EMBED_INTER_BATCH_DELAY = float(os.getenv("FLOWMAP_EMBED_DELAY", "2.5"))
# Extra pause after any batch that hit 429 (helps stay under RPM)
EMBED_COOLDOWN_AFTER_429 = float(os.getenv("FLOWMAP_EMBED_COOLDOWN_AFTER_429", "10.0"))

SUPPORTED_EXTENSIONS = {
    ".py", ".ts", ".js", ".tsx", ".jsx",
    ".go", ".rs", ".java", ".kt",
    ".c", ".cpp", ".h", ".hpp",
    ".rb", ".php", ".cs", ".swift",
    ".yaml", ".yml", ".toml", ".json",
    ".sh", ".bash",
    ".sql", ".graphql", ".proto",
    ".dockerfile", ".tf", ".hcl",
}

SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv",
    "vendor", "dist", "build", ".next", ".nuxt",
    "target", ".idea", ".vscode", ".cursor",
    "coverage", ".tox", "egg-info",
}

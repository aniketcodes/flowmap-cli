# FlowMap

**Cross-repo code intelligence CLI for LLMs.**

FlowMap indexes your codebases with tree-sitter AST parsing, stores them in a local vector database, and gives you fast hybrid search (semantic + keyword + symbol) across all your repos. Built for developers who use LLMs for code navigation and want better context than `grep`.

### Why?

You have 5 repos. You know the retry logic exists *somewhere*. With `grep` you'd need to search each repo, wade through hundreds of string matches, and hope you find the right function. With FlowMap:

```
$ flowmap search "retry logic with exponential backoff"
[1] api-gateway/src/utils/retry.ts:12-45  (rrf: 0.0312, via: ripgrep+semantic)
    export async function withRetry<T>(fn: () => Promise<T>, opts: RetryOptions): Promise<T> { ...

[2] payment-service/src/http/client.py:88-120  (rrf: 0.0198, via: semantic+symbol)
    def retry_with_backoff(func: Callable, max_retries: int = 3, base_delay: float = 1.0): ...

[3] shared-lib/pkg/resilience/retry.go:15-52  (rrf: 0.0147, via: semantic)
    func Retry(ctx context.Context, fn func() error, opts ...Option) error { ...
```

One query. Three repos. Three languages. Ranked by relevance. That's the point.

---

## What it does

- **Indexes your repos** with tree-sitter, extracting functions, classes, methods, and their signatures
- **Hybrid search** fuses 3 channels: ripgrep (keyword), vector similarity (semantic), and symbol lookup (exact match) using Reciprocal Rank Fusion
- **Incremental reindexing** via `git diff` -- only re-embeds changed files
- **Structural history** shows AST-level diffs (function added/removed/signature changed) over time
- **Works across repos** -- search one query, get results from all your projects

## Languages with AST support

Full tree-sitter parsing (functions, classes, methods, signatures):

**Python** | **TypeScript** | **JavaScript** | **TSX/JSX** | **Go** | **Java** | **YAML** | **JSON**

These languages get line-based fallback chunking (indexed but no symbol extraction):

Rust | C | C++ | Kotlin | Ruby | PHP | C# | Swift | SQL | GraphQL | Protobuf | Terraform | Shell | Markdown

---

## Quick start

### 1. Install

```bash
git clone https://github.com/aniket-agi/flowmap-cli.git
cd flowmap-cli
uv sync
```

> Don't have uv? Install it: `curl -LsSf https://astral.sh/uv/install.sh | sh`

### 2. Add to your PATH

Add this to your `~/.zshrc` (or `~/.bashrc`):

```bash
export PATH="/path/to/flowmap-cli/.venv/bin:$PATH"
```

Then reload:

```bash
source ~/.zshrc
```

Verify it works:

```bash
flowmap --help
```

### 3. Install Ollama (for embeddings)

FlowMap uses [Ollama](https://ollama.com) for embeddings by default. It's free, runs locally, and needs no API keys.

```bash
# Install Ollama (macOS)
brew install ollama

# Start the server
ollama serve

# Pull the embedding model (~400MB download, one-time)
ollama pull qwen3-embedding:0.6b
```

### 4. Install ripgrep (for keyword search)

```bash
# macOS
brew install ripgrep

# Ubuntu/Debian
apt install ripgrep

# Or see https://github.com/BurntSushi/ripgrep#installation
```

### 5. Add your repos

```bash
# Initialize config
flowmap init

# Add repos (use absolute paths)
flowmap repos add /path/to/your/project
flowmap repos add /path/to/another/project

# Verify
flowmap repos list
```

### 6. Index

```bash
flowmap index
```

This walks each repo, parses files with tree-sitter, generates embeddings via Ollama, and stores everything locally. First run takes a few minutes depending on repo size. Subsequent runs are incremental (seconds).

### 7. Search

```bash
flowmap search "retry logic"
```

That's it. You're searching across all your repos.

---

## Real-world workflows

These are the things FlowMap is actually good at. Copy-paste these.

### "I just joined a new team. How is this codebase structured?"

```bash
flowmap repos add /path/to/the/repo
flowmap index
flowmap map
```

This gives you every class, function, and file at a glance — across all repos.

### "Where is the code that does X?"

```bash
flowmap search "payment processing"
flowmap search "retry logic"
flowmap search "database connection pool"
```

This is the core use case. You describe what you're looking for in plain English, and FlowMap finds the relevant functions/classes across all your repos. It's not just grep — it understands meaning.

### "I know the function name but not which repo it's in"

```bash
flowmap search "processOrder" --mode symbol
```

Symbol mode does exact/fuzzy matching on function and class names. Instant results, no embeddings needed.

### "I found a function. Show me the full source."

```bash
# From search results, you see: auth-service/src/auth.py:25-70
flowmap cat src/auth.py --repo auth-service --lines 25-70

# Or jump directly to a symbol
flowmap cat src/auth.py --repo auth-service --symbol validateToken
```

### "What changed in this area of the code recently?"

```bash
flowmap history "validateToken"
flowmap history "payment" --repo payment-service --since "3 months ago"
```

This shows AST-level diffs — not just "file changed" but "function `processPayment` had its signature changed on March 3rd."

### "I need to give an LLM context about my codebase"

```bash
# Structural overview
flowmap map --format json

# Find relevant code for a question
flowmap search "how does auth work" --format json

# Read specific files
flowmap cat src/auth.py --repo my-service --format json
```

All commands support `--format json`. Pipe them to Claude, ChatGPT, or any LLM tool.

### "I pulled from git. Update the index."

```bash
flowmap index
```

That's it. FlowMap detects what changed via `git diff` and only re-embeds the modified files. Takes seconds.

### "Something is broken. How do I debug?"

```bash
flowmap doctor
```

This checks: Ollama running? Model pulled? Repos exist? Index healthy? Dimension mismatch? It tells you exactly what's wrong and how to fix it.

### "I want fast grep-style search without Ollama"

```bash
flowmap search "TODO" --mode keyword
flowmap search "FIXME" --mode keyword
```

Keyword mode uses ripgrep directly. No embeddings, no Ollama, instant results.

---

## Cheat sheet

```
flowmap index                    # Build/update the index (incremental)
flowmap index --full             # Force full rebuild
flowmap search "query"           # Hybrid search (semantic + keyword + symbol)
flowmap search "fn" --mode symbol  # Find by function/class name
flowmap search "x" --mode keyword  # Grep-style (no Ollama needed)
flowmap map                      # Show repo structure
flowmap symbols                  # List all functions/classes
flowmap symbols "auth"           # Search symbols by name
flowmap cat file.py --repo R     # Read a file
flowmap cat file.py --symbol fn  # Jump to a symbol
flowmap history "query"          # Show structural changes over time
flowmap status                   # Check index health
flowmap doctor                   # Full system health check
flowmap repos add /path          # Add a repo
flowmap repos list               # List configured repos
flowmap reset --all              # Delete all index data
```

---

## Commands

### `flowmap search`

The main command. Searches across all indexed repos.

```bash
# Default: hybrid search (semantic + keyword + symbol fusion)
flowmap search "database connection pooling"

# Semantic only (vector similarity)
flowmap search "error handling patterns" --mode semantic

# Keyword only (ripgrep -- no embeddings needed)
flowmap search "TODO" --mode keyword

# Symbol lookup (exact/fuzzy match on function/class names)
flowmap search "AuthMiddleware" --mode symbol

# Filter to one repo
flowmap search "validateToken" --repo auth-service

# JSON output (for piping to LLMs)
flowmap search "payment processing" --format json

# Cross-encoder reranking (slower but higher quality)
flowmap search "complex query" --rerank
```

**Search modes:**


| Mode               | What it does                                   | Speed | Needs Ollama? |
| ------------------ | ---------------------------------------------- | ----- | ------------- |
| `hybrid` (default) | Fuses ripgrep + vector + symbol search via RRF | ~1-2s | Yes           |
| `semantic`         | Vector similarity only                         | ~0.5s | Yes           |
| `keyword`          | ripgrep only (live filesystem grep)            | ~0.1s | No            |
| `symbol`           | Exact/suffix/contains match on symbol names    | ~0.1s | No            |


### `flowmap index`

Build or update the search index.

```bash
# Index all repos (incremental -- only re-embeds changed files)
flowmap index

# Force full re-index
flowmap index --full

# Index a specific repo
flowmap index --repo my-service

# Preview what would be indexed (fast, no parsing)
flowmap index --dry-run
```

### `flowmap map`

Show a structural overview of your indexed repos -- classes, functions, file counts, languages.

```bash
flowmap map
flowmap map --repo my-service
flowmap map --format json
```

### `flowmap symbols`

List and search symbols (functions, classes, methods) across repos.

```bash
# List all symbols
flowmap symbols

# Search for symbols matching a name
flowmap symbols "process"

# Filter by type
flowmap symbols --type class
flowmap symbols --type function --repo my-service

# JSON output
flowmap symbols "validate" --format json
```

### `flowmap cat`

Read source files from configured repos. Supports line ranges and symbol-based lookup.

```bash
# Read a file (auto-detects repo from path)
flowmap cat my-service/src/auth.py

# Specific line range
flowmap cat src/auth.py --repo my-service --lines 25-70

# Jump to a symbol
flowmap cat src/auth.py --repo my-service --symbol validateToken

# JSON output (useful for LLM context)
flowmap cat src/service.ts --repo my-service --format json
```

### `flowmap history`

Show a timeline of structural changes -- which functions were added, removed, or had their signatures changed.

```bash
# What changed around "auth"?
flowmap history "validateToken"

# Scoped to a repo and time window
flowmap history "payment" --repo payment-service --since "3 months ago"

# Focus on a specific symbol
flowmap history "OrderProcessor" --symbol OrderProcessor.process

# JSON output
flowmap history "auth" --format json
```

### `flowmap status`

Show index status for all repos.

```bash
flowmap status
```

```
Index: 12,450 total chunks

  my-service                      4,230 chunks  2026-04-01  (main, abc1234)
  auth-service                    3,100 chunks  2026-04-01  (main, def5678)
  shared-lib                      5,120 chunks  2026-03-28  (main, 789abcd)
```

### `flowmap doctor`

Check that everything is set up correctly.

```bash
flowmap doctor
```

Checks: repo paths exist, Ollama is running, embedding model is pulled, ripgrep is installed, index is healthy, no dimension mismatches.

### `flowmap repos`

Manage configured repositories.

```bash
flowmap repos add /path/to/repo          # Add a repo
flowmap repos add /path/to/repo --name custom-name  # Add with a custom alias
flowmap repos list                        # List all repos and their index status
flowmap repos paths                       # Output repo paths (one per line)
```

### `flowmap reset`

Delete index data.

```bash
flowmap reset --repo my-service    # Reset one repo
flowmap reset --all                # Reset everything
```

### `flowmap init`

Create a starter config file.

```bash
flowmap init                # Creates ~/.flowmap/config.yaml
flowmap init --force        # Overwrite existing config (preserves repo list)
```

---

## Configuration

Config lives at `~/.flowmap/config.yaml`. Created by `flowmap init`.

```yaml
# FlowMap configuration

repos:
  - name: my-service
    path: /Users/you/code/my-service
  - name: auth-service
    path: /Users/you/code/auth-service

data_dir: ~/.flowmap/data

embedding:
  backend: ollama                        # ollama | sentence-transformers
  model: qwen3-embedding:0.6b           # model name
  ollama_url: http://localhost:11434

reranking:
  enabled: false                         # Enable with --rerank flag instead
  model: cross-encoder/ms-marco-MiniLM-L-6-v2
```

### Custom config path

```bash
flowmap --config /path/to/config.yaml search "query"
```

### `.flowmapignore`

Add a `.flowmapignore` file to any repo root to exclude files from indexing. Uses gitignore syntax.

```
# .flowmapignore
generated/
*.pb.go
*_test.go
vendor/
```

---

## Embedding backends

### Ollama (default, recommended)

Free, local, no API keys. Runs on CPU or GPU.

```bash
ollama serve
ollama pull qwen3-embedding:0.6b
```

Config:

```yaml
embedding:
  backend: ollama
  model: qwen3-embedding:0.6b
  ollama_url: http://localhost:11434
```

### Sentence-transformers (optional)

Local Python-based embeddings. No external server needed, but requires PyTorch.

```bash
uv sync --extra local-embeddings
```

Config:

```yaml
embedding:
  backend: sentence-transformers
  model: nomic-ai/CodeRankEmbed
```

---

## How it works

```
Your repos                FlowMap                    Search
-----------              ---------                  --------
  .py .ts .go    --->   tree-sitter AST parsing
  .java .yaml            chunk into functions,
                          classes, methods
                              |
                              v
                         Ollama / sentence-transformers
                          generate embeddings
                              |
                              v
                         LanceDB (vectors)  +  SQLite (metadata)
                          local storage, no cloud
                              |
                              v
                         3-way hybrid search  <---  "your query"
                          ripgrep + vector + symbol
                              |
                              v
                         Reciprocal Rank Fusion
                          merge & score results
                              |
                              v
                         Ranked results with
                          file, line, symbol, score
```

**Indexing pipeline:**

1. Walk repos via `git ls-files` (respects `.gitignore`)
2. Parse each file with tree-sitter to extract functions, classes, methods
3. Generate embeddings via Ollama (batched, with retry)
4. Store in LanceDB (vectors) + SQLite (metadata, state tracking)
5. Incremental updates via `git diff` -- only changed files are re-embedded

**Search pipeline:**

1. Classify query (identifier vs natural language vs mixed)
2. Run 3 search channels in parallel: ripgrep, vector similarity, symbol lookup
3. Map ripgrep line hits to stored chunks (dedup before scoring)
4. Fuse with weighted Reciprocal Rank Fusion (weights based on query type)
5. Optional cross-encoder reranking on top-30 candidates

---

## JSON output

All commands support `--format json` for piping to LLMs or scripts:

```bash
# Feed search results to an LLM
flowmap search "auth middleware" --format json | llm "explain these results"

# Get repo map as structured data
flowmap map --format json | jq '.repos[].classes[].name'

# Read a file for LLM context
flowmap cat src/auth.py --repo my-service --format json
```

---

## Troubleshooting

### `flowmap doctor` reports issues

Run `flowmap doctor` first. It checks everything and tells you what's wrong.

### "Ollama not running"

```bash
ollama serve          # Start Ollama
ollama pull qwen3-embedding:0.6b   # Pull the model
```

### "ripgrep (rg) not installed"

Keyword search and hybrid mode need ripgrep. Install it:

```bash
brew install ripgrep   # macOS
apt install ripgrep    # Ubuntu/Debian
```

Without ripgrep, `--mode semantic` and `--mode symbol` still work.

### "Dimension mismatch"

You changed the embedding model after indexing. Fix:

```bash
flowmap index --full
```

### Search returns no results

```bash
flowmap status        # Check if repos are indexed
flowmap index         # Re-index if needed
flowmap doctor        # Check system health
```

### Slow indexing

First index is slow (parses all files + generates embeddings). Subsequent runs are incremental and fast. For very large repos, ensure Ollama has enough resources:

```bash
# Check Ollama is responsive
curl http://localhost:11434/api/tags
```

---

## Requirements

- **Python** >= 3.11
- **Ollama** (for embeddings) -- [install](https://ollama.com)
- **ripgrep** (for keyword search) -- [install](https://github.com/BurntSushi/ripgrep#installation)
- **git** (for file listing and incremental reindex)

---

## Development

```bash
# Clone
git clone https://github.com/aniket-agi/flowmap-cli.git
cd flowmap-cli

# Install with dev dependencies
uv sync --extra dev

# Run tests
uv run pytest tests/ -v

# Lint
uv run ruff check flowmap/
```

### Test suite

291 tests covering:

- Tree-sitter chunking (Python, TypeScript, Go, Java, YAML, JSON)
- LanceDB store operations (real database, not mocked)
- CLI commands (all 11 commands)
- Hybrid search fusion and deduplication
- Incremental reindexing with git
- End-to-end: index -> search -> cat pipeline
- SQL escaping and special character handling
- Crash recovery (embedding failure preserves data)
- History/timeline with structural diffs

---

## FAQ

### How is this different from `grep` / `ripgrep`?

ripgrep finds literal text matches. FlowMap understands code structure -- it knows what a function is, what class it belongs to, and can find semantically similar code even when the exact words don't match. FlowMap actually *uses* ripgrep as one of its three search channels and fuses the results.

### How is this different from GitHub code search?

GitHub code search works on github.com. FlowMap works on your local repos, offline, with no data leaving your machine. It also searches *across* multiple repos at once and provides structural history (AST-level diffs over time).

### Does my code leave my machine?

No. Everything runs locally. Embeddings are generated by Ollama on your machine. Data is stored in `~/.flowmap/data`. No cloud services, no API keys, no telemetry.

### How much disk space does it use?

Roughly 1-2 MB per 1,000 source files. A 10-repo setup with 50K files typically uses ~100 MB for the LanceDB vector store.

### Can I use OpenAI / Anthropic / other API embeddings?

Not currently. FlowMap supports Ollama (recommended) and sentence-transformers. Adding API-based backends is straightforward if there's demand -- open an issue.

### How long does indexing take?

First full index: ~1-5 minutes for a typical repo (depends on size and Ollama speed). Incremental updates after `git pull`: seconds -- only changed files are re-embedded.

### Can I use this with Claude Code / Cursor / Copilot?

Yes. Use `--format json` to pipe structured results into any LLM tool:

```bash
flowmap search "auth middleware" --format json
flowmap cat src/auth.py --repo my-service --format json
flowmap map --format json
```

### What if Ollama is too slow?

- Use a GPU-accelerated Ollama install for faster embeddings
- Use `--mode keyword` or `--mode symbol` for searches that don't need embeddings
- The default model (`qwen3-embedding:0.6b`) is small and fast -- larger models are more accurate but slower

### Can I add support for a new language?

If tree-sitter has a grammar for your language, yes. Add the grammar package to `pyproject.toml`, register the extension mapping in `flowmap/parsing/languages.py`, and define symbol extraction rules in `flowmap/parsing/chunker.py`. PRs welcome.

---

## Contributing

Contributions are welcome. Here's how to get started:

### Setup

```bash
git clone https://github.com/aniket-agi/flowmap-cli.git
cd flowmap-cli
uv sync --extra dev
```

### Run tests before changing anything

```bash
uv run pytest tests/ -v
```

All 291 tests should pass. If they don't, your environment has an issue -- fix that first.

### Making changes

1. **Create a branch** from `master`
2. **Write your code** -- follow the existing style (4-space indent, no docstrings on obvious functions, no unnecessary abstractions)
3. **Add tests** for any new behavior -- look at existing tests for patterns
4. **Run the full test suite** -- `uv run pytest tests/ -v`
5. **Lint** -- `uv run ruff check flowmap/`
6. **Open a PR** with a clear description of what and why

### What makes a good PR

- **Bug fixes** with a test that would have caught the bug
- **New tree-sitter language grammars** (Python, TS, Go, Java are done -- Rust, C, Ruby are not)
- **Performance improvements** with before/after measurements
- **Better error messages** for common failure modes

### What to avoid

- Don't add features nobody asked for -- open an issue first to discuss
- Don't refactor working code for style preferences
- Don't add dependencies without a strong reason
- Don't break the `--format json` contract (other tools depend on it)

### Project structure

```
flowmap/
  cli.py                  # Click commands (entry point)
  config.py               # YAML config loading, defaults
  store.py                # LanceDB vector store
  state.py                # SQLite metadata (indexed SHAs, pending markers)
  embeddings.py           # Ollama + sentence-transformers backends
  indexer.py              # File walking, chunking orchestration
  reindex.py              # Incremental reindex via git diff
  render.py               # Output formatting (text + JSON)
  parsing/
    chunker.py            # Tree-sitter AST chunking
    languages.py          # Grammar registry
  search/
    hybrid.py             # 3-way fusion + RRF + reranking
    ripgrep.py            # ripgrep subprocess wrapper
  services/
    indexing.py            # Index orchestration (full + incremental)
    file_resolver.py       # Resolve file paths across repos
    symbol_lookup.py       # Symbol resolution for --symbol flag
    map_builder.py         # Structural map data builder
  history/
    timeline.py            # Temporal query orchestration
    structural_diff.py     # AST-level diffing between versions
    git_ops.py             # Git log, show, pickaxe wrappers
tests/
  conftest.py              # Shared fixtures (MockBackend, hash_vector)
  test_chunker.py          # 49 tests -- AST parsing for all languages
  test_store_integration.py # 26 tests -- real LanceDB operations
  test_cli_commands.py     # 20 tests -- all CLI commands
  test_end_to_end.py       # 6 tests -- full index->search pipeline
  test_reindex.py          # 18 tests -- incremental reindex + ordering
  test_history.py          # 19 tests -- git ops + structural diff
  ...and more
```

---

## Known limitations

- **Large repos (>50K files):** Memory usage scales with repo size. The entire chunk + embedding set is held in memory during indexing. For very large monorepos, ensure adequate RAM.
- **Windows:** Best-effort support. The advisory file lock behavior differs from Unix. Primary development and testing is on macOS/Linux.
- **AST coverage:** Only Python, TypeScript, JavaScript, Go, Java, YAML, and JSON have full tree-sitter grammars. Other languages fall back to line-based chunking (still indexed, but no symbol extraction).
- **Concurrent access:** Designed for single-user CLI use. Don't run `flowmap index` from multiple terminals simultaneously. Search during indexing may return partial results.

---

## For AI agents

> This section helps AI coding assistants (Claude Code, Cursor, Copilot, Aider, etc.) discover and use FlowMap when it's available on the user's machine.

### What is FlowMap?

FlowMap is a CLI tool installed on this machine that provides cross-repo code search. If the user has configured repos with FlowMap, you can use it to find code faster than reading files manually.

### How to check if FlowMap is available

```bash
flowmap --help        # Check if installed
flowmap status        # Check if repos are indexed
flowmap doctor        # Full health check
```

### How to use FlowMap for code search

```bash
# Find code by meaning (semantic + keyword + symbol fusion)
flowmap search "authentication middleware" --format json

# Find a specific function or class by name
flowmap search "processOrder" --mode symbol --format json

# Grep-style keyword search (no embeddings needed)
flowmap search "TODO" --mode keyword --format json

# Get structural overview of all indexed repos
flowmap map --format json

# Read a specific file with line numbers
flowmap cat src/auth.py --repo my-service --format json

# Jump to a specific symbol's source code
flowmap cat src/auth.py --repo my-service --symbol validateToken --format json

# See what functions changed recently
flowmap history "validateToken" --format json
```

### When to use FlowMap vs reading files directly


| Scenario                             | Use FlowMap                                   | Use file reads                |
| ------------------------------------ | --------------------------------------------- | ----------------------------- |
| "Where is the retry logic?"          | `flowmap search "retry logic"`                | -                             |
| "What does `processOrder` do?"       | `flowmap search "processOrder" --mode symbol` | Then `flowmap cat` the result |
| "Show me all classes in the project" | `flowmap symbols --type class`                | -                             |
| "Read lines 50-100 of auth.py"       | -                                             | Read the file directly        |
| "What changed in auth recently?"     | `flowmap history "auth"`                      | -                             |


### Tips

- Always use `--format json` when calling FlowMap -- it gives structured output you can parse
- `flowmap search` returns results ranked by relevance -- the first result is usually the best
- If `flowmap status` shows "not indexed", the user needs to run `flowmap index` first
- FlowMap searches across ALL configured repos at once -- you don't need to know which repo a function is in

---

## License

MIT
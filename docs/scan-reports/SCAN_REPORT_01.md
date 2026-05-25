SCANNER REPORT — Personal-LLM
==========================================

Session Memory: Fresh — new project detected (bootstrapped AETHER.md Section 18 Session Context)
Project Type: Python CLI / RAG Orchestrator
Root Files:
- [AETHER.md](file:///d:/Git_Personal/Personal-LLM/AETHER.md) (Unified agent metadata, changelog, and session state)
- [AGENTS.md](file:///d:/Git_Personal/Personal-LLM/AGENTS.md) (IDE stub)
- [CLAUDE.md](file:///d:/Git_Personal/Personal-LLM/CLAUDE.md) (IDE stub)
- [README.md](file:///d:/Git_Personal/Personal-LLM/README.md) (Project README)
- [main.py](file:///d:/Git_Personal/Personal-LLM/main.py) (CLI entry point and orchestrator loop)
- [requirements.txt](file:///d:/Git_Personal/Personal-LLM/requirements.txt) (Python dependency manifest)

Key Directories:
- [core/](file:///d:/Git_Personal/Personal-LLM/core/) (Core orchestrator, router, config, and utilities logic)
- [core/knowledge_base/](file:///d:/Git_Personal/Personal-LLM/core/knowledge_base/) (Vector store/ChromaDB integrations, document loading, and parsing)
- [core/tools/](file:///d:/Git_Personal/Personal-LLM/core/tools/) (API and OS execution tools)
- [models/](file:///d:/Git_Personal/Personal-LLM/models/) (LLM model clients and dynamic configuration loaders)
- [data/](file:///d:/Git_Personal/Personal-LLM/data/) (ChromaDB physical store)
- [logs/](file:///d:/Git_Personal/Personal-LLM/logs/) (Operational logs and session artifacts)
- [docs/](file:///d:/Git_Personal/Personal-LLM/docs/) (Typed documentation subdirectories)
- [assets/](file:///d:/Git_Personal/Personal-LLM/assets/) (Media, text, and other reference assets)
- [scripts/](file:///d:/Git_Personal/Personal-LLM/scripts/) (Utility shell scripts)
- [tests/](file:///d:/Git_Personal/Personal-LLM/tests/) (Subsystem verification and regression tests)

Tech Stack:
- Python 3.10+
- ChromaDB (vector database)
- sentence-transformers (for local text embeddings)
- PyTorch & HuggingFace Transformers
- Pydantic & Pydantic-Settings
- httpx (async HTTP requests for Ollama and cloud APIs)

Plan Files: Plan/ not found

Assets: `assets/` exists and is formatted per modern Aether Agent asset taxonomy rules.
- Path: [assets/](file:///d:/Git_Personal/Personal-LLM/assets/)
- Status: NOMINAL

Structural Anomalies: None

Registry Status: IN SYNC
- Rules: 23 installed / 23 on disk
- Foundational Skills: 22 installed / 22 on disk
- Workflows: 19 installed / 19 on disk
- Agent Skills: 23 installed / 23 on disk
- Instincts: 6 installed / 6 on disk

Root Pollution: CLEAN

Orphaned Assets: DETECTED
The following files exist under `assets/` but are not referenced by any `.md` documentation in the project:
1. `assets/images/A_clean_twocolumn_comparison_table_or_splitcard_la_delpmaspu.png` (2.85 MB)
2. `assets/text_files/Creating_Proton_Mail.txt` (624 B)
3. `assets/text_files/How_to_make_venv_in_wsl.txt` (438 B)

Confidence: HIGH

Recommended Next Step: Propose deleting the orphaned assets per 06-asset-pruning instinct.

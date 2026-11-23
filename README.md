# GraphRAG vs Vector RAG

**Build a Knowledge Graph Agent with Neo4j**

A beginner-friendly Colab walkthrough. You build one small prototype from start to finish — ontology, entity resolution, Neo4j and Cypher, hybrid retrieval, and grounded answers with citations — then compare it with ordinary vector RAG on the same data.

Companion video: *GraphRAG vs Vector RAG: Build a Knowledge Graph Agent with Neo4j* (Agentic AI Hub).

## What you will learn

1. What GraphRAG is, with an analogy, before any retrieval code
2. A **vector RAG** baseline that cites its sources
3. A simple **ontology** (people, organisations, products, documents)
4. **Entity resolution** — merging messy names such as Maya Chen and M. Chen
5. Loading the graph into **Neo4j** and querying it with **Cypher**
6. **Hybrid retrieval**: vector search to start, then a few hops on the graph
7. **Grounded answers** that cite source documents, and say "I do not know" when the evidence is thin
8. An **evaluation** against the vector baseline, using real measured numbers
9. When a knowledge graph is **overkill**

## The demo data

The notebook uses **Cedarline Analytics**, a fictional B2B company, with 12 short notes, wiki pages and emails in `data/cedarline/`. Some names are messy on purpose so entity resolution has real work to do.

## Results from the recorded run

These numbers were measured on the eight-question eval set in `data/eval_questions.json`. They are not estimates.

| System | Correct | Accuracy | Citations correct | Average latency |
|--------|---------|----------|-------------------|-----------------|
| Vector RAG | 7 / 8 | 87.5% | 8 / 8 | 0.86 s |
| GraphRAG | 8 / 8 | 100% | 8 / 8 | 0.93 s |

The question vector RAG missed was a multi-hop one: *Who manages the Cedarline person who signed the Harborline contract?* The contract and the reporting line live in different documents. Vector search retrieved the contract and the CEO's biography, but not the "reports to" note, so the model correctly said it did not know. GraphRAG walked `Maya Chen -[:REPORTS_TO]-> Ken Adeyemi` and answered. A policy question about service credits was a tie — both systems were right, and the graph was not needed.

Entity resolution merged **17** extracted names into **14** nodes (3 merges), then loaded **27** relationships and **12** documents into Neo4j.

## How to run it

1. Open `GraphRAG_vs_Vector_RAG.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Add your Anthropic API key as a Colab Secret named `ANTHROPIC_API_KEY`, and turn on notebook access. The notebook never prints the key.
3. Run every cell. The first setup cell starts a **local Neo4j Community** instance inside the runtime, so you do not need an Aura account. If you already have Aura, set `NEO4J_URI`, `NEO4J_USER` and `NEO4J_PASSWORD` in Secrets instead.

Locally:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Install a JRE (Java 21) and start Neo4j, or let the notebook start one.
jupyter notebook GraphRAG_vs_Vector_RAG.ipynb
```

The LLM is Anthropic Claude (`claude-haiku-5-5` by default). Change the `MODEL` constant in `src/graphrag_demo.py` if you want a different Claude model. Embeddings use a local `all-MiniLM-L6-v2` model, so no separate embedding API key is required.

## Repo layout

- `GraphRAG_vs_Vector_RAG.ipynb` — the walkthrough
- `src/graphrag_demo.py` — helpers used by the notebook
- `data/cedarline/` — the 12 demo documents
- `data/eval_questions.json` — the eight evaluation questions and gold checks

## Licence

Demo documents are fictional and original to this tutorial. Use the code freely for learning.

"""GraphRAG vs Vector RAG demo helpers. Never prints API keys."""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "cedarline"
EVAL_PATH = Path(__file__).resolve().parents[1] / "data" / "eval_questions.json"

MODEL = "claude-haiku-5-5"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Local demo password for the Colab / box Neo4j instance. Not a secret.
NEO4J_URI_DEFAULT = "bolt://127.0.0.1:7687"
NEO4J_USER_DEFAULT = "neo4j"
NEO4J_PASSWORD_DEFAULT = "graphpass1"

CANONICAL = {
    "m. chen": "Maya Chen",
    "m chen": "Maya Chen",
    "cedar line inc": "Cedarline Analytics",
    "cedar line": "Cedarline Analytics",
    "cedarline": "Cedarline Analytics",
    "bluepeak": "BluePeak Cloud",
    "blue peak cloud": "BluePeak Cloud",
    "blue peak": "BluePeak Cloud",
    "harborline": "Harborline Retail",
    "northwind": "Northwind Goods",
    "pulse board": "PulseBoard",
    "pulse mail": "PulseMail",
}

REL_TYPES = {
    "WORKS_AT", "REPORTS_TO", "LEADS", "HOSTS", "RUNS_ON",
    "SIGNED", "CUSTOMER_OF", "FORMERLY_WORKED_AT", "MENTIONS",
}


def load_key() -> bool:
    """Load ANTHROPIC_API_KEY from Colab secrets or the environment. Never print it."""
    try:
        from google.colab import userdata  # type: ignore
        key = userdata.get("ANTHROPIC_API_KEY")
        if key:
            os.environ["ANTHROPIC_API_KEY"] = key
    except Exception:
        pass
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def get_client():
    import anthropic
    return anthropic.Anthropic()



def message_text(msg) -> str:
    parts = []
    for block in msg.content:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "\n".join(parts).strip()


def load_docs(data_dir: Path | None = None) -> list[dict]:
    data_dir = data_dir or DATA_DIR
    docs = []
    for path in sorted(data_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        src = "UNKNOWN"
        title = path.stem
        m = re.search(r"^Source:\s*(DOC\d+)", text, re.M)
        if m:
            src = m.group(1)
        t = re.search(r"^Title:\s*(.+)$", text, re.M)
        if t:
            title = t.group(1).strip()
        body = re.sub(r"^Title:.*\nSource:.*\n\n?", "", text)
        docs.append({"id": src, "title": title, "path": path.name, "text": body.strip()})
    return docs


def chunk_docs(docs: list[dict]) -> list[dict]:
    """One document is already short, so each doc is one chunk."""
    chunks = []
    for d in docs:
        chunks.append({
            "chunk_id": d["id"] + "_c0",
            "doc_id": d["id"],
            "title": d["title"],
            "text": d["text"],
        })
    return chunks


_embedder = None

def embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def embed_texts(texts: list[str]):
    import numpy as np
    vecs = embedder().encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vecs, dtype="float32")


def vector_search(query: str, chunks: list[dict], matrix, k: int = 4) -> list[dict]:
    import numpy as np
    q = embed_texts([query])[0]
    scores = matrix @ q
    order = np.argsort(-scores)[:k]
    out = []
    for i in order:
        item = dict(chunks[int(i)])
        item["score"] = float(scores[int(i)])
        out.append(item)
    return out


def normalize_name(name: str) -> str:
    s = name.strip()
    key = re.sub(r"[^a-z0-9 ]+", " ", s.lower())
    key = re.sub(r"\s+", " ", key).strip()
    if key in CANONICAL:
        return CANONICAL[key]
    # try dropping corporate suffixes
    key2 = re.sub(r"\b(inc|ltd|llc|analytics|company|co)\b", "", key)
    key2 = re.sub(r"\s+", " ", key2).strip()
    if key2 in CANONICAL:
        return CANONICAL[key2]
    return s


def extract_from_doc(client, doc: dict) -> dict:
    prompt = f"""Extract entities and relationships from this company document.
Return ONLY valid JSON with this shape:
{{"entities":[{{"name":"...","type":"Person|Organization|Product","aliases":[]}}],
 "relations":[{{"source":"...","rel":"WORKS_AT|REPORTS_TO|LEADS|HOSTS|RUNS_ON|SIGNED|CUSTOMER_OF|FORMERLY_WORKED_AT","target":"..."}}]}}
Rules:
- Use the most complete names you see.
- rel must be one of the listed types.
- HOSTS means a vendor hosts a product. RUNS_ON means a product runs on a vendor or servers.
- SIGNED is a person signing a contract involving an organization.
- LEADS means a person leads a team or organization.
Document id: {doc['id']}
Title: {doc['title']}
Text:
{doc['text']}
"""
    msg = client.messages.create(
        model=MODEL,
        max_tokens=800,
        thinking={"type": "disabled"},
        messages=[{"role": "user", "content": prompt}],
    )
    raw = message_text(msg)
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.S)
        data = json.loads(m.group(0)) if m else {"entities": [], "relations": []}
    data["_usage"] = {
        "input_tokens": msg.usage.input_tokens,
        "output_tokens": msg.usage.output_tokens,
    }
    data["_doc"] = doc["id"]
    return data


def extract_all(client, docs: list[dict]) -> tuple[list[dict], list[dict], dict]:
    entities, relations = [], []
    usage = {"input_tokens": 0, "output_tokens": 0}
    for doc in docs:
        data = extract_from_doc(client, doc)
        usage["input_tokens"] += data["_usage"]["input_tokens"]
        usage["output_tokens"] += data["_usage"]["output_tokens"]
        for e in data.get("entities", []):
            e["source_doc"] = doc["id"]
            entities.append(e)
        for r in data.get("relations", []):
            r["source_doc"] = doc["id"]
            relations.append(r)
    return entities, relations, usage


def resolve_entities(entities: list[dict], relations: list[dict]) -> tuple[list[dict], list[dict], dict]:
    from collections import defaultdict
    from rapidfuzz import fuzz

    # first pass: canonical map + type
    raw_n = len({(e.get("name") or "").strip() for e in entities if e.get("name")})
    groups: dict[str, dict] = {}
    for e in entities:
        name = (e.get("name") or "").strip()
        if not name:
            continue
        typ = e.get("type") or "Entity"
        canon = normalize_name(name)
        aliases = set(e.get("aliases") or [])
        aliases.add(name)
        if name != canon:
            aliases.add(canon)
        key = (typ, canon.lower())
        # fuzzy merge into existing same-type name
        matched = None
        for (t, ck), g in groups.items():
            if t != typ:
                continue
            if fuzz.token_sort_ratio(canon.lower(), ck) >= 90:
                matched = (t, ck)
                break
        if matched is None:
            groups[key] = {
                "name": canon,
                "type": typ,
                "aliases": set(aliases),
                "source_docs": {e["source_doc"]},
            }
        else:
            g = groups[matched]
            g["aliases"] |= aliases
            g["source_docs"].add(e["source_doc"])
            if len(canon) > len(g["name"]):
                g["name"] = canon

    resolved = []
    alias_to_name = {}
    for g in groups.values():
        item = {
            "name": g["name"],
            "type": g["type"],
            "aliases": sorted(a for a in g["aliases"] if a and a != g["name"]),
            "source_docs": sorted(g["source_docs"]),
        }
        resolved.append(item)
        alias_to_name[g["name"].lower()] = g["name"]
        for a in g["aliases"]:
            alias_to_name[a.lower()] = g["name"]
            alias_to_name[normalize_name(a).lower()] = g["name"]

    def resolve_name(n: str) -> str | None:
        if not n:
            return None
        k = n.strip().lower()
        if k in alias_to_name:
            return alias_to_name[k]
        nk = normalize_name(n).lower()
        if nk in alias_to_name:
            return alias_to_name[nk]
        # last chance fuzzy
        best, score = None, 0
        for a, canon in alias_to_name.items():
            s = fuzz.token_sort_ratio(k, a)
            if s > score:
                best, score = canon, s
        return best if score >= 88 else n.strip()

    clean_rels = []
    for r in relations:
        src = resolve_name(r.get("source") or "")
        tgt = resolve_name(r.get("target") or "")
        rel = (r.get("rel") or "MENTIONS").upper()
        if rel not in REL_TYPES:
            rel = "MENTIONS"
        if src and tgt and src != tgt:
            clean_rels.append({
                "source": src,
                "rel": rel,
                "target": tgt,
                "source_doc": r["source_doc"],
            })

    stats = {
        "entities_before": raw_n,
        "entities_after": len(resolved),
        "relations": len(clean_rels),
        "merges": raw_n - len(resolved),
    }
    return resolved, clean_rels, stats


def connect_neo4j(uri=None, user=None, password=None):
    from neo4j import GraphDatabase
    uri = uri or os.environ.get("NEO4J_URI") or NEO4J_URI_DEFAULT
    user = user or os.environ.get("NEO4J_USER") or NEO4J_USER_DEFAULT
    password = password or os.environ.get("NEO4J_PASSWORD") or NEO4J_PASSWORD_DEFAULT
    driver = GraphDatabase.driver(uri, auth=(user, password))
    driver.verify_connectivity()
    return driver


def load_graph(*args, **kwargs):
    return load_graph_clean(*args, **kwargs)


def load_graph_clean(driver, entities: list[dict], relations: list[dict], docs: list[dict]) -> dict:
    """Wipe and load the demo graph. Each relationship keeps its source document id."""
    with driver.session() as s:
        s.run("MATCH (n) DETACH DELETE n")
        for d in docs:
            s.run(
                "MERGE (d:Document {id: $id}) SET d.title = $title, d.text = $text",
                id=d["id"], title=d["title"], text=d["text"],
            )
        for e in entities:
            lab = {"Person": "Person", "Organization": "Organization", "Product": "Product"}.get(e["type"], "Entity")
            s.run(
                f"MERGE (n:{lab} {{name: $name}}) "
                "SET n.type = $type, n.aliases = $aliases, n.source_docs = $docs",
                name=e["name"], type=e["type"], aliases=e["aliases"], docs=e["source_docs"],
            )
            for doc_id in e["source_docs"]:
                s.run(
                    f"MATCH (n:{lab} {{name: $name}}), (d:Document {{id: $doc}}) "
                    "MERGE (n)-[:MENTIONED_IN]->(d)",
                    name=e["name"], doc=doc_id,
                )
        for r in relations:
            rel = r["rel"] if r["rel"] in REL_TYPES else "MENTIONS"
            s.run(
                f"MATCH (a {{name: $src}}), (b {{name: $tgt}}) "
                f"MERGE (a)-[x:{rel} {{source_doc: $doc}}]->(b)",
                src=r["source"], tgt=r["target"], doc=r["source_doc"],
            )
        rec = s.run(
            "OPTIONAL MATCH (n) WHERE NOT n:Document WITH count(n) AS entities "
            "OPTIONAL MATCH ()-[r]->() WHERE type(r) <> 'MENTIONED_IN' WITH entities, count(r) AS rels "
            "OPTIONAL MATCH (d:Document) RETURN entities, rels, count(d) AS docs"
        ).single()
        return {"entities": rec["entities"], "relations": rec["rels"], "documents": rec["docs"]}


def cypher_multihop_host_lead(driver) -> list[dict]:
    q = """
    MATCH path = (p:Product {name: 'PulseBoard'})-[:RUNS_ON|HOSTS*1..2]-(v:Organization)
                 -[:LEADS|WORKS_AT*1..2]-(person:Person)
    RETURN p.name AS product, v.name AS vendor, person.name AS person,
           [rel IN relationships(path) | type(rel)] AS hops
    LIMIT 10
    """
    # HOSTS direction may be vendor-HOSTS->product, so try a more forgiving query
    q = """
    MATCH (p:Product)
    WHERE p.name = 'PulseBoard'
    MATCH (p)-[*1..2]-(v:Organization)
    WHERE v.name CONTAINS 'BluePeak'
    MATCH (v)-[*1..2]-(person:Person)
    RETURN DISTINCT p.name AS product, v.name AS vendor, person.name AS person
    LIMIT 20
    """
    with driver.session() as s:
        return [dict(r) for r in s.run(q)]


def graph_expand(driver, seed_names: list[str], hops: int = 2) -> list[dict]:
    if not seed_names:
        return []
    q = f"""
    MATCH (n)
    WHERE n.name IN $names
    MATCH path = (n)-[r*1..{int(hops)}]-(m)
    WHERE ALL(rel IN r WHERE type(rel) <> 'MENTIONED_IN')
    WITH relationships(path) AS rels, nodes(path) AS nodes
    UNWIND rels AS rel
    WITH startNode(rel) AS a, endNode(rel) AS b, rel
    WHERE a.name IS NOT NULL AND b.name IS NOT NULL
    RETURN DISTINCT a.name AS source, type(rel) AS rel, b.name AS target,
           rel.source_doc AS source_doc
    LIMIT 80
    """
    with driver.session() as s:
        return [dict(r) for r in s.run(q, names=seed_names)]


def names_in_text(text: str, entities: list[dict]) -> list[str]:
    found = []
    low = text.lower()
    for e in entities:
        candidates = [e["name"], *e.get("aliases", [])]
        if any(c.lower() in low for c in candidates if c and len(c) > 2):
            found.append(e["name"])
    return found


def hybrid_retrieve(query: str, chunks, matrix, entities, driver, k: int = 3, hops: int = 2) -> dict:
    hits = vector_search(query, chunks, matrix, k=k)
    seeds = []
    for h in hits:
        seeds.extend(names_in_text(h["text"], entities))
    # also match entity names against the query itself
    seeds.extend(names_in_text(query, entities))
    seeds = list(dict.fromkeys(seeds))
    facts = graph_expand(driver, seeds, hops=hops)
    return {"chunks": hits, "seeds": seeds, "facts": facts}


def format_vector_context(hits: list[dict]) -> str:
    parts = []
    for h in hits:
        parts.append(f"[{h['doc_id']}] {h['title']}\n{h['text']}")
    return "\n\n".join(parts)


def format_graph_context(bundle: dict) -> str:
    lines = ["Retrieved document chunks:"]
    for h in bundle["chunks"]:
        lines.append(f"[{h['doc_id']}] {h['title']}\n{h['text']}")
    lines.append("\nGraph facts (source -> relationship -> target, with source document):")
    if not bundle["facts"]:
        lines.append("(no connected facts)")
    for f in bundle["facts"]:
        doc = f.get("source_doc") or "?"
        lines.append(f"- {f['source']} -[{f['rel']}]-> {f['target']} ({doc})")
    return "\n".join(lines)


def answer_with_claude(client, question: str, context: str, mode: str) -> dict:
    if mode == "vector":
        system = (
            "You answer using only the document chunks provided. "
            "Cite source ids like [DOC07] after the facts you use. "
            "If the chunks are not enough, say you do not know. Do not invent people or links."
        )
    else:
        system = (
            "You answer using only the retrieved chunks and graph facts provided. "
            "Cite source ids like [DOC07] after the facts you use. "
            "If the evidence is not enough, say you do not know. Do not invent people or links."
        )
    t0 = time.perf_counter()
    msg = client.messages.create(
        model=MODEL,
        max_tokens=400,
        thinking={"type": "disabled"},
        system=system,
        messages=[{"role": "user", "content": f"Question: {question}\n\nEvidence:\n{context}"}],
    )
    latency = time.perf_counter() - t0
    text = message_text(msg)
    cites = sorted(set(re.findall(r"DOC\d+", text)))
    return {
        "answer": text,
        "citations": cites,
        "latency_s": round(latency, 3),
        "input_tokens": msg.usage.input_tokens,
        "output_tokens": msg.usage.output_tokens,
    }


def score_answer(item: dict, result: dict) -> dict:
    ans = (result.get("answer") or "").lower()
    ok = all(tok.lower() in ans for tok in item.get("must_contain") or [])
    # "does pulsemail run on bluepeak" must also not claim yes without no
    if item["id"] == "Q5" and not re.search(r"\bno\b", ans):
        ok = False
    # If the model declines to answer, do not count a passing mention of a name.
    if re.search(r"\b(i do not know|i don't know|i cannot tell|not (enough|sufficient) (evidence|information))\b", ans):
        ok = False
    gold_docs = set(item.get("gold_docs") or [])
    cites = set(result.get("citations") or [])
    cite_ok = bool(cites & gold_docs) if gold_docs else True
    return {
        "id": item["id"],
        "kind": item["kind"],
        "correct": bool(ok),
        "citation_ok": bool(cite_ok),
        "latency_s": result.get("latency_s"),
        "citations": result.get("citations") or [],
        "answer": result.get("answer") or "",
    }


def load_eval_questions() -> list[dict]:
    return json.loads(EVAL_PATH.read_text(encoding="utf-8"))


def start_or_connect_neo4j():
    """Connect to a local Neo4j, or start Community Edition in this environment.
    On Colab this downloads Neo4j 5.26.5 and installs a JRE if needed.
    An optional Aura instance can be used by setting NEO4J_URI and NEO4J_PASSWORD
    in Colab Secrets or the environment. Never prints credentials.
    """
    import os, shutil, socket, subprocess, tarfile, urllib.request
    from pathlib import Path

    if os.environ.get("NEO4J_URI"):
        return connect_neo4j()

    def port_open(host, port):
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            return False

    if port_open("127.0.0.1", 7687):
        return connect_neo4j()

    java = shutil.which("java")
    if not java:
        subprocess.check_call(["apt-get", "update", "-qq"])
        subprocess.check_call(["apt-get", "install", "-y", "-qq", "openjdk-21-jre-headless"])
    os.environ.setdefault("JAVA_HOME", "/usr/lib/jvm/java-21-openjdk-amd64")

    version = "5.26.5"
    home = Path("/content") if Path("/content").exists() else Path.home() / "neo4j-demo"
    home.mkdir(parents=True, exist_ok=True)
    neo = home / f"neo4j-community-{version}"
    if not neo.exists():
        tgz = home / f"neo4j-community-{version}-unix.tar.gz"
        url = f"https://dist.neo4j.org/neo4j-community-{version}-unix.tar.gz"
        if not tgz.exists():
            print("Downloading Neo4j Community", version, "...")
            urllib.request.urlretrieve(url, tgz)
        with tarfile.open(tgz) as t:
            t.extractall(home)
        conf = neo / "conf" / "neo4j.conf"
        conf.write_text(conf.read_text() + (
            "\nserver.memory.heap.initial_size=512m\n"
            "server.memory.heap.max_size=512m\n"
            "server.memory.pagecache.size=256m\n"
            "server.default_listen_address=127.0.0.1\n"
            "server.bolt.listen_address=127.0.0.1:7687\n"
            "server.http.listen_address=127.0.0.1:7474\n"
        ))
        subprocess.check_call([str(neo / "bin" / "neo4j-admin"), "dbms", "set-initial-password", NEO4J_PASSWORD_DEFAULT])
    log = subprocess.check_output([str(neo / "bin" / "neo4j"), "start"], text=True)
    print(log.strip())
    for _ in range(40):
        if port_open("127.0.0.1", 7687):
            break
        time.sleep(1)
    else:
        raise RuntimeError("Neo4j did not open port 7687")
    return connect_neo4j()

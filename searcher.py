# searcher.py
# Final searcher:
#  - VSM (lnc.ltc)
#  - Soundex canonical mapping
#  - WordNet expansion (stopword filtered)
#  - Query normalization + stopword removal (prevents 'is', 'the' etc. from being used)
#  - Headline bold, direct tokens highlighted YELLOW, synonym tokens GREEN
#  - Snippets: up to max_lines (default 2), each trimmed to a context window

import re
import math
import heapq
import pickle
import html
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

# project tokenizer + soundex
from utils import tokenize
from soundex import soundex

# optional WordNet + stopwords from nltk
try:
    from nltk.corpus import wordnet as wn
except Exception:
    wn = None

try:
    from nltk.corpus import stopwords as _sw
    STOPWORDS = set(_sw.words("english"))
except Exception:
    # conservative fallback stopword list if nltk stopwords aren't installed
    STOPWORDS = set([
        "a", "an", "the", "is", "are", "am", "was", "were", "be", "being", "been",
        "do", "does", "did", "of", "in", "on", "at", "and", "or", "to", "for",
        "with", "by", "from", "that", "this", "these", "those", "it", "its", "as",
        "i", "you", "he", "she", "they", "we", "us", "them", "my", "your", "his",
        "her"
    ])

_token_re = re.compile(r"\b[a-zA-Z]+\b")

# ----- parameters -----
SYNONYM_WEIGHT = 0.7
MAX_SYNS = 2
PROXIMITY_WINDOW = 8
PROXIMITY_BOOST = 1.2
TITLE_BOOST = 1.3
PHRASE_BOOST = 1.3

SNIPPET_CONTEXT_CHARS = 120
DEFAULT_MAX_LINES = 2

# ----------------- utilities -----------------

def normalize_query_text(q: str) -> str:
    """Normalize the raw query string:
       - remove possessive 's
       - remove non-alphanumeric (keep spaces)
       - collapse whitespace
    """
    if not q:
        return ""
    q = re.sub(r"\'s\b", "", q, flags=re.IGNORECASE)             # Dell's -> Dell
    q = re.sub(r"[^0-9A-Za-z\s]", " ", q)                        # punctuation -> space
    q = re.sub(r"\s+", " ", q).strip()
    return q

def _is_valid_token(t: str) -> bool:
    """Return True if token should be kept for query processing."""
    if not t:
        return False
    if t.lower() in STOPWORDS:
        return False
    if len(t) <= 1:            # drop single-letter tokens (like 's' leftover)
        return False
    if t.isdigit():            # drop pure numbers by default
        return False
    return True

# ----------------- soundex index builder -----------------

def ensure_soundex_index(index_data: dict):
    """
    Build index_data['soundex_index'] = {code: {'rep': canonical_term, 'bucket': [terms...]}}
    rep is the highest-df term in the bucket.
    """
    if not index_data:
        return
    if "soundex_index" in index_data and isinstance(index_data["soundex_index"], dict):
        # guessed good shape — but if bucket entries are strings rather than dicts we'll rebuild
        some = next(iter(index_data["soundex_index"].values()), None)
        if isinstance(some, dict):
            return

    postings = index_data.get("postings", {})
    df = index_data.get("df", {})
    bucket = defaultdict(list)
    for term in postings.keys():
        if not term:
            continue
        tl = term.lower()
        code = soundex(tl)
        bucket[code].append(tl)

    sidx = {}
    for code, terms in bucket.items():
        unique = list(set(terms))
        unique.sort(key=lambda t: df.get(t, 0), reverse=True)
        rep = unique[0] if unique else None
        sidx[code] = {"rep": rep, "bucket": unique}
    index_data["soundex_index"] = sidx

# ----------------- WordNet expansion (stopword filtered) -----------------

def expand_with_wordnet(term: str, max_synonyms: int = MAX_SYNS) -> List[str]:
    """
    Return up to max_synonyms WordNet lemmas for `term`, filtered:
     - no multiword lemmas
     - no identical lemma
     - no stopwords
    """
    syns = []
    if wn is None:
        return syns
    try:
        for synset in wn.synsets(term):
            for lemma in synset.lemmas():
                name = lemma.name().lower().replace("_", " ")
                if " " in name:
                    continue
                if name == term:
                    continue
                if name in STOPWORDS:
                    continue
                if name not in syns:
                    syns.append(name)
                if len(syns) >= max_synonyms:
                    break
            if len(syns) >= max_synonyms:
                break
    except Exception:
        return []
    return syns

# ----------------- candidate resolution -----------------

def get_term_candidates(term: str, index_data: dict) -> List[str]:
    """Return list of candidate index terms for a query token:
       - exact match (case sensitive and lowercased)
       - soundex canonical representative if available
    """
    postings = index_data.get("postings", {})
    if term in postings:
        return [term]
    tl = term.lower()
    if tl in postings:
        return [tl]
    ensure_soundex_index(index_data)
    entry = index_data.get("soundex_index", {}).get(soundex(tl))
    if entry and entry.get("rep"):
        return [entry["rep"]]
    return []

# ----------------- query vector builder (ltc) -----------------

def build_query_vector(query: str, index_data: dict, synonym_weight: float = SYNONYM_WEIGHT):
    """
    Build query vector weights (ltc). Return:
      weights (term -> normalized weight),
      synonym_terms (set of mapped synonym terms in vocab),
      raw_synonyms (set of raw synonym strings for highlighting),
      direct_terms (set of direct candidate terms used)
    """

    # normalize + tokenize + filter stopwords & short tokens & digits
    norm_q = normalize_query_text(query)
    tokens = [t for t in tokenize(norm_q) if _is_valid_token(t)]
    if not tokens:
        return {}, set(), set(), set()

    q_tf = Counter(tokens)
    N = index_data.get("N", 1)
    df = index_data.get("df", {})

    weights: Dict[str, float] = {}
    synonym_terms = set()
    raw_synonyms = set()
    direct_terms = set()

    for qterm, raw_tf in q_tf.items():
        tf_q = 1.0 + math.log10(raw_tf) if raw_tf > 0 else 0.0

        # direct or soundex canonical candidate(s)
        exact_candidates = get_term_candidates(qterm, index_data)
        seen = set()
        for cand in exact_candidates:
            cand_l = cand.lower()
            if cand_l in seen:
                continue
            seen.add(cand_l)
            df_c = df.get(cand_l, 0)
            if df_c == 0:
                continue
            idf = math.log10(N / df_c) if df_c else 0.0
            weights[cand_l] = weights.get(cand_l, 0.0) + (tf_q * idf)
            direct_terms.add(cand_l)

        # WordNet synonyms (filtered by STOPWORDS inside expand_with_wordnet)
        synonyms = expand_with_wordnet(qterm)
        for syn in synonyms:
            raw_synonyms.add(syn.lower())
            syn_cands = get_term_candidates(syn, index_data)
            # fallback substring match if none
            if not syn_cands:
                syn_low = syn.lower()
                syn_cands = [t for t in index_data.get("postings", {}).keys() if syn_low in t.lower()]
                if syn_cands:
                    syn_cands.sort(key=lambda t: index_data.get("df", {}).get(t.lower(), 0), reverse=True)
                    syn_cands = [syn_cands[0]]
            # soundex fallback
            if not syn_cands:
                ensure_soundex_index(index_data)
                entry = index_data.get("soundex_index", {}).get(soundex(syn.lower()))
                if entry and entry.get("rep"):
                    syn_cands = [entry["rep"]]

            if not syn_cands:
                continue

            for sc in syn_cands:
                sc_l = sc.lower()
                if sc_l in seen:
                    continue
                seen.add(sc_l)
                df_sc = df.get(sc_l, 0)
                if df_sc == 0:
                    continue
                idf_sc = math.log10(N / df_sc) if df_sc else 0.0
                add_w = tf_q * idf_sc * synonym_weight
                weights[sc_l] = weights.get(sc_l, 0.0) + add_w
                synonym_terms.add(sc_l)

    # normalize (ltc) - cosine normalization
    denom = math.sqrt(sum(v * v for v in weights.values())) if weights else 1.0
    if denom != 0:
        for t in list(weights.keys()):
            weights[t] = weights[t] / denom

    return weights, synonym_terms, raw_synonyms, direct_terms

# ----------------- doc vector helper (lnc) -----------------

def get_doc_vector(doc_id: str, index_data: dict) -> Dict[str, float]:
    tf_map = index_data.get("doc_term_freqs", {}).get(doc_id, {})
    length = index_data.get("doc_lengths", {}).get(doc_id, 1.0)
    vec = {}
    for term, tf in tf_map.items():
        if tf <= 0:
            continue
        w = 1.0 + math.log10(tf)
        vec[term] = (w / length) if length != 0 else w
    return vec

# ----------------- proximity helper -----------------

def _has_proximity(tokens: List[str], doc_text: str, window: int = PROXIMITY_WINDOW) -> bool:
    if len(tokens) < 2:
        return False
    words = _token_re.findall(doc_text.lower())
    pos_map = {}
    for idx, w in enumerate(words):
        if w in pos_map:
            pos_map[w].append(idx)
        elif w in tokens:
            pos_map[w] = [idx]
    tokens_present = [t for t in tokens if t in pos_map]
    for i in range(len(tokens_present)):
        for j in range(i + 1, len(tokens_present)):
            a = tokens_present[i]; b = tokens_present[j]
            for pa in pos_map[a]:
                for pb in pos_map[b]:
                    if abs(pa - pb) <= window:
                        return True
    return False

# ----------------- ranking (lnc docs, ltc query) + boosts + dedupe -----------------

def rank_documents(query: str, index_data: dict, corpus: Dict[str, str] = None, top_k: int = 10):
    postings = index_data.get("postings", {})
    doc_lengths = index_data.get("doc_lengths", {})

    q_weights, synonym_terms, raw_synonyms, direct_terms = build_query_vector(query, index_data)
    if not q_weights:
        return [], set(), set(), set()

    scores = defaultdict(float)
    for term, q_w in q_weights.items():
        if term not in postings:
            continue
        for doc_id, tf in postings[term]:
            doc_w = 1.0 + math.log10(tf) if tf > 0 else 0.0
            denom = doc_lengths.get(doc_id, 1.0)
            doc_w_norm = doc_w / denom if denom != 0 else doc_w
            scores[doc_id] += doc_w_norm * q_w

    if not scores:
        return [], synonym_terms, raw_synonyms, direct_terms

    # tokens used (normalized & filtered) for proximity/title checks
    norm_q = normalize_query_text(query)
    tokens = [t for t in tokenize(norm_q) if _is_valid_token(t)]
    tokens_lower = [t.lower() for t in tokens]
    bigrams = set(" ".join(bg) for bg in zip(tokens_lower, tokens_lower[1:])) if len(tokens_lower) > 1 else set()

    if corpus:
        for doc_id in list(scores.keys()):
            text = corpus.get(doc_id, "").lower()
            first_chunk = text.splitlines()[0] if text.splitlines() else text[:200]
            if any(t in first_chunk for t in tokens_lower):
                scores[doc_id] *= TITLE_BOOST
            for phrase in bigrams:
                if phrase in text:
                    scores[doc_id] *= PHRASE_BOOST
                    break
            if _has_proximity(tokens_lower, text, window=PROXIMITY_WINDOW):
                scores[doc_id] *= PROXIMITY_BOOST

    # dedupe by normalized headline (keep highest scored doc per headline)
    heap = [(-score, doc_id) for doc_id, score in scores.items()]
    heapq.heapify(heap)

    headline_map = {}
    other_docs = []

    while heap:
        neg_score, doc_id = heapq.heappop(heap)
        score = -neg_score
        raw = corpus.get(doc_id, "") if corpus else ""
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        headline = lines[0] if lines else ""
        headline_key = re.sub(r"[^\w\s]", "", headline.lower()).strip()
        headline_key = re.sub(r"\s+", " ", headline_key)
        if headline_key:
            existing = headline_map.get(headline_key)
            if (existing is None) or (score > existing[0]):
                headline_map[headline_key] = (score, doc_id)
        else:
            other_docs.append((score, doc_id))

    candidates = sorted(headline_map.values(), key=lambda x: -x[0])
    other_docs.sort(key=lambda x: -x[0])
    combined = candidates + other_docs

    top = []
    seen_docids = set()
    for score, doc_id in combined:
        if doc_id in seen_docids:
            continue
        top.append((doc_id, score))
        seen_docids.add(doc_id)
        if len(top) >= top_k:
            break

    top.sort(key=lambda x: (-x[1], x[0]))
    return top, synonym_terms, raw_synonyms, direct_terms

# ----------------- snippet extraction & highlighting -----------------

def _find_line_with_match(text: str, tokens: list, return_top_n: int = DEFAULT_MAX_LINES):
    """Return up to return_top_n (line_text, line_index_1based, matched_tokens_set)."""
    if not text:
        return []

    text = html.unescape(text)
    lines = text.splitlines()
    tokens_lower = [t.lower() for t in tokens if t]
    phrase = " ".join(tokens_lower) if len(tokens_lower) > 1 else None

    scored = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        l_low = line.lower()
        if phrase and phrase in l_low:
            matched = set(tokens_lower) & set(_token_re.findall(l_low))
            scored.append((10 + len(matched), line, i + 1, matched))
            continue
        matched = set()
        for t in set(tokens_lower):
            if re.search(rf"\b{re.escape(t)}\b", line, flags=re.IGNORECASE):
                matched.add(t)
        if matched:
            scored.append((len(matched), line, i + 1, matched))

    if scored:
        scored.sort(key=lambda x: (-x[0], x[2]))
        return [(line, idx, matched) for score, line, idx, matched in scored[:return_top_n]]

    # fallback: first non-empty lines
    out = []
    for i, line in enumerate(lines):
        if line.strip():
            out.append((line, i + 1, set()))
            if len(out) >= return_top_n:
                break
    return out

def _shorten_line_to_context(line: str, matches: List[Tuple[int,int]], max_chars: int = SNIPPET_CONTEXT_CHARS):
    """Return a shortened string including match spans around context windows."""
    if not matches:
        s = line.strip()
        return s if len(s) <= max_chars else (s[:max_chars].rstrip() + " ...")
    parts = []
    cur_start = max(0, matches[0][0] - max_chars//4)
    cur_end = min(len(line), matches[0][1] + max_chars//4)
    for st, ed in matches[1:]:
        if st <= cur_end + (max_chars//4):
            cur_end = min(len(line), ed + max_chars//4)
        else:
            parts.append((cur_start, cur_end))
            cur_start = max(0, st - max_chars//4)
            cur_end = min(len(line), ed + max_chars//4)
    parts.append((cur_start, cur_end))
    snippets = []
    total = 0
    for a,b in parts:
        seg = line[a:b].strip()
        if len(seg) > max_chars:
            seg = seg[:max_chars].rstrip() + " ..."
        snippets.append(seg)
        total += len(seg)
        if total > max_chars * 2:
            break
    return " ... ".join(snippets)

def _highlight_matches_in_line(line: str, direct_tokens:set, synonym_tokens:set):
    """
    Highlight direct tokens YELLOW and synonyms GREEN.
    Avoid overlapping replacements.
    """
    if not (direct_tokens or synonym_tokens):
        return line

    low = line.lower()
    spans = []  # (start, end, type, text)
    # direct tokens first (priority)
    for t in sorted(direct_tokens, key=lambda x:-len(x)):
        for m in re.finditer(rf"\b{re.escape(t)}\b", low, flags=re.IGNORECASE):
            spans.append((m.start(), m.end(), "direct", line[m.start():m.end()]))
    # synonym tokens
    for t in sorted(synonym_tokens, key=lambda x:-len(x)):
        for m in re.finditer(rf"\b{re.escape(t)}\b", low, flags=re.IGNORECASE):
            spans.append((m.start(), m.end(), "syn", line[m.start():m.end()]))

    if not spans:
        return line

    spans.sort(key=lambda x: (x[0], -(x[1]-x[0])))
    merged = []
    last_end = -1
    for s,e,typ,text in spans:
        if s >= last_end:
            merged.append((s,e,typ,text))
            last_end = e
    parts = []
    idx = 0
    for s,e,typ,text in merged:
        if idx < s:
            parts.append(line[idx:s])
        if typ == "direct":
            parts.append(f"\033[92m{text}\033[0m")  # yellow
        else:
            parts.append(f"\033[93m{text}\033[0m")  # green
        idx = e
    if idx < len(line):
        parts.append(line[idx:])
    return "".join(parts)

def highlight_line_with_colors(text: str, query: str, synonym_terms=None, raw_synonyms=None, max_lines:int = DEFAULT_MAX_LINES):
    """
    Return (rendered_headline (bold), rendered_lines [(line_no, rendered_line, matched_set), ...])
    """
    if not text:
        return ("\033[1m(no title)\033[0m", [])

    synonym_terms = synonym_terms or set()
    raw_synonyms = raw_synonyms or set()

    text = html.unescape(text)
    lines = text.splitlines()
    raw_headline = "(no title)"
    for ln in lines:
        if ln.strip():
            raw_headline = ln.strip()
            break

    rendered_headline = f"\033[1m{raw_headline}\033[0m"

    # normalized & filtered direct tokens for highlighting
    norm_q = normalize_query_text(query)
    direct_set = set(t.lower() for t in tokenize(norm_q) if _is_valid_token(t))

    syn_set = set(synonym_terms) | set(raw_synonyms)

    top_lines = _find_line_with_match(text, [t for t in tokenize(norm_q) if _is_valid_token(t)], return_top_n=max_lines)

    rendered = []
    if top_lines:
        for line, ln_no, matched_set in top_lines:
            low = line.lower()
            match_spans = []
            for t in direct_set:
                for m in re.finditer(rf"\b{re.escape(t)}\b", low, flags=re.IGNORECASE):
                    match_spans.append((m.start(), m.end()))
            for t in syn_set:
                for m in re.finditer(rf"\b{re.escape(t)}\b", low, flags=re.IGNORECASE):
                    match_spans.append((m.start(), m.end()))
            match_spans.sort()

            shortened = _shorten_line_to_context(line, match_spans, max_chars=SNIPPET_CONTEXT_CHARS)

            direct_here = {t for t in direct_set if re.search(rf"\b{re.escape(t)}\b", shortened, flags=re.IGNORECASE)}
            syn_here = {t for t in syn_set if re.search(rf"\b{re.escape(t)}\b", shortened, flags=re.IGNORECASE)}

            highlighted_line = _highlight_matches_in_line(shortened, direct_here, syn_here)
            rendered.append((ln_no, highlighted_line, matched_set))
    else:
        s = raw_headline
        if len(s) > SNIPPET_CONTEXT_CHARS:
            s = s[:SNIPPET_CONTEXT_CHARS].rstrip() + " ..."
        rendered.append((1, s, set()))

    return rendered_headline, rendered

# ----------------- wrapper: search + snippet -----------------

def search_and_snippet(query: str, corpus: Dict[str, str], index_data: dict, top_k: int = 10, max_lines:int = DEFAULT_MAX_LINES):
    # normalize & filter query for display and processing
    norm_q = normalize_query_text(query)
    toks = [t for t in tokenize(norm_q) if _is_valid_token(t)]

    if toks:
        print("soundex codes:")
        for t in toks:
            try:
                print(f"  {t} -> {soundex(t)}")
            except Exception:
                print(f"  {t} -> (soundex error)")
    else:
        print("soundex codes: (no tokens)")

    ranked, synonym_terms, raw_synonyms, direct_terms = rank_documents(query, index_data, corpus=corpus, top_k=top_k)

    if raw_synonyms:
        # raw_synonyms are shown for debugging/highlighting only
        print("WordNet raw synonyms (used for highlighting):", sorted(raw_synonyms))

    results = []
    for doc_id, score in ranked:
        raw = corpus.get(doc_id, "")
        rendered_headline, rendered_lines = highlight_line_with_colors(raw, query, synonym_terms, raw_synonyms, max_lines=max_lines)
        results.append((doc_id, score, rendered_headline, rendered_lines))
    return results

# ----------------- CLI for quick testing -----------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="searcher cli - vsm + wordnet + soundex (query normalization + stopword filtering)")
    parser.add_argument("--index", required=True, help="path to index pickle (index.pkl)")
    parser.add_argument("--corpus", required=True, help="path to corpus pickle (corpus.pkl)")
    parser.add_argument("--q", required=True, help="query string (wrap in quotes)")
    parser.add_argument("--k", type=int, default=5, help="top-k results")
    parser.add_argument("--lines", type=int, default=DEFAULT_MAX_LINES, help="max snippet lines per doc (2 or 3)")
    args = parser.parse_args()

    with open(args.index, "rb") as f:
        index_data = pickle.load(f)
    with open(args.corpus, "rb") as f:
        corpus = pickle.load(f)

    res = search_and_snippet(args.q, corpus, index_data, top_k=args.k, max_lines=args.lines)
    if not res:
        print("no results")
    else:
        for i, (doc_id, score, headline, line_info) in enumerate(res, start=1):
            print(f"{i}. {headline} (score={score:.4f}) [{doc_id}]")
            for ln_no, rendered_line, _ in line_info:
                print(f"   line {ln_no}: {rendered_line}")
            print()

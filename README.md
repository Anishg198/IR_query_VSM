#  VSM Search Engine (lnc.ltc + Soundex + WordNet) for IR assignment

This assignment implements a **Vector Space Model (VSM)** search engine over the corpus provide.
It supports **interactive querying** with highlighting, synonym expansion, and fuzzy matching.

##  Retrieval Model

We use the **lnc.ltc weighting scheme**:

- **Document weight (lnc):**
$w(d,t) = \frac{1 + \log_{10}(tf_{d,t})}{|d|}$.

- **Query weight (ltc):**

$$
w(q,t) = \left( 1 + \log_{10}\left(tf_{q,t}\right) \right) \times \log_{10}\left( \frac{N}{df_t} \right)
$$


-**Stopword Removal:** stopwords are filtered out from queries before scoring.  
- **Synonyms (WordNet):** scaled by **0.7** of direct query term weight.  
- **Soundex-mapped terms (typos → canonical terms):** treated as full-weight direct terms.  
- **Final score:** cosine similarity between query and document vectors.

##  Features

- **Core model:** Vector Space Model with lnc.ltc weighting.
- **Stopword Removal:** improves accuracy by ignoring common words (`the`, `is`, `of`, etc.).  
- **Soundex fuzzy matching:** query typos map to canonical terms (e.g., `fues` → `fuse`).  
- **WordNet synonym expansion:** expands query with semantic synonyms (0.7 times the weight).  
- **Boosts:**
  - **Title boost** (+30%) if query terms appear in the headline.  
  - **Phrase boost** (+30%) if query phrase occurs verbatim.  
  - **Proximity boost** (+20%) if query terms appear near each other (window ≤ 8).  
- **Output formatting:**
  - Top **5 documents** shown per query.  
  - Top **2 relevant lines** (snippets) shown per document.  
  - **Headlines:** bold.  
  - **Matching words:** green.  
  - **Synonyms:** yellow.  
- **Interactive search:** type queries live, get ranked results instantly.

##  Project Structure

IR_Query_model<br>

| File/Folder      | Description                                      |
|------------------|--------------------------------------------------|
| indexer.py       | builds index.pkl and corpus.pkl                   |
| searcher.py      | core retrieval engine (VSM + Soundex + WordNet)   |
| interactive.py   | interactive console to type queries               |
| run_example.py   | example run with preset queries                   |
| utils.py         | tokenizer and helper functions                    |
| soundex.py       | Soundex implementation                            |
| index.pkl        | built index (after running indexer)               |
| corpus.pkl       | serialized corpus (after running indexer)         |
| README.md        | this file   


---

##  Running the Model  

### 1. Install dependencies  
```bash
pip install nltk colorama
```
### 2.Then download WordNet (only once):
```bash
   import nltk
   nltk.download('wordnet')
   nltk.download('omw-1.4')
```
  
### 3. Build index
```bash
   python indexer.py
```

### 4. Run interactive search
```bash
python interactive.py
```

## 💻 Sample Session

🔎 interactive VSM searcher
type a query and press enter (type 'exit' or 'quit' to stop)

```bash
query > Dassler brothers
soundex codes:
  dassler -> D246
  brothers -> B636
WordNet raw synonyms (used for highlighting): ['brother', 'buddy']

1. What is puma? (score=0.2416) [puma.txt]
   line 5: was founded in 1948 by Rudolf Dassler. In 1924, Rudolf and his brother Adolf Dassler had jointly formed the company...
   line 13: After founding their company, the Dassler brothers fell out and went separate ways...

(4 more search queries)
```
## Novelty Beyond Assignment Requirements

Compared to a basic VSM implementation, this project adds:

- **Soundex Fuzzy Matching** → handles typos and spelling variations.
- **Stopword Removal** → improves precision by ignoring common words.
- **WordNet Synonym Expansion** → improves recall by retrieving semantically related docs.  
- **Boosting Heuristics** → titles, phrases, and proximity are rewarded.  
- **Color-coded Output** → improves readability in terminal:  
  - Headlines = **bold**  
  - Matching words = **green**  
  - Synonyms = **yellow**  
- **Interactive Console** → lets you search dynamically without rerunning scripts.  
- **Line-level Snippets** → shows the most relevant lines instead of whole documents along with first line of the file as the header for more context, which makes it easier for the user.  

##  Authors

Built by *Anish Gupta*, *Prakhar Sethi* and *Ritwik Bhattacharya* as part of **Lab 1 — VSM Information Retrieval** assignment.

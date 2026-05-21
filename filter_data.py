import os
import re
import argparse
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from tqdm import tqdm
from wordfreq import zipf_frequency
from ftfy import fix_text
from unidecode import unidecode
from collections import Counter

# DEFAULTS 

DEFAULT_INPUT          = "overall.csv"
DEFAULT_OUTPUT_DIR     = "new_clean_output"
KEYWORD_COLUMN         = "Secondary Keywords"
PRIMARY_COLUMN         = "Primary Keywords"

MIN_KEYWORD_LENGTH     = 3
MIN_WORD_FREQ          = 1.8
JUNK_WORD_RATIO        = 0.65   
CORPUS_FREQ_FLOOR      = 3  
BRAND_MIN_COUNT        = 2      
TYPO_FREQ_THRESHOLD    = 1.0

# Stop-words
STOP_WORDS = frozenset([
    "in", "at", "the", "a", "an", "for", "with", "by", "of",
    "near", "and", "or", "to", "from", "on", "is", "are", "was",
    "be", "as", "its", "it", "this", "that", "these", "those",
    "how", "what", "why", "when", "where", "which", "who",
    "do", "does", "did", "has", "have", "had", "not", "no",
    "vs", "vs.", "per", "via", "into",
])

ACRONYM_PATTERN = re.compile(r'^[A-Z0-9]{2,6}$')

JUNK_PATTERNS = [
    re.compile(r'^(.)\1{3,}$'),          
    re.compile(r'^\d+$'),                 
    re.compile(r'^[^a-zA-Z0-9]+$'),        
    re.compile(r'\b(xxx|porn|nude)\b', re.I),
    re.compile(r'http[s]?://', re.I),
    re.compile(r'^[a-z0-9]{25,}$'),        
    re.compile(r'\.(com|net|org|io|co)\b', re.I),  
]

# LANGUAGE / SCRIPT 

def detect_script(text: str) -> str:
    if re.search(r'[\u0600-\u06FF]', text):  return "Arabic"
    if re.search(r'[\u4E00-\u9FFF]', text):  return "Chinese"
    if re.search(r'[\u0400-\u04FF]', text):  return "Cyrillic"
    if re.search(r'[\u0900-\u097F]', text):  return "Devanagari"
    if re.search(r'[\u0080-\u024F]', text):  return "Extended-Latin"
    return "Unknown"


def is_any_non_ascii(text: str) -> bool:
    """True if text contains ANY character outside plain ASCII (0x00–0x7F)."""
    try:
        text.encode('ascii')
        return False
    except UnicodeEncodeError:
        return True


def is_transliterated_foreign(text: str) -> bool:
    """
    Catch Spanish/French/Italian keywords that have been stripped of accents
    and thus pass the ASCII check.  Strategy: unidecode the text, re-check
    if the original contained accented chars.  If the unidecoded version equals
    the original we already know it's ASCII; if it differs the word had accents
    stripped upstream — we catch those via is_any_non_ascii.

    Here we focus on a different case: words that are pure ASCII but are
    clearly non-English (e.g. "atencion al cliente", "excelente servicio").
    We detect this by checking if ALL content words score 0 on English zipf
    AND none exist in the domain / corpus vocab.
    This function is called only after domain_vocab and corpus_words are built.
    """
    # Implemented inline in the pipeline where vocab is available.
    pass

# TEXT NORMALISATION

def normalize_text(text: str) -> str:
    """Unicode fix → lowercase → strip non-alphanumeric (keep spaces) → collapse spaces."""
    if pd.isna(text):
        return ""
    text = str(text).strip()
    text = fix_text(text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_primary(label: str) -> str:
    if pd.isna(label):
        return ""
    label = str(label).strip()
    label = re.sub(r'^[\[\(]+|[\]\)]+$', '', label)
    label = re.sub(r'[\.\!\?]+$', '', label)
    label = re.sub(r'\s+', ' ', label).strip()
    return label.lower()

#  DOMAIN VOCABULARY EXTRACTION  

def extract_domain_vocab(series: pd.Series, top_n: int = 500) -> set:
    """
    Build domain vocabulary from the corpus itself.
    Any token appearing >= CORPUS_FREQ_FLOOR times is trusted,
    plus the top_n most frequent tokens.
    """
    word_counter: Counter = Counter()
    for text in series.dropna().astype(str):
        if is_any_non_ascii(text):
            continue
        normed = normalize_text(text)
        for token in normed.split():
            if len(token) >= 2:
                word_counter[token] += 1
    domain = {w for w, c in word_counter.items() if c >= CORPUS_FREQ_FLOOR}
    domain |= {w for w, _ in word_counter.most_common(top_n)}
    return domain


def extract_brand_whitelist(series: pd.Series) -> set:
    """
    Single-token keywords that appear >= BRAND_MIN_COUNT times
    and have low English-dictionary frequency are likely brand/product names.
    Also include any token from multi-word keywords that appears >= 3 times.
    """
    single_counter: Counter = Counter()
    multi_token_counter: Counter = Counter()

    for text in series.dropna().astype(str):
        if is_any_non_ascii(text):
            continue
        normed = normalize_text(text)
        tokens = normed.split()
        if len(tokens) == 1 and len(tokens[0]) >= 3:
            single_counter[tokens[0]] += 1
        for t in tokens:
            if len(t) >= 3:
                multi_token_counter[t] += 1

    brands: set = set()
    # Single-token brands: appear ≥ BRAND_MIN_COUNT and low dict freq
    for w, c in single_counter.items():
        if c >= BRAND_MIN_COUNT and zipf_frequency(w, "en") < 2.5:
            brands.add(w)
    # Multi-token brands: appear ≥ 3 times and low dict freq
    for w, c in multi_token_counter.items():
        if c >= 3 and zipf_frequency(w, "en") < 2.0:
            brands.add(w)
    return brands


def build_corpus_words(series: pd.Series) -> tuple[dict, set]:
    """Return (word_counts dict, corpus_words set) from all ASCII keywords."""
    word_counts: dict = {}
    for kw in series.dropna().astype(str):
        if not is_any_non_ascii(kw):
            for w in normalize_text(kw).split():
                if len(w) >= 2:
                    word_counts[w] = word_counts.get(w, 0) + 1
    corpus_words = {w for w, c in word_counts.items() if c >= CORPUS_FREQ_FLOOR}
    return word_counts, corpus_words

# TOKEN GUARDS  

def is_acronym(word: str) -> bool:
    return bool(ACRONYM_PATTERN.match(word.upper()) and len(word) <= 6)


def token_is_known(word: str, domain_vocab: set, brand_whitelist: set,
                   corpus_words: set, min_freq: float) -> bool:
    """True if a token is recognisable by ANY means."""
    if len(word) <= 2:              return True   
    if word in STOP_WORDS:          return True
    if is_acronym(word):            return True
    if word in brand_whitelist:     return True
    if word in domain_vocab:        return True
    if word in corpus_words:        return True
    if zipf_frequency(word, "en") >= min_freq:  return True
    return False


def looks_like_compound(word: str, domain_vocab: set, brand_whitelist: set,
                         corpus_words: set) -> bool:
    """
    Check if an unknown single token could be a compound of two known words,
    e.g. 'voicebots' = 'voice' + 'bots', 'callbar' = 'call' + 'bar'.
    Tries all split points; returns True if both halves are known English words.
    """
    if len(word) < 5:
        return False
    for i in range(3, len(word) - 2):
        left, right = word[:i], word[i:]
        left_known  = (zipf_frequency(left,  "en") >= 2.0 or left  in domain_vocab
                       or left  in brand_whitelist or left  in corpus_words)
        right_known = (zipf_frequency(right, "en") >= 2.0 or right in domain_vocab
                       or right in brand_whitelist or right in corpus_words)
        if left_known and right_known:
            return True
    return False

# JUNK PATTERNS  

def is_junk_pattern(keyword: str) -> bool:
    for pat in JUNK_PATTERNS:
        if pat.search(keyword):
            return True
    return False


def has_dangling_filler(words: list) -> bool:
    """
    True if the keyword starts or ends with a stop-word/conjunction,
    making it a fragment rather than a real search phrase.
    Examples: 'and customer service', 'customer relationship management and'
    """
    if not words:
        return False
    # Leading stop-word (only flag if it's clearly a fragment, not a question)
    leading_questions = {"how", "what", "why", "when", "where", "which", "who"}
    if words[0] in STOP_WORDS and words[0] not in leading_questions:
        return True
    # Trailing conjunction / preposition
    trailing_bad = {"and", "or", "of", "in", "at", "for", "with", "by",
                    "to", "from", "on", "the", "a", "an"}
    if words[-1] in trailing_bad:
        return True
    return False


def has_repeated_content_word(words: list) -> bool:
    """
    True if any non-stop word appears more than once in the keyword.
    Example: 'callback callback', 'management contact center management'
    """
    content = [w for w in words if w not in STOP_WORDS and len(w) > 2]
    return len(content) != len(set(content))


def is_ascii_foreign(normed: str, domain_vocab: set, brand_whitelist: set,
                     corpus_words: set) -> bool:
    """
    Detect keywords that are entirely non-English but happen to be pure ASCII
    (Spanish/Italian without accents: 'atencion al cliente', 'excelente servicio').
    Heuristic: ALL content words (len > 3) score 0 on English zipf AND none are
    in any vocab.  We require at least 2 content words to avoid false positives.
    """
    content_words = [w for w in normed.split()
                     if len(w) > 3 and w not in STOP_WORDS]
    if len(content_words) < 2:
        return False
    for w in content_words:
        if (zipf_frequency(w, "en") > 2.0
                or w in domain_vocab
                or w in brand_whitelist
                or w in corpus_words):
            return False   
    return True            


# MAIN JUNK DETECTOR  

def classify_keyword(normed: str,
                     domain_vocab: set,
                     brand_whitelist: set,
                     corpus_words: set,
                     single_token_set: set,
                     min_freq: float) -> tuple[bool, str]:
    """
    Returns (should_remove: bool, reason: str).
    Checks in order of cheapest → most expensive.
    """
    # 1. Empty / too short
    if not normed:
        return True, "EMPTY"
    if len(normed) < MIN_KEYWORD_LENGTH:
        return True, "TOO_SHORT"

    # 2. No alphabetic character at all
    if not re.search(r"[a-zA-Z]", normed):
        return True, "NO_ALPHA"

    # 3. Junk patterns (URLs, symbol strings, pure numbers, etc.)
    if is_junk_pattern(normed):
        return True, "JUNK_PATTERN"

    words = normed.split()

    # 4. Dangling filler fragment ("and customer service", "… management and")
    if len(words) >= 2 and has_dangling_filler(words):
        return True, "DANGLING_FILLER"

    # 5. Repeated content word ("callback callback")
    if len(words) >= 2 and has_repeated_content_word(words):
        return True, "REPEATED_WORD"

    # 6. Split-brand noise: a single known corpus token that got split by space
    #    e.g. "talk desk" when "talkdesk" exists in single_token_set
    if len(words) == 2:
        merged = "".join(words)
        if merged in single_token_set and merged not in STOP_WORDS:
            return True, "SPLIT_BRAND_NOISE"

    # 7. ASCII-only foreign language (Spanish/Italian without accents)
    if is_ascii_foreign(normed, domain_vocab, brand_whitelist, corpus_words):
        return True, "NON_ENGLISH_ASCII"

    # 8. Single-token checks
    if len(words) == 1:
        w = words[0]
        if is_acronym(w):
            return False, ""
        if token_is_known(w, domain_vocab, brand_whitelist, corpus_words, min_freq):
            return False, ""
        # Unknown single token — check if it could be a compound word
        if looks_like_compound(w, domain_vocab, brand_whitelist, corpus_words):
            return False, ""
        # Truly unrecognised single token → remove
        return True, "SINGLE_LOW_FREQ"

    # 9. Multi-word: count unknown content-word tokens
    unknown_count = 0
    for w in words:
        if len(w) <= 2 or w in STOP_WORDS:
            continue  # skip short/stop words
        if token_is_known(w, domain_vocab, brand_whitelist, corpus_words, min_freq):
            continue
        if looks_like_compound(w, domain_vocab, brand_whitelist, corpus_words):
            continue
        unknown_count += 1

    content_words = [w for w in words if len(w) > 2 and w not in STOP_WORDS]
    if content_words and (unknown_count / len(content_words)) > JUNK_WORD_RATIO:
        return True, "LOW_QUALITY_WORDS"

    return False, ""

# MAIN PIPELINE  

def run(input_file: str,
        output_dir: str,
        keyword_col: str  = KEYWORD_COLUMN,
        primary_col: str  = PRIMARY_COLUMN,
        min_freq: float   = MIN_WORD_FREQ,
        keep_extended_latin: bool = False):

    os.makedirs(output_dir, exist_ok=True)
    output_clean       = os.path.join(output_dir, "clean_data.csv")
    output_removed     = os.path.join(output_dir, "removed_data.csv")
    output_non_english = os.path.join(output_dir, "non_english_data.csv")
    output_stats       = os.path.join(output_dir, "cluster_stats.csv")

    # Load 
    print(f"\nLoading {input_file} …")
    df = pd.read_csv(input_file)
    for col in [keyword_col, primary_col]:
        if col not in df.columns:
            raise ValueError(f"Missing column: '{col}'  —  found: {list(df.columns)}")

    df[primary_col] = df[primary_col].apply(normalize_primary)

    # Cluster assignment 
    non_empty = df[primary_col].replace("", pd.NA).dropna()
    if len(non_empty) == 0:
        print("  ⚠  Primary column is empty — assigning all rows to cluster 'ALL'")
        df["_cluster"] = "ALL"
    else:
        # Forward-fill cluster label within groups; do NOT collapse distinct values
        df["_cluster"] = (
            df[primary_col]
            .replace("", pd.NA)
            .ffill()
            .fillna("ALL")
        )

    total_input = len(df)
    print(f"  {total_input:,} rows  |  {df['_cluster'].nunique():,} clusters")

    # Build vocabularies 
    print("\nBuilding vocabularies …")
    domain_vocab    = extract_domain_vocab(df[keyword_col])
    brand_whitelist = extract_brand_whitelist(df[keyword_col])
    word_counts, corpus_words = build_corpus_words(df[keyword_col])
    print(f"  Domain vocab    : {len(domain_vocab):,} tokens")
    print(f"  Brand whitelist : {len(brand_whitelist):,} tokens")
    print(f"  Corpus words    : {len(corpus_words):,} tokens")

    # Build set of all single normalised tokens that appear in the corpus
    # (used for split-brand-noise detection)
    single_token_set: set = set()
    for kw in df[keyword_col].dropna().astype(str):
        if not is_any_non_ascii(kw):
            normed = normalize_text(kw)
            if len(normed.split()) == 1 and len(normed) >= 4:
                single_token_set.add(normed)
    # Also add corpus words that are long enough
    single_token_set |= {w for w in corpus_words if len(w) >= 4}

    # Per-row processing 
    print("\nProcessing rows …")
    clean_rows:       list = []
    removed_rows:     list = []
    non_english_rows: list = []
    seen_exact: set = set()

    for _, row in tqdm(df.iterrows(), total=total_input, desc="Processing"):
        original = row[keyword_col]
        primary  = row[primary_col]
        cluster  = row["_cluster"]

        def remove(reason: str):
            removed_rows.append({
                keyword_col:      original,
                primary_col:      primary,
                "_cluster":       cluster,
                "removal_reason": reason,
            })

        # Language filter 
        if not pd.isna(original) and is_any_non_ascii(str(original)):
            script = detect_script(str(original))
            if script == "Extended-Latin" and keep_extended_latin:
                pass   
            else:
                non_english_rows.append({
                    keyword_col: original,
                    primary_col: primary,
                    "_cluster":  cluster,
                    "script":    script,
                })
                remove(f"NON_ENGLISH_{script.upper()}")
                continue

        # Normalise
        normed = normalize_text(original)

        # Classify 
        remove_flag, reason = classify_keyword(
            normed, domain_vocab, brand_whitelist,
            corpus_words, single_token_set, min_freq
        )
        if remove_flag:
            remove(reason)
            continue

        # Exact dedup 
        if normed in seen_exact:
            remove("EXACT_DUPLICATE")
            continue
        seen_exact.add(normed)

        clean_rows.append({
            keyword_col: normed,
            primary_col: primary,
            "_cluster":  cluster,
        })

    # Cluster stats
    print("\nBuilding cluster stats …")
    clean_df   = pd.DataFrame(clean_rows)
    stats_rows = []
    if not clean_df.empty:
        for cluster_label, grp in clean_df.groupby("_cluster"):
            size = len(grp)
            stats_rows.append({
                "cluster_label": cluster_label,
                "keyword_count": size,
                "quality": (
                    "good"      if size >= 5 else
                    "thin"      if size >= 2 else
                    "singleton"
                ),
            })
    stats_df = pd.DataFrame(stats_rows).sort_values("keyword_count", ascending=False)

    # Save
    print("\nSaving outputs …")
    final_clean = (
        clean_df[[keyword_col, primary_col]]
        if not clean_df.empty
        else pd.DataFrame(columns=[keyword_col, primary_col])
    )
    final_clean.to_csv(output_clean,    index=False)
    pd.DataFrame(removed_rows).to_csv(output_removed,     index=False)
    pd.DataFrame(non_english_rows).to_csv(output_non_english, index=False)
    stats_df.to_csv(output_stats, index=False)

    # Summary 
    total_removed      = len(removed_rows)
    noise_pct          = total_removed / total_input * 100 if total_input else 0
    total_clusters_in  = df["_cluster"].nunique()
    total_clusters_out = clean_df["_cluster"].nunique() if not clean_df.empty else 0

    print("  CLEANING SUMMARY  (v5.0)")
    print(f"  Input rows              : {total_input:>10,}")
    print(f"  Clean rows              : {len(clean_rows):>10,}")
    print(f"  Removed rows            : {total_removed:>10,}  ({noise_pct:.1f}%)")
    print(f"  Non-English isolated    : {len(non_english_rows):>10,}")
    print(f"  Clusters in             : {total_clusters_in:>10,}")
    print(f"  Clusters out            : {total_clusters_out:>10,}")
    if removed_rows:
        rdf = pd.DataFrame(removed_rows)
        for reason, count in rdf["removal_reason"].value_counts().items():
            print(f"  {reason:<40}: {count:>6,}")
    if not stats_df.empty:
        good       = (stats_df["quality"] == "good").sum()
        thin       = (stats_df["quality"] == "thin").sum()
        singletons = (stats_df["quality"] == "singleton").sum()
        print(f"  Clusters ≥5 keywords    : {good:>10,}  (good)")
        print(f"  Clusters 2-4 keywords   : {thin:>10,}  (thin)")
        print(f"  Singleton clusters      : {singletons:>10,}  (review)")
    print(f"\n  Outputs saved to → {output_dir}/")
    print(f"    clean_data.csv          — feed this to your model")
    print(f"    removed_data.csv        — full audit trail with reasons")
    print(f"    non_english_data.csv    — all non-English keywords")
    print(f"    cluster_stats.csv       — per-cluster keyword counts\n")


# ENTRY POINT 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Category-agnostic keyword cleaner — v5.0",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input",       default=DEFAULT_INPUT,      help="Path to input CSV")
    parser.add_argument("--output",      default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--min-freq",    type=float, default=MIN_WORD_FREQ,
                        help="Min Zipf frequency to consider a word 'known English'")
    parser.add_argument("--keyword-col", default=KEYWORD_COLUMN,     help="Secondary keyword column name")
    parser.add_argument("--primary-col", default=PRIMARY_COLUMN,     help="Primary / cluster label column name")
    parser.add_argument("--keep-extended-latin", action="store_true",
                        help="Keep French/Spanish/German (accented) keywords instead of removing them")
    args = parser.parse_args()

    run(
        input_file          = args.input,
        output_dir          = args.output,
        keyword_col         = args.keyword_col,
        primary_col         = args.primary_col,
        min_freq            = args.min_freq,
        keep_extended_latin = args.keep_extended_latin,
    )

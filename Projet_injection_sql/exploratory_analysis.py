#!/usr/bin/env python3
from pathlib import Path
import sys
import argparse
import json
import math
import re
from collections import Counter, defaultdict
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import entropy

# --- CONFIG ---
DEFAULT_OUT = Path("data") / "processed" / "analysis"
plt.style.use("ggplot")

# -------------------------
# Utilities
# -------------------------
def find_cleaned_file():
    candidates = [
        Path.cwd() / "cleaned_data.csv",
        Path.cwd() / "data" / "processed" / "cleaned_data.csv",
        Path.cwd() / "data" / "cleaned_data.csv"
    ]
    for p in candidates:
        if p.exists():
            return p.resolve()
    return None

def ensure_outdir(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(exist_ok=True)
    (out_dir / "tables").mkdir(exist_ok=True)

def save_fig(fig, out_dir: Path, name: str, dpi=150):
    path = out_dir / "figures" / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    print(f"[INFO] Saved figure: {path}")

def dump_json(obj, out_dir: Path, name: str):
    path = out_dir / "tables" / f"{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved json: {path}")

def save_df(df: pd.DataFrame, out_dir: Path, name: str):
    path = out_dir / "tables" / f"{name}.csv"
    df.to_csv(path, index=False)
    print(f"[INFO] Saved table CSV: {path}")

# -------------------------
# Text utilities
# -------------------------
TOKEN_RE = re.compile(r"[A-Za-z_]+|[0-9]+|==|!=|<=|>=|<>|[\(\)\[\]{};,:.=<>*/+\-]|--|/\*|\*/|'|\"")

def simple_tokenize(s: str):
    if not isinstance(s, str):
        return []
    # keep punctuation tokens as separate
    return TOKEN_RE.findall(s)

def char_ngrams(s: str, n=3):
    s = re.sub(r"\s+", " ", s)
    s = f" {s} "
    return [s[i:i+n] for i in range(len(s)-n+1)]

def compute_shannon_entropy_of_tokens(tokens):
    if not tokens:
        return 0.0
    c = Counter(tokens)
    probs = np.array(list(c.values())) / sum(c.values())
    return float(entropy(probs, base=2))

# -------------------------
# Analysis functions
# -------------------------
def basic_stats(df: pd.DataFrame):
    total = len(df)
    label_counts = df["Label"].value_counts().to_dict()
    lengths = df["Sentence"].astype(str).map(len)
    avg_len = float(lengths.mean())
    median_len = int(lengths.median())
    std_len = float(lengths.std())
    long_pct = float((lengths > (median_len*3)).sum()) / total if total else 0.0
    return {
        "total_rows": int(total),
        "label_counts": label_counts,
        "avg_length": avg_len,
        "median_length": median_len,
        "std_length": std_len,
        "long_pct": long_pct
    }

def length_distributions(df: pd.DataFrame, out_dir: Path, save_plots: bool):
    lens = df["Sentence"].astype(str).map(len)
    fig, ax = plt.subplots(figsize=(8,4))
    sns.histplot(lens, bins=50, ax=ax, kde=True)
    ax.set_title("Distribution des longueurs (caractères)")
    ax.set_xlabel("Longueur (caractères)")
    ax.set_ylabel("Nombre")
    if save_plots: save_fig(fig, out_dir, "length_distribution")
    plt.show()

    # per-label
    fig, ax = plt.subplots(figsize=(8,4))
    sns.boxplot(x=df["Label"].astype(str), y=lens, ax=ax)
    ax.set_title("Longueur des requêtes par classe")
    ax.set_xlabel("Label")
    ax.set_ylabel("Longueur")
    if save_plots: save_fig(fig, out_dir, "length_by_label")
    plt.show()

def token_and_ngram_stats(df: pd.DataFrame, out_dir: Path, top_n=100, save_plots: bool=True):
    # tokens: use simple_tokenize
    all_tokens = []
    per_label_tokens = {0: [], 1: []}
    entropies = []
    for _, row in df.iterrows():
        s = str(row["Sentence"])
        tokens = simple_tokenize(s)
        all_tokens.extend(tokens)
        per_label_tokens[int(row["Label"])].extend(tokens)
        entropies.append(compute_shannon_entropy_of_tokens(tokens))

    # top tokens overall and per class
    tok_counts = Counter(all_tokens).most_common(top_n)
    tok_df = pd.DataFrame(tok_counts, columns=["token","count"])
    save_df(tok_df, out_dir, "top_tokens_overall")

    tok0_df = pd.DataFrame(Counter(per_label_tokens[0]).most_common(top_n), columns=["token","count"])
    tok1_df = pd.DataFrame(Counter(per_label_tokens[1]).most_common(top_n), columns=["token","count"])
    save_df(tok0_df, out_dir, "top_tokens_label0")
    save_df(tok1_df, out_dir, "top_tokens_label1")

    # entropy stats
    ent_arr = np.array(entropies)
    ent_report = {"mean_entropy": float(ent_arr.mean()), "median_entropy": float(np.median(ent_arr)), "std_entropy": float(ent_arr.std())}
    dump_json(ent_report, out_dir, "entropy_stats")

    # char ngrams with CountVectorizer (fast)
    cv_char3 = CountVectorizer(analyzer="char", ngram_range=(3,3), max_features=5000)
    cv_char4 = CountVectorizer(analyzer="char", ngram_range=(4,4), max_features=5000)
    corpus = df["Sentence"].astype(str).tolist()
    X3 = cv_char3.fit_transform(corpus)
    X4 = cv_char4.fit_transform(corpus)
    sums3 = np.asarray(X3.sum(axis=0)).ravel()
    sums4 = np.asarray(X4.sum(axis=0)).ravel()
    names3 = np.array(cv_char3.get_feature_names_out())
    names4 = np.array(cv_char4.get_feature_names_out())
    top3_idx = np.argsort(-sums3)[:top_n]
    top4_idx = np.argsort(-sums4)[:top_n]
    char3_df = pd.DataFrame({"ngram": names3[top3_idx], "count": sums3[top3_idx]})
    char4_df = pd.DataFrame({"ngram": names4[top4_idx], "count": sums4[top4_idx]})
    save_df(char3_df, out_dir, "top_char3_ngrams")
    save_df(char4_df, out_dir, "top_char4_ngrams")

    # visualize top char3 as bar
    fig, ax = plt.subplots(figsize=(10,5))
    sns.barplot(x="count", y="ngram", data=char3_df.head(30), ax=ax)
    ax.set_title("Top char-3 ngrams")
    if save_plots: save_fig(fig, out_dir, "top_char3_ngrams")
    plt.show()

    return {"tok_counts": tok_counts, "char3_top": char3_df.head(50).to_dict(orient="records")}

def tfidf_top_features(df: pd.DataFrame, out_dir: Path, analyzer="char", ngram_min=3, ngram_max=6, top_n=50, save_plots: bool=True):
    corpus = df["Sentence"].astype(str).tolist()
    tf = TfidfVectorizer(analyzer=analyzer, ngram_range=(ngram_min, ngram_max), max_features=20000)
    X = tf.fit_transform(corpus)
    # sum tfidf across docs to get important features
    scores = np.asarray(X.sum(axis=0)).ravel()
    names = np.array(tf.get_feature_names_out())
    idx = np.argsort(-scores)[:top_n]
    feat_df = pd.DataFrame({"feature": names[idx], "score": scores[idx]})
    save_df(feat_df, out_dir, "tfidf_top_features")
    # bar
    fig, ax = plt.subplots(figsize=(10,6))
    sns.barplot(x="score", y="feature", data=feat_df, ax=ax)
    ax.set_title(f"Top {top_n} TF-IDF features ({analyzer} {ngram_min}-{ngram_max})")
    if save_plots: save_fig(fig, out_dir, "tfidf_top_features")
    plt.show()
    return feat_df

def class_token_diffs(df: pd.DataFrame, out_dir: Path, top_n=50, save_plots: bool=True):
    # Compare TF-IDF of class 1 vs class 0 using CountVectorizer then ratio
    df0 = df[df["Label"] == 0]
    df1 = df[df["Label"] == 1]
    if len(df1) == 0 or len(df0) == 0:
        print("[WARN] Not enough samples in one of the classes for class_token_diffs.")
        return None

    cv = CountVectorizer(analyzer="char", ngram_range=(3,5), max_features=10000)
    X = cv.fit_transform(df["Sentence"].astype(str).tolist())
    names = np.array(cv.get_feature_names_out())
    sums0 = np.asarray(cv.transform(df0["Sentence"].astype(str).tolist()).sum(axis=0)).ravel()
    sums1 = np.asarray(cv.transform(df1["Sentence"].astype(str).tolist()).sum(axis=0)).ravel()
    # add 1 to avoid divide by zero
    ratio = (sums1 + 1) / (sums0 + 1)
    idx = np.argsort(-ratio)[:top_n]
    diff_df = pd.DataFrame({"feature": names[idx], "ratio": ratio[idx], "count_class1": sums1[idx], "count_class0": sums0[idx]})
    save_df(diff_df, out_dir, "feature_ratio_class1_over_0")
    # plot
    fig, ax = plt.subplots(figsize=(10,6))
    sns.barplot(x="ratio", y="feature", data=diff_df, ax=ax)
    ax.set_title("Features most over-represented in class 1 (SQLi) vs class 0")
    if save_plots: save_fig(fig, out_dir, "feature_ratio_class1_over_0")
    plt.show()
    return diff_df

def clustering_and_embedding(df: pd.DataFrame, out_dir: Path, analyzer="char", ngram_min=3, ngram_max=5, n_components=50, tsne_perplexity=30, save_plots: bool=True, sample_size=3000):
    # sample for speed
    if sample_size and sample_size < len(df):
        df_s = df.sample(sample_size, random_state=42).reset_index(drop=True)
    else:
        df_s = df.reset_index(drop=True)
    corpus = df_s["Sentence"].astype(str).tolist()

    tf = TfidfVectorizer(analyzer=analyzer, ngram_range=(ngram_min, ngram_max), max_features=5000)
    X = tf.fit_transform(corpus)

    # reduce with SVD
    svd = TruncatedSVD(n_components=min(n_components, X.shape[1]-1), random_state=42)
    X_red = svd.fit_transform(X)

    # kmeans clustering
    n_clusters = min(8, max(2, int(math.sqrt(len(df_s)/10))))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_ids = kmeans.fit_predict(X_red)

    # TSNE for 2D visualization (on reduced)
    tsne = TSNE(n_components=2, perplexity=min(tsne_perplexity, max(5, len(df_s)//4)), random_state=42, init="pca", learning_rate="auto")
    X_tsne = tsne.fit_transform(X_red)

    plot_df = pd.DataFrame({"x": X_tsne[:,0], "y": X_tsne[:,1], "cluster": cluster_ids, "label": df_s["Label"].astype(int)})
    fig, ax = plt.subplots(figsize=(10,7))
    sns.scatterplot(data=plot_df, x="x", y="y", hue="cluster", style="label", palette="tab10", ax=ax, s=40)
    ax.set_title(f"TSNE visualization (n={len(df_s)}, clusters={n_clusters})")
    if save_plots: save_fig(fig, out_dir, "tsne_clusters")
    plt.show()

    # save top terms per cluster
    centroids = kmeans.cluster_centers_
    # map centroids back to features via svd.components_ pseudo-inverse (approx)
    # Instead get top terms by average tf-idf in cluster
    cluster_terms = {}
    X_arr = X.toarray()
    for cid in range(n_clusters):
        idxs = np.where(cluster_ids == cid)[0]
        if len(idxs) == 0:
            cluster_terms[cid] = []
            continue
        avg = np.mean(X_arr[idxs, :], axis=0)
        top_idx = np.argsort(-avg)[:15]
        terms = np.array(tf.get_feature_names_out())[top_idx].tolist()
        cluster_terms[cid] = terms
    dump_json(cluster_terms, out_dir, "cluster_top_terms")
    return {"n_clusters": n_clusters, "cluster_terms": cluster_terms}

# -------------------------
# Main workflow
# -------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description="Advanced data analysis for SQLi dataset (cleaned_data.csv)")
    parser.add_argument("--save-plots", action="store_true", help="Save generated plots to data/processed/analysis/figures/")
    parser.add_argument("--outdir", type=str, default=str(DEFAULT_OUT), help="Output directory root")
    parser.add_argument("--top-n", type=int, default=100, help="Top N tokens/ngrams to keep")
    parser.add_argument("--sample", type=int, default=5000, help="Sample size for heavy ops (clustering/tsne). Use 0 for full dataset")
    args = parser.parse_args(argv)

    out_dir = Path(args.outdir)
    ensure_outdir(out_dir)

    path = find_cleaned_file()
    if path is None:
        print("ERROR: cleaned_data.csv not found in cwd or data/processed/. Place it and re-run.")
        sys.exit(1)

    print(f"[INFO] Loading {path}")
    df = pd.read_csv(path)
    if "Sentence" not in df.columns or "Label" not in df.columns:
        print("ERROR: cleaned_data.csv must contain Sentence and Label columns.")
        sys.exit(1)

    # Basic statistics
    report = {}
    report["basic"] = basic_stats(df)
    print("[INFO] Basic stats:", report["basic"])

    # Length distributions
    length_distributions(df, out_dir, save_plots=args.save_plots)

    # Token & ngram stats
    tok_stats = token_and_ngram_stats(df, out_dir, top_n=args.top_n, save_plots=args.save_plots)
    report["token_stats_summary"] = {"top_tokens_sample": tok_stats["tok_counts"][:20] if "tok_counts" in tok_stats else []}

    # TF-IDF top features
    tfidf_feats = tfidf_top_features(df, out_dir, analyzer="char", ngram_min=3, ngram_max=6, top_n=100, save_plots=args.save_plots)
    report["tfidf_top_count"] = len(tfidf_feats)

    # class token diffs
    diffs = class_token_diffs(df, out_dir, top_n=60, save_plots=args.save_plots)
    report["class_token_diff_example"] = (diffs.head(10).to_dict(orient="records") if diffs is not None else [])

    # clustering + embedding
    clust_res = clustering_and_embedding(df, out_dir, analyzer="char", ngram_min=3, ngram_max=5, n_components=50, tsne_perplexity=30, save_plots=args.save_plots, sample_size=(args.sample if args.sample>0 else None))
    report["clustering"] = clust_res

    # Save report
    report_path = out_dir / "analysis_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved analysis report: {report_path}")

    print("[DONE] Advanced analysis complete.")

if __name__ == "__main__":
    main()

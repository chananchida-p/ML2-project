# =============================================================================
# Problem Set #2
# Amazon Beauty Reviews: Topic Modelling and Brand Response Prioritisation
#
# Research question:
#   What topics in low-rated Amazon Beauty reviews signal product quality
#   issues, and how can brands use this to prioritise response?
#
# Pipeline overview:
#   01 – Data loading, cleaning, and EDA
#   02 – TF-IDF construction (vocabulary for LDA and coherence evaluation)
#   03 – LDA coherence search (select best K)
#   04 – LDA final fit with best K  +  BERTopic pipeline
#   05 – Emotion classification (j-hartmann/emotion-english-distilroberta-base)
#   06 – Brand response prioritisation matrix
# =============================================================================

import matplotlib
matplotlib.use("Agg")   # non-interactive backend for saving figures

import warnings
warnings.filterwarnings("ignore")

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel

from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import umap
import hdbscan


# =============================================================================
# GLOBAL SETTINGS
# =============================================================================

RANDOM_STATE     = 42       # ensures reproducibility across all stochastic steps
MAX_LOW_STAR_DOCS = 12000   # stratified sample size from low-star corpus
                             # 12,000 gives HDBSCAN enough density to separate
                             # sub-themes that collapse at n=2,000
TOP_N_WORDS      = 10       # top keywords per topic displayed in console output

# BERTopic topic count: "auto" means keep whatever HDBSCAN discovers.
# reduce_topics() is only called if the discovered count exceeds this limit.
BERTOPIC_NR_TOPICS = "auto"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def simple_tokenise(text):
    """Lowercase and strip non-alpha characters; used for Gensim coherence."""
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return text.split()


def classify_emotion_factory(emotion_classifier):
    """
    Returns a safe wrapper around the HuggingFace emotion pipeline.
    Handles both list-of-list and list-of-dict output formats.
    Falls back to 'neutral' on any exception (e.g. empty string).
    """
    def classify_emotion(text, max_chars=512):
        try:
            result = emotion_classifier(str(text)[:max_chars])
            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], list) and len(result[0]) > 0:
                    return result[0][0]["label"]
                if isinstance(result[0], dict):
                    return result[0]["label"]
            return "neutral"
        except Exception:
            return "neutral"
    return classify_emotion


def assign_tier(corpus_pct, severity):
    """
    Map (volume, emotional severity) to a priority tier.
    Thresholds:
      - Critical: corpus_pct >= 12% OR severity >= 2.5 (disgust or fear)
        Disgust-dominant topics (severity=2.5) reflect sensory rejection or
        safety concerns and warrant immediate escalation regardless of volume.
      - High:     corpus_pct >= 8% OR severity >= 2.0 (anger)
      - Medium:   corpus_pct >= 4% OR severity >= 1.5 (sadness)
      - Low:      everything else
    """
    if corpus_pct >= 12 or severity >= 2.5:
        return "Critical"
    elif corpus_pct >= 8 or severity >= 2.0:
        return "High"
    elif corpus_pct >= 4 or severity >= 1.5:
        return "Medium"
    return "Low"


def assign_team(dominant_emotion):
    """Route each dominant emotion to the most appropriate brand team."""
    mapping = {
        "fear":    "Product Safety / QA",
        "disgust": "Product Safety / QA",
        "anger":   "Operations / CX",
        "sadness": "Product Development",
    }
    return mapping.get(dominant_emotion, "Customer Experience")


def assign_sla(tier):
    """Return a response SLA string for each priority tier."""
    return {"Critical": "24h", "High": "48h", "Medium": "1 week", "Low": "Monthly"}[tier]


def shorten_topic_name(name, max_words=3):
    """Trim BERTopic auto-generated topic names for chart annotations."""
    return " ".join(str(name).replace("_", " ").split()[:max_words])


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():

    # =========================================================================
    # 01 – DATA LOADING AND FILTERING
    # =========================================================================
    print("=" * 70)
    print("01 – DATA LOADING AND FILTERING")
    print("=" * 70)

    # Load from Hugging Face; dataset is ~700k Amazon Beauty reviews
    ds = load_dataset("jhan21/amazon-beauty-reviews-dataset")
    df = pd.DataFrame(ds["train"])

    print(f"Full dataset: {len(df):,} reviews")
    print(df.head())
    print("\nColumns:", df.columns.tolist())

    # ------------------------------------------------------------------
    # Basic cleaning: drop nulls, empty strings, coerce rating to int
    # ------------------------------------------------------------------
    df = df.dropna(subset=["text", "rating"]).copy()
    df["text"]   = df["text"].astype(str)
    df           = df[df["text"].str.strip() != ""].copy()
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df           = df.dropna(subset=["rating"]).copy()
    df["rating"] = df["rating"].astype(int)

    # ------------------------------------------------------------------
    # EDA figure A: full rating distribution
    # Shows the positivity bias (Hu et al., 2009) motivating restriction
    # to 1-2 star reviews.
    # ------------------------------------------------------------------
    df["rating"].value_counts().sort_index().plot(
        kind="bar", color="steelblue", edgecolor="black"
    )
    plt.title("Rating Distribution – Full Amazon Beauty Corpus")
    plt.xlabel("Star Rating")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("figure_rating_distribution_full.png", dpi=150)
    plt.close()

    # ------------------------------------------------------------------
    # Restrict to 1-2 star reviews (complaint-heavy corpus)
    # ------------------------------------------------------------------
    df_low = df[df["rating"].isin([1, 2])].copy().reset_index(drop=True)

    print(f"\nLow-star corpus before sampling: {len(df_low):,} reviews")
    print("Rating distribution:")
    print(df_low["rating"].value_counts().sort_index())

    # ------------------------------------------------------------------
    # Stratified sampling: maintain the natural 1-star / 2-star ratio.
    # Using 12,000 reviews instead of the original 2,000 so that HDBSCAN
    # has sufficient embedding density to separate complaint sub-themes.
    # ------------------------------------------------------------------
    if len(df_low) > MAX_LOW_STAR_DOCS:
        df_low = (
            df_low.groupby("rating", group_keys=False)
            .apply(lambda x: x.sample(
                n=int(MAX_LOW_STAR_DOCS * len(x) / len(df_low)),
                random_state=RANDOM_STATE
            ))
            .reset_index(drop=True)
        )
        # Top up by a few rows if integer rounding left us short
        if len(df_low) < MAX_LOW_STAR_DOCS:
            shortfall = MAX_LOW_STAR_DOCS - len(df_low)
            extra = (
                df[df["rating"].isin([1, 2])]
                .drop(df_low.index, errors="ignore")
                .sample(n=shortfall, random_state=RANDOM_STATE)
            )
            df_low = pd.concat([df_low, extra]).reset_index(drop=True)
        print(f"\nStratified sample: {len(df_low):,} reviews")
        print(df_low["rating"].value_counts().sort_index())
    else:
        print(f"\nUsing full low-star corpus: {len(df_low):,} reviews")

    # ------------------------------------------------------------------
    # EDA figure B: review length distribution
    # Shows the short-text nature of the corpus (median ~21 words),
    # which limits lexical diversity and constrains topic separability.
    # ------------------------------------------------------------------
    df_low["review_length"] = df_low["text"].apply(lambda x: len(str(x).split()))

    plt.figure(figsize=(10, 4))
    plt.hist(df_low["review_length"], bins=50, color="steelblue", edgecolor="black")
    plt.axvline(50,  color="red",    linestyle="--", label="50 words")
    plt.axvline(100, color="orange", linestyle="--", label="100 words")
    plt.axvline(200, color="green",  linestyle="--", label="200 words")
    plt.xlim(0, 300)
    plt.title("Review Length Distribution – Low-Star Corpus")
    plt.xlabel("Number of Words")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figure_review_length_low_star.png", dpi=150)
    plt.close()

    print("\nReview length summary:")
    print(df_low["review_length"].describe())

    docs = df_low["text"].tolist()
    print(f"\nTotal documents for analysis: {len(docs):,}")

    # =========================================================================
    # 02 – VOCABULARY CONSTRUCTION (TF-IDF + COUNT)
    # =========================================================================
    print("\n" + "=" * 70)
    print("02 – VOCABULARY CONSTRUCTION")
    print("=" * 70)

    # TF-IDF: used for LDA preprocessing (highlights informative terms)
    tfidf_vectorizer = TfidfVectorizer(
        stop_words="english",
        min_df=5,           # ignore very rare terms
        max_df=0.95,        # ignore near-universal terms
        ngram_range=(1, 2)  # unigrams and bigrams
    )
    tfidf_matrix = tfidf_vectorizer.fit_transform(docs)
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"Vocabulary size: {len(tfidf_vectorizer.vocabulary_):,}")

    # Count vectorizer: required by sklearn LDA (expects raw counts, not tf-idf weights)
    count_vectorizer = CountVectorizer(
        stop_words="english",
        min_df=5,
        max_df=0.95,
        ngram_range=(1, 2)
    )
    count_matrix = count_vectorizer.fit_transform(docs)
    feature_names = count_vectorizer.get_feature_names_out()

    # Gensim tokenised docs for C_v coherence evaluation
    tokenised_docs = [simple_tokenise(d) for d in docs]
    dictionary     = Dictionary(tokenised_docs)
    dictionary.filter_extremes(no_below=5, no_above=0.95)

    # =========================================================================
    # 03 – LDA COHERENCE SEARCH (select best K)
    # =========================================================================
    print("\n" + "=" * 70)
    print("03 – COHERENCE EVALUATION AND K SELECTION")
    print("=" * 70)

    k_values           = [5, 10, 15, 20, 25, 30]
    coherence_scores   = []

    print("Searching K in {5, 10, 15, 20, 25, 30} for best C_v coherence...")
    for k in k_values:
        lda_k = LatentDirichletAllocation(
            n_components=k,
            max_iter=20,
            learning_method="online",
            random_state=RANDOM_STATE,
            n_jobs=1
        )
        lda_k.fit(count_matrix)

        topic_words = [
            [feature_names[i] for i in topic.argsort()[::-1][:10]]
            for topic in lda_k.components_
        ]
        cm = CoherenceModel(
            topics=topic_words,
            texts=tokenised_docs,
            dictionary=dictionary,
            coherence="c_v",
            window_size=50
        )
        score = cm.get_coherence()
        coherence_scores.append(score)
        print(f"  K={k:2d} -> C_v = {score:.4f}")

    # Save coherence curve (Figure C in report)
    plt.figure(figsize=(8, 4))
    plt.plot(k_values, coherence_scores, marker="o")
    plt.title("LDA: Coherence Score vs Number of Topics (K)")
    plt.xlabel("K (number of topics)")
    plt.ylabel("C_v Coherence Score")
    plt.xticks(k_values)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figure_lda_coherence.png", dpi=150)
    plt.close()

    best_k = k_values[int(np.argmax(coherence_scores))]
    print(f"\nBest K for LDA: {best_k} (C_v = {max(coherence_scores):.4f})")

    # =========================================================================
    # 04 – LDA FINAL FIT + BERTOPIC PIPELINE
    # =========================================================================
    print("\n" + "=" * 70)
    print("04 – LDA FINAL FIT")
    print("=" * 70)

    # Fit LDA with the best K found above (not a hardcoded value)
    lda_model = LatentDirichletAllocation(
        n_components=best_k,
        max_iter=25,
        learning_method="online",
        random_state=RANDOM_STATE,
        n_jobs=1
    )
    lda_model.fit(count_matrix)
    print(f"\nLDA fitted with best K = {best_k}")

    print("\nLDA topic keywords:")
    print("-" * 70)
    for idx, topic in enumerate(lda_model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[::-1][:TOP_N_WORDS]]
        print(f"  Topic {idx:2d}: {', '.join(top_words)}")

    # These LDA keywords are used later to sub-characterise sub-themes
    # within the dominant BERTopic cluster (cross-model validation).

    print("\n" + "=" * 70)
    print("04 – BERTOPIC PIPELINE")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Sentence embeddings via all-MiniLM-L6-v2 (Wang et al., 2020)
    # 384-dimensional semantic vectors; 512-token input limit.
    # ------------------------------------------------------------------
    print("Encoding documents with all-MiniLM-L6-v2...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedding_model.encode(
        docs, show_progress_bar=True, batch_size=32
    )
    print(f"Embeddings shape: {embeddings.shape}")

    # ------------------------------------------------------------------
    # UMAP dimensionality reduction (McInnes et al., 2018)
    # Reduces 384-dim embeddings to 5-dim before clustering.
    # cosine metric is recommended for sentence-transformer embeddings.
    # ------------------------------------------------------------------
    umap_model = umap.UMAP(
        n_neighbors=15,     # local vs global structure trade-off
        n_components=5,     # target dimensionality for HDBSCAN
        min_dist=0.0,       # tighter packing aids cluster formation
        metric="cosine",
        random_state=RANDOM_STATE,
        low_memory=True
    )

    # ------------------------------------------------------------------
    # HDBSCAN clustering (McInnes et al., 2017)
    # min_cluster_size=15 (reduced from 30) allows finer-grained topics
    # to form; at n=12,000 there is sufficient density to support this.
    # ------------------------------------------------------------------
    hdbscan_model = hdbscan.HDBSCAN(
        min_cluster_size=15,
        min_samples=5,
        cluster_selection_method="eom",
        prediction_data=True    # required for approximate_distribution later
    )

    # CountVectorizer for c-TF-IDF keyword extraction within BERTopic
    vectorizer_model = CountVectorizer(
        stop_words="english",
        min_df=2,
        ngram_range=(1, 2)
    )

    # KeyBERTInspired improves keyword coherence by re-ranking via embedding similarity
    representation_model = KeyBERTInspired()

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        verbose=True
    )

    topics, probabilities = topic_model.fit_transform(docs, embeddings=embeddings)

    topic_info      = topic_model.get_topic_info()
    n_initial       = len(topic_info[topic_info["Topic"] != -1])
    n_outliers      = (np.array(topics) == -1).sum()

    print(f"\nInitial topics found:   {n_initial}")
    print(f"Outlier documents:      {n_outliers} ({100 * n_outliers / len(docs):.1f}%)")
    print("\nInitial topic summary:")
    print(topic_info.head(20))

    # Save interactive heatmap (topic similarity matrix)
    try:
        topic_model.visualize_heatmap().write_html("bertopic_heatmap.html")
        print("Saved bertopic_heatmap.html")
    except Exception as e:
        print(f"Could not save heatmap: {e}")

    # ------------------------------------------------------------------
    # Conditional topic reduction
    # Only reduce if BERTopic found more topics than BERTOPIC_NR_TOPICS.
    # When BERTOPIC_NR_TOPICS="auto", all discovered topics are kept.
    # ------------------------------------------------------------------
    if BERTOPIC_NR_TOPICS != "auto" and n_initial > BERTOPIC_NR_TOPICS:
        topic_model = topic_model.reduce_topics(docs, nr_topics=BERTOPIC_NR_TOPICS)
        topic_info  = topic_model.get_topic_info()
        print(f"\nReduced to {len(topic_info[topic_info['Topic'] != -1])} topics")
    else:
        print(f"\nKeeping all {n_initial} discovered topics")

    topic_info        = topic_model.get_topic_info()
    topic_assignments = np.array(topic_model.topics_)
    n_topics_final    = len(topic_info[topic_info["Topic"] != -1])
    print(f"Final topic count: {n_topics_final}")

    # ------------------------------------------------------------------
    # BERTopic evaluation metrics
    # ------------------------------------------------------------------
    # Filter keywords to only unigrams present in the Gensim dictionary.
    # BERTopic may produce bigrams (e.g. "hair loss") or OOV terms that
    # CoherenceModel cannot handle — these are dropped before evaluation.
    valid_tokens = set(dictionary.token2id.keys())

    raw_topic_words = [
        [w for w, _ in topic_model.get_topic(tid)[:10]]
        for tid in topic_info["Topic"]
        if tid != -1 and topic_model.get_topic(tid)
    ]
    bertopic_topic_words = [
        [w for w in words if w in valid_tokens and " " not in w]
        for words in raw_topic_words
    ]
    # Keep only topics with at least 2 valid tokens
    bertopic_topic_words = [t for t in bertopic_topic_words if len(t) >= 2]

    if bertopic_topic_words:
        cm_bert        = CoherenceModel(
            topics=bertopic_topic_words,
            texts=tokenised_docs,
            dictionary=dictionary,
            coherence="c_v",
            window_size=50
        )
        coherence_bert = cm_bert.get_coherence()
    else:
        print("Warning: no valid topic words for coherence — defaulting to 0.0")
        coherence_bert = 0.0

    all_words      = [w for t in bertopic_topic_words for w in t]
    diversity_bert = len(set(all_words)) / len(all_words) if all_words else 0
    n_assigned     = (topic_assignments != -1).sum()
    coverage_bert  = n_assigned / len(docs)

    print(f"\nBERTopic evaluation:")
    print(f"  C_v Coherence:     {coherence_bert:.3f}")
    print(f"  Topic Diversity:   {diversity_bert:.2f}")
    print(f"  Document Coverage: {coverage_bert * 100:.1f}%")

    # Model comparison summary (Table 1 in report)
    print("\n" + "=" * 60)
    print("Model Comparison Summary")
    print("=" * 60)
    print(f"{'Metric':<30} {'LDA+TF-IDF':>12} {'BERTopic':>12}")
    print("-" * 60)
    print(f"{'C_v Coherence':<30} {max(coherence_scores):>12.3f} {coherence_bert:>12.3f}")
    print(f"{'Topic Diversity':<30} {'N/A':>12} {diversity_bert:>12.2f}")
    print(f"{'Document Coverage (%)':<30} {'N/A':>12} {coverage_bert * 100:>11.1f}%")
    print(f"{'K pre-specified?':<30} {'Yes':>12} {'No':>12}")
    print(f"{'Handles polysemy?':<30} {'No':>12} {'Yes':>12}")

    # ------------------------------------------------------------------
    # Validation 1: Manual sample (5 reviews per topic)
    # Used to confirm that auto-generated topic labels are interpretable.
    # ------------------------------------------------------------------
    print("\n--- VALIDATION: Manual sample (5 reviews per topic) ---")
    document_info = topic_model.get_document_info(docs)
    sampled = (
        document_info
        .groupby("Topic", group_keys=False)
        .apply(lambda x: x.sample(n=min(5, len(x)), random_state=RANDOM_STATE))
        .reset_index(drop=True)
    )
    cols = [c for c in ["Document", "Topic", "Name", "Representation"] if c in sampled.columns]
    sampled[cols].to_csv("validation_samples.csv", index=False)
    print("Saved validation_samples.csv")

    # ------------------------------------------------------------------
    # Validation 2: Robustness across 3 random seeds
    # If the same topic count emerges across seeds, it is a stable
    # property of the corpus rather than an artefact of initialisation.
    # ------------------------------------------------------------------
    print("\n--- VALIDATION: Robustness across 3 random seeds ---")
    for seed in [42, 123, 999]:
        umap_s   = umap.UMAP(
            n_neighbors=15, n_components=5, min_dist=0.0,
            metric="cosine", random_state=seed, low_memory=True
        )
        hdbscan_s = hdbscan.HDBSCAN(
            min_cluster_size=15, min_samples=5,
            cluster_selection_method="eom", prediction_data=True
        )
        tm_s = BERTopic(
            embedding_model=embedding_model, umap_model=umap_s,
            hdbscan_model=hdbscan_s, vectorizer_model=vectorizer_model,
            representation_model=representation_model, verbose=False
        )
        tm_s.fit_transform(docs, embeddings=embeddings)
        n_s = len(tm_s.get_topic_info().query("Topic != -1"))
        print(f"  Seed {seed}: {n_s} topics")

    # Topics to follow up (up to 12 largest by document count)
    topic_info_valid = topic_info[topic_info["Topic"] != -1].copy()
    top_topics = (
        topic_info_valid
        .sort_values("Count", ascending=False)["Topic"]
        .head(12).tolist()
    )
    print(f"\nTop topics for follow-up: {top_topics}")

    # ------------------------------------------------------------------
    # Figure 1: UMAP 2D embedding coloured by topic
    # Visual validation that topic clusters are spatially distinct.
    # ------------------------------------------------------------------
    print("\n--- FIGURE 1: UMAP 2D embedding ---")
    umap_2d = umap.UMAP(
        n_neighbors=15, n_components=2, min_dist=0.0,
        metric="cosine", random_state=RANDOM_STATE, low_memory=True
    )
    emb_2d = umap_2d.fit_transform(embeddings)

    plt.figure(figsize=(12, 8))
    for tid in top_topics:
        mask = topic_assignments == tid
        if mask.sum() > 0:
            plt.scatter(emb_2d[mask, 0], emb_2d[mask, 1], s=5, alpha=0.5, label=f"T-{tid:02d}")
    other_mask = ~np.isin(topic_assignments, top_topics)
    plt.scatter(emb_2d[other_mask, 0], emb_2d[other_mask, 1],
                s=2, alpha=0.1, color="lightgrey", label="Other/Outlier")
    plt.legend(markerscale=3, fontsize=9, loc="upper right")
    plt.title("Figure 1: UMAP 2D – BERTopic Clusters")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.tight_layout()
    plt.savefig("figure1_umap_topics.png", dpi=150)
    plt.close()
    print("Saved figure1_umap_topics.png")

    # Save interactive BERTopic visualisations
    for fname, fn in [
        ("bertopic_topics_map.html",  topic_model.visualize_topics),
        ("bertopic_hierarchy.html",   topic_model.visualize_hierarchy),
        ("bertopic_barchart.html",    lambda: topic_model.visualize_barchart(
            top_n_topics=min(12, len(top_topics))
        )),
    ]:
        try:
            fn().write_html(fname)
            print(f"Saved {fname}")
        except Exception as e:
            print(f"Could not save {fname}: {e}")

    # =========================================================================
    # 05 – EMOTION CLASSIFICATION
    # =========================================================================
    print("\n" + "=" * 70)
    print("05 – EMOTION CLASSIFICATION")
    print("=" * 70)

    # j-hartmann/emotion-english-distilroberta-base (Hartmann et al., 2023)
    # Chosen for its cross-domain benchmark accuracy (Hartmann et al., 2023).
    # device=-1 forces CPU inference; change to device=0 for GPU.
    print("Loading emotion classifier...")
    emotion_clf = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=1,
        device=-1
    )
    classify_emotion = classify_emotion_factory(emotion_clf)

    print("Classifying emotions (may take several minutes on CPU)...")
    df_low["emotion"] = df_low["text"].apply(classify_emotion)
    df_low["topic"]   = topic_model.topics_

    print("\nOverall emotion distribution:")
    print(df_low["emotion"].value_counts())

    # Dominant emotion per topic
    print("\nDominant emotion per top topic:")
    print("-" * 60)
    for tid in top_topics:
        sub = df_low[df_low["topic"] == tid]
        if len(sub) == 0:
            continue
        dominant = sub["emotion"].value_counts().idxmax()
        dist     = sub["emotion"].value_counts(normalize=True).round(3).to_dict()
        print(f"  T-{tid:02d}: dominant = {dominant:<10} | {dist}")

    # ------------------------------------------------------------------
    # Figure 2: Row-normalised emotion distribution per topic (stacked bar)
    # ------------------------------------------------------------------
    print("\n--- FIGURE 2: Emotion distribution per topic ---")
    quality_df = df_low[df_low["topic"].isin(top_topics)].copy()

    emotion_order  = ["fear", "disgust", "anger", "sadness", "surprise", "neutral", "joy"]
    emotion_colors = {
        "fear": "#d62728", "disgust": "#e377c2", "anger": "#ff7f0e",
        "sadness": "#1f77b4", "surprise": "#bcbd22", "neutral": "#7f7f7f", "joy": "#2ca02c"
    }

    crosstab = pd.crosstab(quality_df["topic"], quality_df["emotion"], normalize="index")
    crosstab = crosstab.reindex(
        columns=[e for e in emotion_order if e in crosstab.columns], fill_value=0
    )

    # Replace numeric topic IDs with readable labels for the chart
    label_map = {
        tid: f"T-{tid:02d} {topic_info_valid[topic_info_valid['Topic'] == tid].iloc[0]['Name']}"
        for tid in top_topics
        if len(topic_info_valid[topic_info_valid["Topic"] == tid]) > 0
    }
    crosstab.index = [label_map.get(i, f"T-{i}") for i in crosstab.index]

    crosstab.plot(
        kind="bar", stacked=True, figsize=(14, 6),
        color=[emotion_colors.get(e, "grey") for e in crosstab.columns]
    )
    plt.title("Figure 2: Emotion Distribution per Top BERTopic Topic")
    plt.xlabel("Topic")
    plt.ylabel("Proportion of Reviews")
    plt.xticks(rotation=30, ha="right")
    plt.legend(title="Emotion", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig("figure2_emotion_distribution.png", dpi=150)
    plt.close()
    print("Saved figure2_emotion_distribution.png")

    print("\nEmotion distribution per topic (row-normalised):")
    print(crosstab.round(3))

    # ------------------------------------------------------------------
    # Star-rating emotion stratification
    # Splitting by 1-star vs 2-star reveals different emotional profiles:
    # 1-star = stronger disgust (product rejection)
    # 2-star = stronger sadness (unmet expectation)
    # This gradient supports within-topic routing without extra modelling.
    # ------------------------------------------------------------------
    print("\n--- Emotion by star rating ---")
    for r in [1, 2]:
        sub  = df_low[df_low["rating"] == r]
        dist = sub["emotion"].value_counts(normalize=True).round(3).to_dict()
        print(f"  {r}-star (n={len(sub):,}): {dist}")

    print("\nPer-topic emotion by rating:")
    for tid in top_topics:
        for r in [1, 2]:
            sub = df_low[(df_low["topic"] == tid) & (df_low["rating"] == r)]
            if len(sub) > 5:
                dom = sub["emotion"].value_counts().idxmax()
                pct = sub["emotion"].value_counts(normalize=True).iloc[0]
                print(f"  T-{tid:02d} | {r}-star (n={len(sub)}): {dom} ({pct:.1%})")

    # =========================================================================
    # 06 – PRIORITISATION MATRIX
    # =========================================================================
    print("\n" + "=" * 70)
    print("06 – PRIORITISATION MATRIX")
    print("=" * 70)

    # Numeric severity score per emotion class
    # fear > disgust > anger > sadness > surprise > neutral > joy
    emotion_severity = {
        "fear": 3.0, "disgust": 2.5, "anger": 2.0,
        "sadness": 1.5, "surprise": 1.0, "neutral": 0.5, "joy": 0.2
    }

    topic_counts = df_low["topic"].value_counts(normalize=True) * 100
    priority_rows = []

    for tid in top_topics:
        sub = df_low[df_low["topic"] == tid]
        if len(sub) == 0:
            continue

        dominant    = sub["emotion"].value_counts().idxmax()
        severity    = emotion_severity.get(dominant, 1.0)
        corpus_pct  = float(topic_counts.get(tid, 0))
        topic_row   = topic_info_valid[topic_info_valid["Topic"] == tid]
        topic_name  = topic_row.iloc[0]["Name"] if len(topic_row) > 0 else f"Topic {tid}"

        tier   = assign_tier(corpus_pct, severity)
        team   = assign_team(dominant)
        sla    = assign_sla(tier)
        action = f"Review root cause for {dominant}-driven complaints"

        priority_rows.append({
            "Topic": tid, "Theme": topic_name,
            "CorpusSharePct": round(corpus_pct, 1),
            "DominantEmotion": dominant, "Severity": severity,
            "Tier": tier, "Team": team, "SLA": sla, "Action": action
        })

    priority_df = pd.DataFrame(priority_rows)

    if not priority_df.empty:
        tier_order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
        priority_df["TierOrder"] = priority_df["Tier"].map(tier_order)
        priority_df = priority_df.sort_values(
            ["TierOrder", "CorpusSharePct"], ascending=[True, False]
        ).reset_index(drop=True)

        print("\nBrand Response Prioritisation Matrix")
        print("=" * 120)
        print(priority_df[
            ["Topic", "Theme", "CorpusSharePct", "DominantEmotion", "Tier", "Team", "SLA", "Action"]
        ].to_string(index=False))

        # Figure 3: scatter plot of corpus share vs emotion severity
        tier_colors = {"Critical": "red", "High": "orange", "Medium": "gold", "Low": "green"}
        fig, ax = plt.subplots(figsize=(10, 6))

        for _, row in priority_df.iterrows():
            ax.scatter(row["CorpusSharePct"], row["Severity"],
                       s=300, color=tier_colors[row["Tier"]], zorder=3, edgecolors="black")
            ax.annotate(
                f"T-{row['Topic']}\n{shorten_topic_name(row['Theme'])}",
                (row["CorpusSharePct"], row["Severity"]),
                textcoords="offset points", xytext=(6, 4), fontsize=8
            )

        patches = [mpatches.Patch(color=c, label=t) for t, c in tier_colors.items()]
        ax.legend(handles=patches, title="Priority Tier", loc="upper right")
        ax.set_xlabel("Corpus Share (%)")
        ax.set_ylabel("Emotion Severity")
        ax.set_title("Brand Response Prioritisation: Volume vs Emotion Severity")
        ax.set_yticks([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        ax.set_yticklabels(["Neutral", "Surprise", "Sadness", "Anger", "Disgust", "Fear"])
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("figure3_prioritisation_matrix.png", dpi=150)
        plt.close()
        print("Saved figure3_prioritisation_matrix.png")

        priority_df.to_csv("prioritisation_matrix.csv", index=False)
    else:
        print("No rows generated for prioritisation matrix.")
        pd.DataFrame().to_csv("prioritisation_matrix.csv", index=False)

    # =========================================================================
    # SAVE ALL OUTPUTS
    # =========================================================================
    topic_info.to_csv("bertopic_topic_info.csv", index=False)
    df_low.to_csv("low_star_reviews_with_topics_emotions.csv", index=False)

    print("\nAnalysis complete. Outputs saved:")
    for f in [
        "figure_rating_distribution_full.png",
        "figure_review_length_low_star.png",
        "figure_lda_coherence.png",
        "figure1_umap_topics.png",
        "figure2_emotion_distribution.png",
        "figure3_prioritisation_matrix.png",
        "bertopic_topic_info.csv",
        "validation_samples.csv",
        "low_star_reviews_with_topics_emotions.csv",
        "prioritisation_matrix.csv",
        "bertopic_heatmap.html",
        "bertopic_topics_map.html",
        "bertopic_hierarchy.html",
        "bertopic_barchart.html",
    ]:
        print(f"  - {f}")


if __name__ == "__main__":
    main()

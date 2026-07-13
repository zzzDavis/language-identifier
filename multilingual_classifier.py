import os
import re
import random
import unicodedata

import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from scipy.sparse import hstack, csr_matrix


# =========================
# CONFIG
# =========================
SEED = 42
TEST_RATIO = 0.20

ENGLISH_FILE = "english.txt"
GERMAN_FILE = "german.txt"
SPANISH_FILE = "spanish.txt"

LABEL_EN = 0
LABEL_DE = 1
LABEL_ES = 2

# Keep letters for English + German + Spanish (and common diacritics).
_ALLOWED_LETTERS_RE = re.compile(r"[^a-zA-ZäöüßÄÖÜñáéíóúüÁÉÍÓÚÜ]", re.UNICODE)


# =========================
# NORMALIZATION / LOADING
# =========================
def normalize_word(w: str) -> str:
    w = w.strip().lower()
    w = unicodedata.normalize("NFC", w)
    w = _ALLOWED_LETTERS_RE.sub("", w)
    return w


def load_words_from_file(path: str) -> list[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read().split()

    words = []
    seen = set()
    for token in raw:
        w = normalize_word(token)
        if len(w) == 5 and w not in seen:
            seen.add(w)
            words.append(w)
    return words


def remove_overlaps(en: list[str], de: list[str], es: list[str]):
    """
    Remove any word that appears in more than one language list.
    Shared words make the classification *impossible* for 5-letter words.
    """
    en_set, de_set, es_set = set(en), set(de), set(es)
    overlap = (en_set & de_set) | (en_set & es_set) | (de_set & es_set)

    if overlap:
        en = [w for w in en if w not in overlap]
        de = [w for w in de if w not in overlap]
        es = [w for w in es if w not in overlap]

    return en, de, es, len(overlap)


# =========================
# MANUAL PER-LANGUAGE SPLIT (80/20 each)
# =========================
def manual_split_per_language(en: list[str], de: list[str], es: list[str],
                              test_ratio: float, seed: int):
    rng = random.Random(seed)

    def split_one(words):
        words = words[:]  # copy
        rng.shuffle(words)
        test_size = max(1, int(len(words) * test_ratio))
        test = words[:test_size]
        train = words[test_size:]
        return train, test

    en_tr, en_te = split_one(en)
    de_tr, de_te = split_one(de)
    es_tr, es_te = split_one(es)

    X_train = en_tr + de_tr + es_tr
    y_train = ([LABEL_EN] * len(en_tr)) + ([LABEL_DE] * len(de_tr)) + ([LABEL_ES] * len(es_tr))

    X_test = en_te + de_te + es_te
    y_test = ([LABEL_EN] * len(en_te)) + ([LABEL_DE] * len(de_te)) + ([LABEL_ES] * len(es_te))

    # shuffle combined sets manually
    train = list(zip(X_train, y_train))
    test = list(zip(X_test, y_test))
    rng.shuffle(train)
    rng.shuffle(test)

    X_train, y_train = map(list, zip(*train))
    X_test, y_test = map(list, zip(*test))
    return X_train, X_test, y_train, y_test


# =========================
# FEATURE ENGINEERING (this is the accuracy booster)
# =========================
def positional_tokens(word: str) -> str:
    """
    Convert "canto" -> "p0=c p1=a p2=n p3=t p4=o"
    Position matters a lot for short words.
    """
    return " ".join([f"p{i}={ch}" for i, ch in enumerate(word)])


def build_feature_matrices(X_train_words: list[str], X_test_words: list[str]):
    """
    Combine:
      1) char n-gram TF-IDF (strong general signal)
      2) positional char tokens TF-IDF (helps separate similar languages)
    """
    # (1) char n-grams
    vec_char = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        min_df=2,
        sublinear_tf=True
    )

    # (2) positional tokens (treated as "words")
    vec_pos = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"(?u)\bp\d=\S+\b",  # tokens like p0=a
        lowercase=False
    )

    Xtr_char = vec_char.fit_transform(X_train_words)
    Xte_char = vec_char.transform(X_test_words)

    Xtr_pos = vec_pos.fit_transform([positional_tokens(w) for w in X_train_words])
    Xte_pos = vec_pos.transform([positional_tokens(w) for w in X_test_words])

    # Combine sparse matrices
    Xtr = hstack([Xtr_char, Xtr_pos]).tocsr()
    Xte = hstack([Xte_char, Xte_pos]).tocsr()

    return Xtr, Xte


# =========================
# TRAIN + EVAL
# =========================
def train_eval_models(X_train_words, X_test_words, y_train, y_test):
    Xtr, Xte = build_feature_matrices(X_train_words, X_test_words)

    results = {}

    # --- SVM (usually strong baseline) ---
    svm = LinearSVC(C=3.0)
    svm.fit(Xtr, y_train)
    pred_svm = svm.predict(Xte)
    results["SVM"] = (
        accuracy_score(y_test, pred_svm),
        confusion_matrix(y_test, pred_svm, labels=[0, 1, 2])
    )

    # --- KNN (needs dimensionality reduction to behave well) ---
    # Reduce to a manageable dense space then KNN with Euclidean.
    svd = TruncatedSVD(n_components=300, random_state=SEED)
    l2 = Normalizer(copy=False)
    reducer = make_pipeline(svd, l2)

    Xtr_red = reducer.fit_transform(Xtr)
    Xte_red = reducer.transform(Xte)

    knn = KNeighborsClassifier(
        n_neighbors=7,
        weights="distance",
        metric="euclidean"
    )
    knn.fit(Xtr_red, y_train)
    pred_knn = knn.predict(Xte_red)
    results["KNN"] = (
        accuracy_score(y_test, pred_knn),
        confusion_matrix(y_test, pred_knn, labels=[0, 1, 2])
    )

    # --- MLP (dense features work best; reuse reduced space) ---
    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        max_iter=350,
        early_stopping=True,
        n_iter_no_change=12,
        random_state=SEED
    )
    mlp.fit(Xtr_red, y_train)
    pred_mlp = mlp.predict(Xte_red)
    results["MLP"] = (
        accuracy_score(y_test, pred_mlp),
        confusion_matrix(y_test, pred_mlp, labels=[0, 1, 2])
    )

    return results


# =========================
# MAIN
# =========================
def main():
    random.seed(SEED)
    np.random.seed(SEED)

    # Load lists
    en = load_words_from_file(ENGLISH_FILE)
    de = load_words_from_file(GERMAN_FILE)
    es = load_words_from_file(SPANISH_FILE)

    print(f"[INFO] English 5-letter words: {len(en)}")
    print(f"[INFO] German  5-letter words: {len(de)}")
    print(f"[INFO] Spanish 5-letter words: {len(es)}")

    # Remove overlaps (BIG accuracy gain)
    en, de, es, overlap_count = remove_overlaps(en, de, es)
    print(f"[INFO] Removed {overlap_count} overlapping 5-letter words shared across languages.")

    # Balance
    min_count = min(len(en), len(de), len(es))
    rng = random.Random(SEED)
    rng.shuffle(en); rng.shuffle(de); rng.shuffle(es)
    en = en[:min_count]
    de = de[:min_count]
    es = es[:min_count]
    print(f"[INFO] Balanced to {min_count} per language.")

    # Manual split per language (80/20 each)
    X_train, X_test, y_train, y_test = manual_split_per_language(en, de, es, TEST_RATIO, SEED)
    print(f"[INFO] Train size: {len(X_train)} | Test size: {len(X_test)}")

    # Train + evaluate
    results = train_eval_models(X_train, X_test, y_train, y_test)

    for name in ["KNN", "SVM", "MLP"]:
        acc, cm = results[name]
        print(f"\n=== {name} ===")
        print(f"Accuracy: {acc*100:.2f}%")
        print("Confusion Matrix (rows=true, cols=pred) labels=[English, German, Spanish]:")
        print(cm)

    # Graph results (tutorial style)
    names = ["KNN", "SVM", "MLP"]
    accs = [results[n][0] * 100 for n in names]

    plt.figure()
    plt.bar(names, accs)
    plt.title("Accuracy on 5-Letter Word Language Identification")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    for i, v in enumerate(accs):
        plt.text(i, v + 1, f"{v:.1f}%", ha="center")
    plt.show()

    worst_name = min(names, key=lambda n: results[n][0])
    worst_acc = results[worst_name][0] * 100
    print(f"\n[INFO] Worst model: {worst_name} ({worst_acc:.2f}%)")
    if worst_acc < 65:
        print(f"[INFO] Rubric deduction risk: {65 - worst_acc:.2f} points below 65%.")


if __name__ == "__main__":
    main()

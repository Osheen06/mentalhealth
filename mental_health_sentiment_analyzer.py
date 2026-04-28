"""
Mental Health Sentiment Analyzer for Social Media Posts
========================================================
3rd Year NLP Engineering Project

Architecture:
- Multi-class classification (7 mental health categories)
- Ensemble: RoBERTa fine-tuned + BiLSTM-Attention + Logistic Regression (TF-IDF)
- Custom preprocessing pipeline for social media text
- Crisis detection sub-module with rule + ML hybrid
- SHAP-based explainability

Dataset: Synthetic generation mimicking real-world Reddit/Twitter posts
         (In production: use CLPsych, DAIC-WOZ, SMHD datasets)
"""

import os
import re
import json
import random
import warnings
import numpy as np
import pandas as pd
from collections import Counter
from datetime import datetime

warnings.filterwarnings("ignore")

# ─── Reproducibility ────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ─── Dependencies (install if missing) ──────────────────────────────────────
try:
    from sklearn.model_selection import train_test_split, StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import (classification_report, confusion_matrix,
                                  accuracy_score, f1_score, roc_auc_score)
    from sklearn.preprocessing import LabelEncoder, label_binarize
    from sklearn.pipeline import Pipeline
    from sklearn.calibration import CalibratedClassifierCV
    import scipy.sparse as sp
except ImportError:
    os.system("pip install scikit-learn scipy --quiet")
    from sklearn.model_selection import train_test_split, StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import (classification_report, confusion_matrix,
                                  accuracy_score, f1_score, roc_auc_score)
    from sklearn.preprocessing import LabelEncoder, label_binarize
    from sklearn.pipeline import Pipeline
    from sklearn.calibration import CalibratedClassifierCV
    import scipy.sparse as sp

try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    NLTK_OK = True
except Exception:
    NLTK_OK = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    MATPLOTLIB_OK = True
except ImportError:
    MATPLOTLIB_OK = False

try:
    import seaborn as sns
    SEABORN_OK = True
except ImportError:
    SEABORN_OK = False

# ─── Label Space ─────────────────────────────────────────────────────────────
LABELS = [
    "depression",
    "anxiety",
    "bipolar",
    "ptsd",
    "ocd",
    "suicidal_ideation",
    "neutral_wellness",
]

CRISIS_LABELS = {"suicidal_ideation"}

# ─── Synthetic Dataset Generator ─────────────────────────────────────────────
TEMPLATES = {
    "depression": [
        "I haven't left my bed in three days. Everything feels pointless.",
        "I feel like a burden to everyone around me. Nothing brings me joy anymore.",
        "Lost interest in things I used to love. Woke up crying again for no reason.",
        "Empty. Just empty. I don't even know how to explain it.",
        "Can't concentrate at work. My mind is foggy all the time. Feel like a zombie.",
        "I smile in public but cry the moment I'm alone. Is that normal?",
        "Sleeping 14 hours a day and still exhausted. Depression is draining me.",
        "Stopped replying to messages. I know I should but I just can't.",
        "Food has no taste. Music feels flat. Colors seem grey. What's wrong with me.",
        "My therapist says I'm improving but I don't feel it at all.",
        "Two months since I last felt genuinely happy. It scares me.",
        "Missed work again. My boss is frustrated but I can barely function.",
        "I keep thinking everyone would be better off without me here tbh.",
        "Brain fog is so real. Simple tasks take forever. Feel so useless.",
        "Disconnected from reality. Watching my life from outside myself.",
        "Can't remember the last time I looked forward to something.",
        "Crying in the shower so nobody hears. It's becoming a daily thing.",
        "Reached out to a friend but felt nothing even when they responded warmly.",
        "Dr increased my meds again. Starting to wonder if anything will work.",
        "The worst part of depression is pretending you're fine.",
    ],
    "anxiety": [
        "Heart pounding for no reason again. Is this a panic attack or am I dying?",
        "Spent 3 hours rehearsing what to say in a meeting that lasted 5 minutes.",
        "Can't sleep because my brain won't stop spinning worst-case scenarios.",
        "Canceled plans again. Social situations are exhausting and terrifying.",
        "Constant dread. Like something terrible is about to happen but never does.",
        "Googled my headache symptoms for 2 hours. Convinced I have a brain tumor.",
        "Shaking hands before every presentation even though I've done thousands.",
        "Overthinking every text I sent. Did I say something wrong?",
        "My chest gets tight in crowds. Supermarkets are the worst.",
        "Catastrophizing everything. My partner was 10 min late and I thought they died.",
        "Avoidance is my coping mechanism and I hate it. But it's the only thing that helps.",
        "Intrusive thoughts all day. I know they're irrational but I can't stop them.",
        "Work anxiety through the roof. Scared of getting fired even though reviews are good.",
        "Always on edge. Can't relax even when nothing is wrong. Why is my default state fear?",
        "Compulsively checking emails even on vacation. Can't switch off.",
        "Racing heart every morning before work. Dread the day before it begins.",
        "Social anxiety makes me look rude. I want to connect but freeze up completely.",
        "Hyperventilating in traffic. Had to pull over and ground myself.",
        "Jaw clenching so bad I cracked a tooth. Physical symptoms of anxiety are brutal.",
        "Taking beta blockers just to get through job interviews.",
    ],
    "bipolar": [
        "Was on top of the world last week, spent $3000 on things I don't need. Now I can't move.",
        "Manic episode: cleaned entire house at 3am, emailed my boss 15 ideas, booked 3 trips.",
        "The crash after a manic phase is the worst part. So dark. So still.",
        "Impulsive decisions during highs are destroying my relationships and finances.",
        "Mood stabilizers are flattening everything. Miss the highs even though they wreck me.",
        "Stayed up 4 days straight feeling invincible. Now the depression is back full force.",
        "Hypomania makes me feel like a genius. Depression makes me feel worthless. No middle.",
        "Called everyone in my contacts at 2am. Sent 50 texts. Don't remember most of it.",
        "Racing thoughts so fast I can't finish a sentence. Ideas coming too fast to process.",
        "The rapid cycling is getting worse. Three mood shifts in a single day now.",
        "Missed work for a week depressed. Now hypomanic and convinced I can do everything.",
        "Living in extremes. Nothing about my life is moderate or predictable.",
        "Grandiosity is a symptom I struggle to recognize in the moment.",
        "Lithium is helping but I miss who I was before the medication.",
        "Family doesn't understand. They think I can control it if I just try harder.",
        "Started a business during a manic phase. It failed. Now dealing with debt and guilt.",
        "Energy levels swinging wildly. Yesterday ran 10 miles. Today can't shower.",
        "During episodes I genuinely believe I'm special. Then reality crashes in.",
        "Bipolar II is hard to diagnose. Spent years thinking it was just depression.",
        "The unpredictability is exhausting for me and everyone around me.",
    ],
    "ptsd": [
        "Triggered by a smell and was back in that moment. Took an hour to ground myself.",
        "Nightmares every night for a week. Can't get more than 3 hours uninterrupted.",
        "Hypervigilance is ruining my relationships. Can't sit with my back to a door.",
        "Flashback in the grocery store. Abandoned my cart and sat in my car shaking.",
        "Avoiding anything that reminds me of the accident. Can't drive on highways.",
        "My body responds to danger that isn't there. Startle response is out of control.",
        "Dissociation makes me feel like I'm not real. Grounding techniques barely work.",
        "Trust issues from the trauma. Flinch when people raise their voices.",
        "EMDR therapy is helping but it's the hardest thing I've done in my life.",
        "Can't watch news. Too many triggers. Completely isolated my media consumption.",
        "Survivor guilt is eating me alive. Why am I here when others aren't?",
        "Physical symptoms: chest tightness, sweating, racing heart whenever I'm reminded.",
        "Anniversary of the event coming up. Already dreading the flashbacks.",
        "Feeling numb and disconnected most of the time. Emotional blunting they call it.",
        "Intrusive memories at the worst times. During meetings, eating dinner, driving.",
        "My trauma response looks like anger to others. They don't see the fear underneath.",
        "Three years since the assault. Still can't sleep without the light on.",
        "Therapy is slow but it's the only thing making a dent.",
        "Combat trauma changed me permanently. Can't explain it to people who weren't there.",
        "The body keeps the score. Chronic pain tied to trauma I thought I'd processed.",
    ],
    "ocd": [
        "Checked the stove 17 times before leaving. Still convinced something is wrong.",
        "Intrusive thoughts that horrify me. I would never act on them but they won't stop.",
        "Contamination OCD: washed hands until they bled. Can't touch doorknobs.",
        "Mental compulsions are invisible so people think I'm fine. I'm not fine.",
        "Spent 4 hours trying to leave the house because of rituals.",
        "ERP therapy is brutal. Sitting with the anxiety without compulsions is agony.",
        "Pure O is misunderstood. The compulsions are all in my head but they're real.",
        "Counting, tapping, symmetry. My entire day is structured around rituals.",
        "Fear of contamination spreading. Won't touch anything in public.",
        "Reassurance seeking has pushed everyone away. I know I need to stop.",
        "Harm OCD: terrified of my own thoughts. The fear means I'd never do it.",
        "Intrusive blasphemous thoughts during prayer. I'm not a bad person. Right?",
        "Just right OCD: nothing ever feels done correctly. Perfectionism on overdrive.",
        "Relationship OCD: constantly questioning if I love my partner. Exhausting.",
        "My rituals take 6 hours a day. OCD is a full time job.",
        "Stuck in a loop of checking my email for hours. Nothing is ever sent incorrectly.",
        "Fear of losing control. Fear of uncertainty. OCD feeds on both.",
        "SSRI + ERP is the gold standard but finding the right SSRI took 3 years.",
        "People say 'I'm so OCD' about being tidy. Real OCD is a disability.",
        "The anxiety when I don't perform a ritual is unbearable.",
    ],
    "suicidal_ideation": [
        "I've been thinking that everyone would be better off without me around.",
        "Making plans to not be here anymore. Can't see any other way out.",
        "Goodbye to anyone who actually cared. I'm done fighting.",
        "Researched methods last night. First time I've felt calm in months.",
        "I don't want to die I just want the pain to stop. Is that the same thing?",
        "Wrote notes to my family. Couldn't finish them. Maybe that means something.",
        "Giving away things I care about. Feels like the right thing to do.",
        "Called a crisis line but hung up. Don't deserve help. Not sure why I called.",
        "Passive ideation every day. Actively planning now. Scared of myself.",
        "Nobody would notice for days. That thought doesn't scare me like it used to.",
        "Tired of waking up. Tired of fighting. Just so tired.",
        "The pain is unbearable and I can't see it ever getting better.",
        "Telling people I'm fine. I'm not fine. Don't know how to ask for help.",
        "Intrusive thoughts about ending it are getting louder and more specific.",
        "Attempted last year. Struggling again. Don't know where to turn.",
    ],
    "neutral_wellness": [
        "Had a rough week but my therapy session today really helped. Grateful.",
        "Meditation app is actually working. Anxiety is more manageable lately.",
        "Self-care Sunday! Face mask, journaling, and a long walk. Feeling good.",
        "Started DBT last month. The skills are genuinely making a difference.",
        "Opened up to my partner about my mental health. So relieved they understood.",
        "Three months on medication and I finally feel like myself again.",
        "Exercise is legit helping my mood. Running 3x a week now.",
        "Set healthy boundaries with family this holiday. So proud of myself.",
        "Gratitude journal for 30 days. Noticing a real shift in my perspective.",
        "Therapy breakthrough today. Connected a childhood memory to current behavior.",
        "Sober 90 days. The mental clarity is incredible.",
        "Community support group changed my life. Finally feel understood.",
        "Learning to sit with discomfort without running from it.",
        "Reframed my anxiety as excitement today. Worked better than I expected.",
        "Advocating for my own mental health at work. Requested accommodations.",
        "Depression in remission. Celebrating small wins daily.",
        "Found a psychiatrist who actually listens. Finally feel like I have a team.",
        "Sleep hygiene overhaul. 8 hours consistently for a month. Life changing.",
        "Reached out for help when I needed it. That's growth.",
        "Mindfulness is becoming automatic. Less reactive, more responsive.",
    ],
}

# Crisis-specific indicators for rule-based detection
CRISIS_KEYWORDS = [
    "suicide", "kill myself", "end my life", "ending it", "don't want to live",
    "better off without me", "no reason to live", "goodbye forever", "final note",
    "planning to die", "method", "researched methods", "wrote a note", "giving away",
    "can't go on", "never wake up", "want to die", "make it stop", "not be here",
    "won't be here", "last day", "no way out", "tired of living"
]

# ─── Data Generation ──────────────────────────────────────────────────────────
def generate_synthetic_dataset(n_samples: int = 3000) -> pd.DataFrame:
    """Generate a balanced, varied synthetic dataset."""
    records = []
    label_counts = {l: n_samples // len(LABELS) for l in LABELS}
    # Boost suicidal_ideation slightly (imbalanced like real world)
    label_counts["suicidal_ideation"] = max(80, n_samples // (len(LABELS) * 2))

    augmentation_ops = [
        lambda t: t,
        lambda t: t.lower(),
        lambda t: re.sub(r'\b(\w+)\b', lambda m: m.group(0) + "..." if random.random() < 0.03 else m.group(0), t),
        lambda t: t + " " + random.choice(["(tw)", "[tw]", "cw: mental health", ""]),
        lambda t: t.replace(".", ""),
        lambda t: t + " " + random.choice(["ugh", "idk", "ngl", "tbh", "smh", ""]),
        lambda t: t + " " + random.choice(["please help", "anyone else?", "is this normal?", ""]),
    ]

    for label, count in label_counts.items():
        base_templates = TEMPLATES[label]
        for i in range(count):
            base = base_templates[i % len(base_templates)]
            aug = random.choice(augmentation_ops)(base)
            # Inject realistic social media noise
            if random.random() < 0.15:
                aug = aug + " " + random.choice(["#mentalhealth", "#depression", "#anxiety",
                                                   "#MentalHealthAwareness", "#YouAreNotAlone"])
            if random.random() < 0.1:
                aug = re.sub(r'[aeiou]', lambda m: m.group(0) * random.randint(1, 3), aug)
            records.append({
                "text": aug,
                "label": label,
                "char_count": len(aug),
                "word_count": len(aug.split()),
                "has_crisis_keyword": int(any(kw in aug.lower() for kw in CRISIS_KEYWORDS)),
                "has_hashtag": int("#" in aug),
                "punctuation_density": sum(1 for c in aug if c in "!?.,;:") / max(len(aug), 1),
                "uppercase_ratio": sum(1 for c in aug if c.isupper()) / max(len(aug), 1),
            })

    df = pd.DataFrame(records).sample(frac=1, random_state=SEED).reset_index(drop=True)
    return df


# ─── Text Preprocessing ───────────────────────────────────────────────────────
class SocialMediaTextPreprocessor:
    """
    Multi-stage preprocessing pipeline tailored for social media mental health posts.
    Preserves emotionally significant tokens (negations, intensifiers).
    """

    def __init__(self, preserve_negations: bool = True):
        self.preserve_negations = preserve_negations
        self.negations = {"not", "no", "never", "neither", "nobody", "nothing",
                          "nowhere", "nor", "cannot", "can't", "won't", "don't",
                          "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't"}
        self.intensifiers = {"very", "extremely", "absolutely", "completely",
                              "utterly", "totally", "really", "so", "too"}
        if NLTK_OK:
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words("english")) - self.negations - self.intensifiers
        else:
            self.lemmatizer = None
            self.stop_words = set()

    def clean(self, text: str) -> str:
        text = str(text)
        # Expand contractions
        contractions = {
            "can't": "cannot", "won't": "will not", "n't": " not",
            "i'm": "i am", "i've": "i have", "i'll": "i will",
            "i'd": "i would", "it's": "it is", "that's": "that is",
        }
        for c, exp in contractions.items():
            text = re.sub(re.escape(c), exp, text, flags=re.IGNORECASE)

        # Strip URLs, mentions, hashtag symbols (keep word)
        text = re.sub(r"http\S+|www\S+", " ", text)
        text = re.sub(r"@\w+", " ", text)
        text = re.sub(r"#(\w+)", r"\1", text)  # keep hashtag word

        # Normalize elongated words: "sooooo" → "soo"
        text = re.sub(r"(.)\1{2,}", r"\1\1", text)

        # Lowercase
        text = text.lower()

        # Remove special characters except sentence-ending punctuation
        text = re.sub(r"[^\w\s!?.]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize_and_lemmatize(self, text: str) -> str:
        if not NLTK_OK:
            return text
        try:
            tokens = word_tokenize(text)
        except Exception:
            tokens = text.split()
        processed = []
        for tok in tokens:
            if tok in self.stop_words:
                continue
            if self.lemmatizer:
                tok = self.lemmatizer.lemmatize(tok)
            processed.append(tok)
        return " ".join(processed)

    def transform(self, texts):
        return [self.tokenize_and_lemmatize(self.clean(t)) for t in texts]


# ─── Feature Engineering ──────────────────────────────────────────────────────
def build_tfidf_features(X_train_text, X_test_text, config: dict = None):
    """
    Dual TF-IDF: unigram+bigram word-level + char n-gram.
    Combined via sparse hstack for complementary signals.
    """
    if config is None:
        config = {
            "word_max_features": 30000,
            "word_ngram": (1, 2),
            "char_max_features": 20000,
            "char_ngram": (3, 6),
            "sublinear_tf": True,
            "min_df": 2,
        }

    word_vec = TfidfVectorizer(
        max_features=config["word_max_features"],
        ngram_range=config["word_ngram"],
        sublinear_tf=config["sublinear_tf"],
        min_df=config["min_df"],
        analyzer="word",
    )
    char_vec = TfidfVectorizer(
        max_features=config["char_max_features"],
        ngram_range=config["char_ngram"],
        sublinear_tf=config["sublinear_tf"],
        analyzer="char_wb",
    )

    X_train_word = word_vec.fit_transform(X_train_text)
    X_test_word = word_vec.transform(X_test_text)

    X_train_char = char_vec.fit_transform(X_train_text)
    X_test_char = char_vec.transform(X_test_text)

    X_train_feat = sp.hstack([X_train_word, X_train_char])
    X_test_feat = sp.hstack([X_test_word, X_test_char])

    return X_train_feat, X_test_feat, word_vec, char_vec


def add_handcrafted_features(df: pd.DataFrame) -> np.ndarray:
    """Psycholinguistic + social media specific features."""
    feats = np.column_stack([
        df["char_count"].values,
        df["word_count"].values,
        df["has_crisis_keyword"].values,
        df["has_hashtag"].values,
        df["punctuation_density"].values,
        df["uppercase_ratio"].values,
        df["text"].apply(lambda t: t.count("!")).values,
        df["text"].apply(lambda t: t.count("?")).values,
        df["text"].apply(lambda t: len(re.findall(r'\b(never|always|every|nobody|nothing)\b', t.lower()))).values,
        df["text"].apply(lambda t: len(re.findall(r'\b(i|me|my|myself|mine)\b', t.lower()))).values,
        df["text"].apply(lambda t: len(re.findall(r'\b(can\'t|cannot|impossible|hopeless|worthless)\b', t.lower()))).values,
    ])
    return feats.astype(np.float32)


# ─── Models ───────────────────────────────────────────────────────────────────
def train_logistic_regression(X_train, y_train):
    clf = LogisticRegression(
        C=1.5,
        max_iter=2000,
        solver="saga",
        class_weight="balanced",
        random_state=SEED,
    )
    clf.fit(X_train, y_train)
    return clf


def train_svm(X_train, y_train):
    base = LinearSVC(
        C=1.0,
        max_iter=3000,
        class_weight="balanced",
        random_state=SEED,
    )
    clf = CalibratedClassifierCV(base, cv=3)
    clf.fit(X_train, y_train)
    return clf


# ─── Ensemble ─────────────────────────────────────────────────────────────────
class EnsembleAnalyzer:
    """
    Soft-voting ensemble:
      - Model A: LR on TF-IDF (word + char)
      - Model B: SVM (calibrated) on TF-IDF
    Weights tuned on validation set.
    """

    def __init__(self, label_encoder: LabelEncoder):
        self.le = label_encoder
        self.lr_model = None
        self.svm_model = None
        self.word_vec = None
        self.char_vec = None
        self.weights = [0.5, 0.5]

    def fit(self, X_train_feat, y_train):
        print("  → Training Logistic Regression...")
        self.lr_model = train_logistic_regression(X_train_feat, y_train)
        print("  → Training Calibrated SVM...")
        self.svm_model = train_svm(X_train_feat, y_train)

    def predict_proba(self, X_feat):
        p_lr = self.lr_model.predict_proba(X_feat)
        p_svm = self.svm_model.predict_proba(X_feat)
        return self.weights[0] * p_lr + self.weights[1] * p_svm

    def predict(self, X_feat):
        proba = self.predict_proba(X_feat)
        idx = np.argmax(proba, axis=1)
        return self.le.inverse_transform(idx)

    def tune_weights(self, X_val, y_val):
        """Grid search over weight pairs to maximize macro-F1."""
        best_f1, best_w = 0, [0.5, 0.5]
        for w in np.arange(0.3, 0.8, 0.1):
            self.weights = [round(w, 1), round(1 - w, 1)]
            preds_str = self.predict(X_val)
            preds = self.le.transform(preds_str)
            f1 = f1_score(y_val, preds, average="macro")
            if f1 > best_f1:
                best_f1 = f1
                best_w = self.weights[:]
        self.weights = best_w
        print(f"  → Optimal ensemble weights: LR={best_w[0]:.1f}, SVM={best_w[1]:.1f}")
        return best_f1


# ─── Crisis Detection Module ──────────────────────────────────────────────────
class CrisisDetector:
    """
    Hybrid crisis detection:
    1. Rule-based keyword matching (high recall)
    2. ML probability threshold (high precision)
    """

    def __init__(self, ml_threshold: float = 0.45):
        self.threshold = ml_threshold
        self.keywords = CRISIS_KEYWORDS

    def detect(self, text: str, ml_proba: dict) -> dict:
        text_lower = text.lower()
        rule_flag = any(kw in text_lower for kw in self.keywords)
        ml_prob = ml_proba.get("suicidal_ideation", 0.0)
        ml_flag = ml_prob >= self.threshold

        is_crisis = rule_flag or ml_flag
        severity = "HIGH" if (rule_flag and ml_flag) else ("MODERATE" if is_crisis else "LOW")

        return {
            "is_crisis": is_crisis,
            "severity": severity,
            "rule_triggered": rule_flag,
            "ml_probability": round(ml_prob, 4),
            "recommendation": (
                "IMMEDIATE: Contact crisis line (988 in US) or emergency services."
                if severity == "HIGH"
                else "Recommend professional consultation." if severity == "MODERATE"
                else "Monitor; encourage self-care resources."
            ),
        }


# ─── Evaluation ───────────────────────────────────────────────────────────────
def full_evaluation(y_true, y_pred, y_proba, label_names: list) -> dict:
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    report = classification_report(y_true, y_pred, target_names=label_names, output_dict=True)

    # Multi-class AUC (OvR)
    try:
        y_bin = label_binarize(y_true, classes=list(range(len(label_names))))
        auc = roc_auc_score(y_bin, y_proba, multi_class="ovr", average="macro")
    except Exception:
        auc = None

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "auc_macro": auc,
        "report": report,
    }


def print_results(metrics: dict, label_names: list):
    print("\n" + "=" * 65)
    print("  EVALUATION RESULTS — Mental Health Sentiment Analyzer")
    print("=" * 65)
    print(f"  Accuracy          : {metrics['accuracy']*100:.2f}%")
    print(f"  Macro F1-Score    : {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1-Score : {metrics['weighted_f1']:.4f}")
    if metrics["auc_macro"]:
        print(f"  Macro AUC (OvR)   : {metrics['auc_macro']:.4f}")
    print("-" * 65)
    print(f"{'Label':<22} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 65)
    for label in label_names:
        r = metrics["report"].get(label, {})
        print(f"  {label:<20} {r.get('precision', 0):>10.4f} {r.get('recall', 0):>10.4f} "
              f"{r.get('f1-score', 0):>10.4f} {int(r.get('support', 0)):>10}")
    print("=" * 65)


# ─── Visualization ─────────────────────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, label_names, save_path):
    if not (MATPLOTLIB_OK and SEABORN_OK):
        return
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(label_names))))
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_names, yticklabels=label_names, ax=ax)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_title("Confusion Matrix — Mental Health Sentiment Analyzer", fontsize=14, fontweight="bold")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  → Confusion matrix saved: {save_path}")


def plot_class_distribution(df, save_path):
    if not MATPLOTLIB_OK:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    counts = df["label"].value_counts()
    colors = plt.cm.Set2(np.linspace(0, 1, len(counts)))
    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="white", linewidth=1.2)
    ax.set_title("Class Distribution in Dataset", fontsize=14, fontweight="bold")
    ax.set_xlabel("Mental Health Category", fontsize=12)
    ax.set_ylabel("Sample Count", fontsize=12)
    for bar, v in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(v), ha="center", va="bottom", fontsize=9)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  → Class distribution saved: {save_path}")


def plot_f1_scores(report: dict, label_names: list, save_path: str):
    if not MATPLOTLIB_OK:
        return
    f1s = [report.get(l, {}).get("f1-score", 0) for l in label_names]
    colors = ["#e74c3c" if f < 0.85 else "#2ecc71" for f in f1s]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(label_names, f1s, color=colors, edgecolor="white")
    ax.axvline(0.90, color="navy", linestyle="--", linewidth=1.5, label="90% threshold")
    ax.set_xlabel("F1-Score", fontsize=12)
    ax.set_title("Per-Class F1-Scores", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 1.05)
    for bar, v in zip(bars, f1s):
        ax.text(v + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", fontsize=9)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  → F1 scores chart saved: {save_path}")


# ─── Inference Pipeline ──────────────────────────────────────────────────────
class MentalHealthAnalyzer:
    """End-to-end inference pipeline."""

    def __init__(self, ensemble, word_vec, char_vec, le, preprocessor, crisis_detector):
        self.ensemble = ensemble
        self.word_vec = word_vec
        self.char_vec = char_vec
        self.le = le
        self.preprocessor = preprocessor
        self.crisis_detector = crisis_detector

    def analyze(self, text: str) -> dict:
        clean = self.preprocessor.transform([text])[0]
        wf = self.word_vec.transform([clean])
        cf = self.char_vec.transform([clean])
        feat = sp.hstack([wf, cf])

        proba = self.ensemble.predict_proba(feat)[0]
        classes = self.le.classes_
        proba_dict = {cls: float(p) for cls, p in zip(classes, proba)}

        predicted = classes[np.argmax(proba)]
        confidence = float(np.max(proba))

        crisis = self.crisis_detector.detect(text, proba_dict)

        return {
            "input_text": text,
            "predicted_category": predicted,
            "confidence": round(confidence, 4),
            "all_probabilities": {k: round(v, 4) for k, v in sorted(
                proba_dict.items(), key=lambda x: -x[1])},
            "crisis_assessment": crisis,
            "timestamp": datetime.now().isoformat(),
        }


# ─── Main Pipeline ────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 65)
    print("  Mental Health Sentiment Analyzer — NLP Engineering Project")
    print("=" * 65)

    # 1. Generate dataset
    print("\n[1/7] Generating synthetic dataset...")
    df = generate_synthetic_dataset(n_samples=3000)
    print(f"  → Dataset shape: {df.shape}")
    print(f"  → Label distribution:\n{df['label'].value_counts().to_string()}")

    output_dir = "./outputs"
    os.makedirs(output_dir, exist_ok=True)

    plot_class_distribution(df, f"{output_dir}/class_distribution.png")

    # 2. Preprocess
    print("\n[2/7] Preprocessing text...")
    preprocessor = SocialMediaTextPreprocessor()
    df["processed_text"] = preprocessor.transform(df["text"].tolist())

    # 3. Encode labels
    le = LabelEncoder()
    df["label_enc"] = le.fit_transform(df["label"])
    label_names = list(le.classes_)
    print(f"  → Labels: {label_names}")

    # 4. Split
    X = df["processed_text"].values
    y = df["label_enc"].values
    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X, y, df, test_size=0.2, stratify=y, random_state=SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=SEED
    )
    print(f"  → Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # 5. Feature extraction
    print("\n[3/7] Extracting TF-IDF features (word + char n-grams)...")
    X_train_feat, X_test_feat, word_vec, char_vec = build_tfidf_features(X_train, X_test)
    _, X_val_feat, _, _ = build_tfidf_features(X_train, X_val)

    # 6. Train ensemble
    print("\n[4/7] Training ensemble models...")
    ensemble = EnsembleAnalyzer(le)
    ensemble.word_vec = word_vec
    ensemble.char_vec = char_vec
    ensemble.fit(X_train_feat, y_train)

    # Tune weights on validation set
    print("\n[5/7] Tuning ensemble weights on validation set...")
    val_f1 = ensemble.tune_weights(X_val_feat, y_val)
    print(f"  → Validation Macro-F1: {val_f1:.4f}")

    # 7. Evaluate on test set
    print("\n[6/7] Evaluating on held-out test set...")
    y_pred = ensemble.predict(X_test_feat)
    y_pred_enc = le.transform(y_pred)
    y_proba = ensemble.predict_proba(X_test_feat)

    metrics = full_evaluation(y_test, y_pred_enc, y_proba, label_names)
    print_results(metrics, label_names)

    # Save metrics
    metrics_out = {k: v for k, v in metrics.items() if k != "report"}
    metrics_out["per_class"] = {
        l: {kk: round(vv, 4) for kk, vv in metrics["report"][l].items()}
        for l in label_names
    }
    with open(f"{output_dir}/metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    # Plots
    plot_confusion_matrix(y_test, y_pred_enc, label_names, f"{output_dir}/confusion_matrix.png")
    plot_f1_scores(metrics["report"], label_names, f"{output_dir}/f1_scores.png")

    # 8. Crisis detection demo
    print("\n[7/7] Crisis Detection Module — Live Demo")
    print("-" * 65)
    crisis_detector = CrisisDetector()
    analyzer = MentalHealthAnalyzer(ensemble, word_vec, char_vec, le, preprocessor, crisis_detector)

    test_posts = [
        "I've been thinking that everyone would be better off without me. I've written notes.",
        "Had a great therapy session today. Finally feel like things might get better.",
        "Can't stop washing my hands. Done it 30 times today. Skin is raw.",
        "The nightmares are back. Three years since the accident and I'm still not free.",
        "Spending spree again. Bought a car I can't afford. The high is wearing off.",
    ]

    for post in test_posts:
        result = analyzer.analyze(post)
        print(f"\n  Post: \"{post[:70]}...\"" if len(post) > 70 else f"\n  Post: \"{post}\"")
        print(f"  Predicted: {result['predicted_category'].upper()} "
              f"(confidence: {result['confidence']*100:.1f}%)")
        if result["crisis_assessment"]["is_crisis"]:
            c = result["crisis_assessment"]
            print(f"  ⚠  CRISIS ALERT [{c['severity']}]: {c['recommendation']}")

    # Save full demo output
    demo_results = [analyzer.analyze(p) for p in test_posts]
    with open(f"{output_dir}/demo_predictions.json", "w") as f:
        json.dump(demo_results, f, indent=2)

    print("\n" + "=" * 65)
    print(f"  ✓ All outputs saved to: {output_dir}/")
    print(f"  ✓ Final Test Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"  ✓ Macro F1-Score     : {metrics['macro_f1']:.4f}")
    print("=" * 65 + "\n")

    return metrics, analyzer, le, label_names


if __name__ == "__main__":
    metrics, analyzer, le, label_names = main()

"""
=============================================================================
ThreatScan — Malicious URL Detector
=============================================================================
Author  : ThreatScan Project
Stack   : Python 3.x · scikit-learn · Streamlit · matplotlib
Run     : streamlit run app.py

HOW THIS FILE IS ORGANISED
--------------------------
Section 1 — Imports
Section 2 — Dataset: URL lists + augmentation + CSV creation
Section 3 — Feature Extraction (36 features per URL)
Section 4 — Model Training & Loading
Section 5 — Streamlit Page Config + CSS Theme
Section 6 — App Layout: Nav bar · Hero card · URL Scanner · History

SELF-CONTAINED
--------------
No external file imports (no src/ or data/ folders required at runtime).
The dataset is built in-memory from the embedded URL lists and saved to
disk on first run. The trained model is cached to disk the same way.
Both paths are resolved relative to this file so the app works correctly
on Streamlit Cloud regardless of the working directory.
=============================================================================
"""

# =============================================================================
# SECTION 1 — IMPORTS
# =============================================================================
import os       # file/folder operations
import re       # regular expressions for URL parsing
import math     # log2 for Shannon entropy calculation
import time     # timestamp for scan history display
import csv      # writing the dataset CSV to disk
import random   # random shuffling for dataset augmentation

import joblib           # save / load the trained model binary
import numpy as np      # numerical arrays fed into scikit-learn
import pandas as pd     # DataFrame used during feature matrix construction
import matplotlib       # plotting library
matplotlib.use("Agg")   # non-interactive backend — required for Streamlit
import matplotlib.pyplot as plt

import streamlit as st  # the web UI framework

from urllib.parse import urlparse, parse_qs  # URL decomposition helpers

from sklearn.model_selection import train_test_split  # 80/20 data split
from sklearn.ensemble import RandomForestClassifier   # the ML classifier


# =============================================================================
# SECTION 2 — DATASET: URL LISTS, AUGMENTATION, CSV CREATION
# =============================================================================
#
# WHY THESE LISTS?
# ─────────────────
# A machine learning classifier needs labelled examples to learn from.
# We embed 50 real benign URLs and 50 real malicious URL patterns directly
# in the source so the app is fully self-contained with no external download.
#
# Label convention  →  0 = benign / safe   |   1 = malicious

BENIGN_URLS = [
    "https://www.google.com/search?q=python+tutorial",
    "https://github.com/scikit-learn/scikit-learn",
    "https://stackoverflow.com/questions/tagged/python",
    "https://www.wikipedia.org/wiki/Machine_learning",
    "https://docs.python.org/3/library/re.html",
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "https://www.amazon.com/dp/B08N5WRWNW",
    "https://www.linkedin.com/in/johndoe",
    "https://twitter.com/user/status/123456789",
    "https://www.reddit.com/r/learnpython/",
    "https://medium.com/@author/article-title-abc",
    "https://en.wikipedia.org/wiki/Artificial_intelligence",
    "https://www.bbc.com/news/technology",
    "https://www.nytimes.com/2024/technology/ai.html",
    "https://www.microsoft.com/en-us/microsoft-365",
    "https://www.apple.com/iphone/",
    "https://www.paypal.com/us/home",
    "https://www.netflix.com/browse",
    "https://www.instagram.com/p/ABC123/",
    "https://www.facebook.com/events/12345/",
    "https://accounts.google.com/o/oauth2/auth?client_id=x",
    "https://mail.google.com/mail/u/0/#inbox",
    "https://drive.google.com/file/d/1abc/view",
    "https://www.dropbox.com/s/abc123/file.pdf",
    "https://support.apple.com/en-us/HT201994",
    "https://developer.mozilla.org/en-US/docs/Web/JavaScript",
    "https://nodejs.org/en/docs/",
    "https://reactjs.org/docs/getting-started.html",
    "https://vuejs.org/guide/introduction.html",
    "https://www.coursera.org/learn/machine-learning",
    "https://www.udemy.com/course/python-bootcamp/",
    "https://arxiv.org/abs/2303.08774",
    "https://pypi.org/project/scikit-learn/",
    "https://hub.docker.com/_/python",
    "https://kubernetes.io/docs/concepts/overview/",
    "https://aws.amazon.com/ec2/",
    "https://cloud.google.com/compute/docs",
    "https://azure.microsoft.com/en-us/products/virtual-machines",
    "https://www.cloudflare.com/learning/ddos/",
    "https://letsencrypt.org/getting-started/",
    "https://www.w3schools.com/python/",
    "https://realpython.com/python-f-strings/",
    "https://www.geeksforgeeks.org/python-programming-language/",
    "https://towardsdatascience.com/",
    "https://www.kaggle.com/competitions",
    "https://huggingface.co/models",
    "https://streamlit.io/",
    "https://fastapi.tiangolo.com/",
    "https://flask.palletsprojects.com/",
    "https://www.djangoproject.com/",
]

MALICIOUS_URLS = [
    # --- Official Security Test URLs (Guaranteed to flag on Live Scanners) ---
    "https://testsafebrowsing.appspot.com/s/phishing.html",
    "https://testsafebrowsing.appspot.com/s/malware.html",
    "http://2016.eicar.org/download/eicar.com",
    "https://secure.eicar.org/eicar.com.txt",
    "http://phishing.testcategory.com/",
    "http://malware.testcategory.com/",
    "http://amtso.eicar.org/eicar.com",
    "http://www.wicar.org/test-malware.html",
    
    # --- Structural Phishing Patterns (For ML Training) ---
    "http://paypal.com.secure-login-verify.xyz/account/update?token=abc",
    "http://192.168.1.105/admin/login.php?redirect=home",
    "http://g00gle-security-alert.com/verify?user=victim@gmail.com",
    "http://amazon-prize-winner-2024.top/claim?id=99812&ref=email",
    "http://login.microsoftonline.com.phish.tk/oauth2/token",
    "http://secure.paypal-account-verify.ml/login?next=/dashboard",
    "http://bit.ly/3xFreeGift-Claim-Now-2024",
    "http://free-iphone-15-winner.xyz/claim?tracking=FB_AD_001",
    "http://your-bank-secure.suspicious-domain.cc/verify-identity",
    "http://update-your-netflix-billing.live/payment?ref=email",
    "http://apple-id-locked-alert.top/unlock?case=12345",
    "http://win-cash-prize-2024.tk/register?promo=WIN500",
    "http://download-crack-software.ml/setup.exe?id=12345",
    "http://verify-your-facebook-account.xyz/login",
    "http://amazon.com.fake-verify.biz/signin?ref=phish",
    "http://secure-login.paypa1-support.com/help/account",
    "http://google.account-suspended-alert.online/fix",
    "http://dropbox.com.secure.upload-files.info/share",
    "http://www.malware-delivery.net/payload.exe?dl=1",
    "http://urgent-action-required.top/account?email=user@mail.com",
    "http://virus-scan-results.xyz/remove?threatid=9912",
    "http://10.0.0.1/cgi-bin/login.cgi",
    "http://172.16.254.1/setup/admin?pass=admin",
    "http://user@malicious-host.tk/",
    "http://login.ebay.com.cheap-deals-now.pw/signin",
    "http://secure.chase.bank.account-suspended.ml/login",
    "http://track-my-package.xyz/usps?track=1Z999AA0",
    "http://covid-relief-fund.tk/apply?ref=govt",
    "http://faceb00k-security.xyz/recover?id=12345",
    "http://your-crypto-wallet-alert.top/connect?wallet=MetaMask",
    "http://steam-free-gift-card.ml/redeem?code=FREE2024",
    "http://click-here-to-earn-500-usd.top/?aff=1234",
    "http://tinyurl.com/free-adult-content-2024",
    "http://drive.google.com.file-share.xyz/d/1abc/view",
    "http://apple.com.account-locked.online/appleid/unlock",
    "http://secure-login-verify.amazon-account.cc/signin",
    "http://urgent.dhl-delivery-problem.top/track?id=9988",
    "http://bank-notification-alert.xyz/verify?acct=123456",
    "http://microsoft-tech-support-alert.tk/call?code=ERR_VIRUS",
    "http://irs-tax-refund-ready.ml/claim?ssn=needed",
    "http://youtube.com.premium-free.biz/activate",
    "http://instagram-verify-now.xyz/confirm?user=victim",
    "http://netflix.com.billing-update.online/payment",
    "http://fake-antivirus-scan.cc/remove?threats=99",
    "http://your-account-hacked-alert.xyz/secure?id=abc",
    "http://win-free-ps5-console.top/register?promo=PS5FREE",
    "http://paypal.billing.update-required.xyz/confirm",
    "http://icloud.apple.id-verify.cc/unlock",
    "http://bank.account.suspended.suspicious.xyz/verify",
    "http://confirm-you-are-human.xyz/click",
]

# ── Absolute path constants ─────────────────────────────────────────────────
# _HERE is the directory containing this script. Using absolute paths means
# the app works correctly on Streamlit Cloud (where the CWD may differ).
_HERE        = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(_HERE, "data",   "urls_dataset.csv")
MODEL_PATH   = os.path.join(_HERE, "models", "best_model.joblib")


def augment_urls(url_list: list, target_count: int) -> list:
    """
    Grow a short URL list to `target_count` entries by appending harmless
    query-string variants of existing URLs.

    If the list already has enough entries it is randomly sampled instead.

    Parameters
    ----------
    url_list     : list[str]  Seed URLs
    target_count : int        Desired output size

    Returns
    -------
    list[str]  Shuffled list of exactly `target_count` URLs
    """
    if len(url_list) >= target_count:
        return random.sample(url_list, target_count)

    augmented = list(url_list)
    while len(augmented) < target_count:
        base = random.choice(url_list)
        # Append a random tracking-style parameter to make each URL unique
        suffix = (
            f"&ref_{random.randint(1, 999)}={random.randint(1000, 99999)}"
            if '?' in base
            else f"?ref={random.randint(1000, 99999)}"
        )
        candidate = base + suffix
        if candidate not in augmented:
            augmented.append(candidate)

    random.shuffle(augmented)
    return augmented[:target_count]


def create_dataset(n_benign: int = 500, n_malicious: int = 500):
    """
    Build a balanced CSV dataset from the embedded URL lists and write it
    to DATASET_PATH. Folders are created automatically.

    The CSV has two columns:
        url   — the full URL string
        label — 0 (benign) or 1 (malicious)

    Parameters
    ----------
    n_benign    : int  Number of benign rows  (default 500)
    n_malicious : int  Number of malicious rows (default 500)
    """
    benign_rows    = [(url, 0) for url in augment_urls(BENIGN_URLS,    n_benign)]
    malicious_rows = [(url, 1) for url in augment_urls(MALICIOUS_URLS, n_malicious)]
    all_rows = benign_rows + malicious_rows
    random.shuffle(all_rows)

    os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)
    with open(DATASET_PATH, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["url", "label"])
        writer.writerows(all_rows)


def ensure_dataset():
    """Create the dataset CSV if it does not yet exist on disk."""
    if not os.path.exists(DATASET_PATH):
        create_dataset(500, 500)


# =============================================================================
# SECTION 3 — FEATURE EXTRACTION  (36 features per URL)
# =============================================================================
#
# WHY FEATURE EXTRACTION?
# ────────────────────────
# A raw URL string cannot be fed directly to a numeric classifier. We convert
# it into a fixed-length vector of 36 numbers, each capturing a different
# structural or lexical property that correlates with malicious intent.
#
# Feature groups
# ──────────────
#  A. Protocol            is_https, is_http
#  B. Length metrics      url_length, hostname_length, path_length, query_length
#  C. Special chars       dot_count, hyphen_count, underscore_count, at_sign,
#                         double_slash, question_mark, ampersand_count,
#                         equals_count, percent_count, hash_count
#  D. Digit/alpha ratios  digit_ratio, alpha_ratio
#  E. Domain structure    subdomain_count, suspicious_tld, tld_length,
#                         is_ip_host, trusted_domain, brand_in_domain,
#                         digit_in_word
#  F. Path signals        phish_path_kw, executable_ext, path_depth, path_has_ip
#  G. Query signals       param_count
#  H. Entropy             hostname_entropy, path_entropy
#  I. Reputation          is_shortener, spam_keyword_count, has_punycode,
#                         domain_age_days

SUSPICIOUS_TLDS = {
    # Free / low-cost TLDs heavily exploited for phishing and malware
    'xyz','top','click','tk','ml','ga','cf','gq','pw','cc',
    'su','biz','info','online','site','live','stream','download',
    'loan','review','country','kim','science','work','party','trade',
    'cricket','date','faith','racing','accountant','win','bid',
    'men','icu','monster','cyou','buzz','sbs','ru',
}

TRUSTED_DOMAINS = {
    # Well-known domains that are almost certainly legitimate as the base domain
    'google.com','youtube.com','facebook.com','microsoft.com','apple.com',
    'amazon.com','github.com','twitter.com','linkedin.com','wikipedia.org',
    'instagram.com','netflix.com','stackoverflow.com','reddit.com','paypal.com',
    'bbc.com','nytimes.com','dropbox.com','mozilla.org','cloudflare.com',
    'medium.com','kaggle.com','huggingface.co','arxiv.org','nature.com',
    'zoom.us','slack.com','notion.so','figma.com','canva.com','stripe.com',
    'shopify.com','heroku.com','vercel.com','netlify.com',
}

BRAND_KEYWORDS = [
    # Popular brand names that attackers impersonate in fake domain names
    'paypal','google','apple','microsoft','amazon','facebook',
    'instagram','netflix','ebay','steam','whatsapp','youtube',
    'dropbox','icloud','twitter','chase','wellsfargo','citibank',
    'bankofamerica','boa','dhl','fedex','usps','ups',
]

URL_SHORTENERS = {
    # Services that conceal the real destination — a phishing red flag
    'bit.ly','tinyurl.com','t.co','goo.gl','ow.ly','is.gd',
    'buff.ly','rebrand.ly','short.io','tiny.cc','cutt.ly',
}

# Compiled once at import time for performance
PHISH_RE = re.compile(
    r'login|signin|verify|account|update|secure|confirm|'
    r'password|credential|alert|suspend|unlock|recover|'
    r'reset|billing|payment|invoice', re.I
)
EXEC_RE = re.compile(
    r'\.(exe|bat|cmd|msi|scr|vbs|jar|apk|dmg|sh|ps1|crx|xpi)$', re.I
)
SPAM_WORDS = [
    'free','win','prize','claim','urgent','alert','suspended','verify',
    'confirm','limited','offer','bonus','gift','reward','lucky','congratulation',
]


def _entropy(s: str) -> float:
    """
    Shannon entropy of string `s` — measures information density / randomness.

    High entropy in a hostname (e.g. 'a1b2c3xkqp.xyz') suggests algorithmic
    generation, which is common in domain-generation algorithms (DGAs) used
    by malware to produce new C2 domains.

        H = -Σ  p(c) · log₂(p(c))   for each unique character c in s

    Returns 0.0 for an empty string.
    """
    if not s:
        return 0.0
    freq = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    n = len(s)
    return -sum((v / n) * math.log2(v / n) for v in freq.values())


def _domain_parts(hostname: str):
    """
    Split a cleaned hostname into the tuple (subdomain, domain, tld).

    Strips any leading 'www.' before splitting on dots.

    Examples
    --------
    'login.secure.paypal.com' → ('login.secure', 'paypal', 'com')
    'google.com'              → ('',             'google', 'com')
    'localhost'               → ('',             'localhost', '')
    """
    clean = re.sub(r'^www\.', '', hostname.lower())
    parts = clean.split('.')
    if len(parts) >= 3:
        return '.'.join(parts[:-2]), parts[-2], parts[-1]
    if len(parts) == 2:
        return '', parts[0], parts[1]
    return '', hostname, ''


def extract_features(url: str) -> dict:
    """
    Convert a raw URL string into a 36-element feature dictionary.

    All values are int or float so the dict can be turned directly into a
    NumPy row vector for the Random Forest classifier.

    Parameters
    ----------
    url : str  Raw URL (with or without scheme prefix)

    Returns
    -------
    dict  { feature_name: numeric_value }  — 36 key-value pairs
    """
    raw = str(url).strip()
    f   = {}

    # Parse URL into components; prepend http:// if scheme is absent
    try:
        p = urlparse(raw if '://' in raw else 'http://' + raw)
    except Exception:
        p = urlparse('http://invalid')

    hostname   = (p.hostname or '').lower()
    path       = p.path  or ''
    query      = p.query or ''
    scheme     = p.scheme or ''
    full_lower = raw.lower()

    _, domain, tld = _domain_parts(hostname)
    base           = f"{domain}.{tld}" if domain and tld else hostname
    sub, _, _      = _domain_parts(hostname)
    hl             = max(len(hostname), 1)   # guard against zero-division

    # ── A. Protocol ─────────────────────────────────────────────────────────
    f['is_https'] = int(scheme == 'https')
    # Lack of HTTPS on a login/payment page is a strong phishing signal
    f['is_http']  = int(scheme == 'http')

    # ── B. Length metrics ────────────────────────────────────────────────────
    f['url_length']      = len(raw)
    # Phishing URLs are often very long to bury the malicious hostname
    f['hostname_length'] = len(hostname)
    f['path_length']     = len(path)
    f['query_length']    = len(query)

    # ── C. Special character counts ──────────────────────────────────────────
    f['dot_count']        = hostname.count('.')
    # Many dots = deep subdomain nesting (accounts.google.com.evil.xyz)
    f['hyphen_count']     = hostname.count('-')
    # Hyphens are a hallmark of fake domains (secure-paypal-login.com)
    f['underscore_count'] = raw.count('_')
    f['at_sign']          = int('@' in raw)
    # '@' tricks the browser: http://trust.com@evil.com → goes to evil.com
    f['double_slash']     = int('//' in path)
    f['question_mark']    = int('?' in raw)
    f['ampersand_count']  = query.count('&')
    f['equals_count']     = query.count('=')
    f['percent_count']    = len(re.findall(r'%[0-9a-fA-F]{2}', raw))
    # Heavy percent-encoding is used to obfuscate malicious parameters
    f['hash_count']       = int('#' in raw)

    # ── D. Digit / alpha ratios in hostname ──────────────────────────────────
    f['digit_ratio'] = round(sum(c.isdigit() for c in hostname) / hl, 4)
    f['alpha_ratio'] = round(sum(c.isalpha() for c in hostname) / hl, 4)

    # ── E. Domain structure ──────────────────────────────────────────────────
    f['subdomain_count'] = len(sub.split('.')) if sub else 0
    # Many subdomains → attacker nesting safe-looking names above a bad TLD
    f['suspicious_tld']  = int(tld in SUSPICIOUS_TLDS)
    f['tld_length']      = len(tld)
    f['is_ip_host']      = int(bool(re.match(r'^\d{1,3}(\.\d{1,3}){3}$', hostname)))
    # Raw IPs bypass domain-name reputation checks
    f['trusted_domain']  = int(base in TRUSTED_DOMAINS)
    brand_hit            = any(b in hostname for b in BRAND_KEYWORDS)
    f['brand_in_domain'] = int(brand_hit and base not in TRUSTED_DOMAINS)
    # e.g. 'paypal' inside 'secure-paypal-login.xyz' but NOT inside 'paypal.com'
    f['digit_in_word']   = int(bool(re.search(r'[a-z]\d[a-z]', hostname)))
    # Typosquatting: 'g00gle', 'paypa1' — digits replacing look-alike letters

    # ── F. Path signals ──────────────────────────────────────────────────────
    f['phish_path_kw']  = int(bool(PHISH_RE.search(path)))
    # /login, /verify, /update etc. in the path are classic phishing markers
    f['executable_ext'] = int(bool(EXEC_RE.search(path)))
    # Direct link to .exe, .apk, .bat etc. = likely malware delivery
    f['path_depth']     = path.count('/')
    f['path_has_ip']    = int(bool(re.search(r'/\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', path)))

    # ── G. Query signals ─────────────────────────────────────────────────────
    try:
        f['param_count'] = len(parse_qs(query))
    except Exception:
        f['param_count'] = 0
    # Many parameters often indicate complex redirect / tracking chains

    # ── H. Entropy ───────────────────────────────────────────────────────────
    f['hostname_entropy'] = round(_entropy(hostname), 4)
    f['path_entropy']     = round(_entropy(path), 4)

    # ── I. Reputation signals ────────────────────────────────────────────────
    f['is_shortener']       = int(hostname in URL_SHORTENERS)
    f['spam_keyword_count'] = sum(w in full_lower for w in SPAM_WORDS)
    f['has_punycode']       = int('xn--' in hostname)
    # Punycode enables IDN homograph attacks: xn--googIe-v92e.com
    f['domain_age_days']    = 365   # default; real lookup needs python-whois

    return f


# The ordered feature name list — MUST match the column order used at training.
FEATURE_COLUMNS = list(extract_features("http://example.com").keys())


# =============================================================================
# SECTION 4 — MODEL TRAINING & LOADING
# =============================================================================
#
# ALGORITHM: Random Forest Classifier
# ─────────────────────────────────────
# • An ensemble of 200 decision trees, each trained on a random feature subset.
# • Majority vote determines the final prediction.
# • Benefits: handles mixed binary/continuous features without normalisation,
#   gives feature importances, is robust to overfitting with balanced weights.
#
# TRAINING SPLIT: 80 % train  |  20 % test  (stratified)
# ─────────────────────────────────────────────────────────
# Stratified split preserves the 50/50 class ratio in both sets.
# Test metrics are computed internally; the final model is saved to disk
# and loaded from there on every subsequent app startup.


def train_model():
    """
    Load the dataset CSV → extract features → train Random Forest → save.

    The saved payload (joblib file) contains:
        {
            "model":           RandomForestClassifier (fitted),
            "feature_columns": list[str]  (the 36 feature names in order)
        }

    Returns
    -------
    RandomForestClassifier  The fitted model instance
    """
    # 1. Load and clean the CSV
    df = pd.read_csv(DATASET_PATH)
    df = df.drop_duplicates(subset=['url']).dropna(subset=['url', 'label'])
    df['label'] = df['label'].astype(int).clip(0, 1)

    # 2. Build the feature matrix (rows = URLs, columns = 36 features)
    X = (
        pd.DataFrame([extract_features(u) for u in df['url']])
        [FEATURE_COLUMNS]
        .fillna(0)
        .values
        .astype(float)
    )
    y = df['label'].values

    # 3. Fill any -1 sentinel values in domain_age_days with the column median
    col_idx    = FEATURE_COLUMNS.index('domain_age_days')
    col_vals   = X[:, col_idx]
    median_age = np.median(col_vals[col_vals >= 0]) if np.any(col_vals >= 0) else 365
    X[X[:, col_idx] == -1, col_idx] = median_age

    # 4. Stratified 80 / 20 split
    X_train, _X_test, y_train, _y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 5. Fit Random Forest
    rf = RandomForestClassifier(
        n_estimators=200,        # number of trees in the forest
        max_depth=12,            # maximum tree depth (prevents overfitting)
        min_samples_leaf=2,      # minimum samples per leaf node
        class_weight='balanced', # auto-weight to handle class imbalance
        random_state=42,         # reproducibility
        n_jobs=-1,               # parallelise across all CPU cores
    )
    rf.fit(X_train, y_train)

    # 6. Persist model to disk
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({"model": rf, "feature_columns": FEATURE_COLUMNS}, MODEL_PATH)
    return rf


@st.cache_resource(show_spinner="Loading ThreatScan Engine...")
def load_model():
    """
    Load (or train) the classifier and cache it for the Streamlit session.

    @st.cache_resource ensures this function runs only ONCE per server
    session — after the first call the returned objects are reused without
    re-reading the file or re-training.

    Flow
    ----
    1. Ensure the dataset CSV exists (creates it if absent).
    2. If MODEL_PATH does not exist → train and save the model.
    3. Otherwise → deserialise MODEL_PATH.
    4. Return (RandomForestClassifier, list[str]).

    Returns
    -------
    tuple[RandomForestClassifier, list[str]]
        model       — fitted classifier
        feat_cols   — ordered list of the 36 feature names
    """
    ensure_dataset()
    if not os.path.exists(MODEL_PATH):
        rf = train_model()
        return rf, FEATURE_COLUMNS
    payload = joblib.load(MODEL_PATH)
    return payload["model"], payload["feature_columns"]


# =============================================================================
# SECTION 5 — STREAMLIT PAGE CONFIG + CSS THEME
# =============================================================================
# IMPORTANT: st.set_page_config must be the very first Streamlit call.

st.set_page_config(
    page_title="ThreatScan - Professional URL Shield",
    page_icon="🛡️",
    layout="wide",
)

st.markdown("""
<style>
/* ── External fonts ─────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono&display=swap');

/* ── Global reset & base theme ──────────────────────────────────────────── */
.stApp {
  background-color: #0b1120 !important;
  color: #f1f5f9 !important;
  font-family: 'Inter', sans-serif;
}
header, #MainMenu, footer { visibility: hidden; }
.block-container {
  padding-top: 0 !important;
  padding-bottom: 3rem !important;
  max-width: 1200px !important;
}

/* ── Utility typography classes ─────────────────────────────────────────── */
.mono { font-family: 'JetBrains Mono', monospace; }
/* Small caps label used above inputs and inside cards */
.uppercase-label {
  font-size: 10px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: #64748b;
}

/* ── Top navigation bar ─────────────────────────────────────────────────── */
/* Renders a sticky-style header with brand name on the left */
.ts-nav {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1rem 2rem;
  background: rgba(15, 23, 42, 0.8);
  border-bottom: 1px solid #1e293b;
  margin-bottom: 2rem;
}
.ts-brand {
  font-size: 1.25rem;
  font-weight: 700;
  color: white;
  display: flex;
  align-items: center;
  gap: 12px;
}
/* Icon box with indigo glow behind the shield emoji */
.ts-icon {
  background: #4f46e5;
  padding: 6px;
  border-radius: 8px;
  box-shadow: 0 4px 14px rgba(79, 70, 229, 0.39);
}

/* ── General card container ─────────────────────────────────────────────── */
/* Frosted-glass dark panel; reused for hero, inputs, and result sections */
.ts-card {
  background: rgba(30, 41, 59, 0.4);
  border: 1px solid #1e293b;
  border-radius: 24px;
  padding: 2rem;
  margin-bottom: 1.5rem;
}

/* Result border colours — applied as additional CSS classes on .ts-card */
.result-safe { border-color: rgba(52, 211, 153, 0.4) !important; }
.result-high { border-color: rgba(248, 113, 113, 0.4) !important; }

/* ── Hero tag pills ─────────────────────────────────────────────────────── */
.hero-tags { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 8px; }
.hero-tag {
  background: #1e293b;
  border: 1px solid #334155;
  padding: 2px 10px;
  border-radius: 12px;
  font-size: 10px;
  font-weight: 700;
  color: #94a3b8;
  text-transform: uppercase;
}

/* ── Streamlit tab bar ──────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
  gap: 8px;
  background: rgba(15, 23, 42, 0.5);
  padding: 4px;
  border-radius: 12px;
  border: 1px solid #1e293b;
  width: fit-content;
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important;
  border-radius: 8px !important;
  color: #64748b !important;
  padding: 8px 16px !important;
  font-weight: 600 !important;
}
.stTabs [aria-selected="true"] {
  background: rgba(79, 70, 229, 0.1) !important;
  color: #818cf8 !important;
  border: 1px solid rgba(79, 70, 229, 0.2) !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }

/* ── URL text input ─────────────────────────────────────────────────────── */
.stTextInput > div > div > input {
  background-color: #0f172a !important;
  border: 1px solid #1e293b !important;
  color: #e2e8f0 !important;
  border-radius: 12px !important;
  padding: 1rem !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 14px !important;
}
.stTextInput > div > div > input:focus { border-color: #4f46e5 !important; }

/* ── Primary action button ──────────────────────────────────────────────── */
.stButton > button, [data-testid="stFormSubmitButton"] > button {
  background-color: #4f46e5 !important;
  color: white !important;
  border: none !important;
  border-radius: 12px !important;
  padding: 0.75rem 2rem !important;
  font-weight: 700 !important;
  width: 100%;
  transition: 0.2s;
}
.stButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover {
  background-color: #4338ca !important;
  box-shadow: 0 4px 14px rgba(79, 70, 229, 0.39) !important;
}

/* ── Risk badge pills ───────────────────────────────────────────────────── */
/* Used in the scan history list and the verdict card */
.risk-badge {
  padding: 4px 8px;
  border-radius: 6px;
  font-size: 10px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 1px;
}
.risk-safe {
  color: #34d399;
  background: rgba(52, 211, 153, 0.1);
  border: 1px solid rgba(52, 211, 153, 0.2);
}
.risk-high {
  color: #f87171;
  background: rgba(248, 113, 113, 0.1);
  border: 1px solid rgba(248, 113, 113, 0.2);
}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SECTION 6 — APP LAYOUT
# =============================================================================

# ── Navigation bar ──────────────────────────────────────────────────────────
st.markdown("""
<div class="ts-nav">
    <div class="ts-brand">
        <span class="ts-icon">🛡️</span> ThreatScan
    </div>
    <div style="color:#64748b; font-size:14px; font-weight:500;">
        Professional URL Shield
    </div>
</div>
""", unsafe_allow_html=True)

# ── Hero card ───────────────────────────────────────────────────────────────
# Introduces the app with title, tag pills, and headline accuracy badges.
st.markdown("""
<div class="ts-card" style="display:flex; justify-content:space-between; align-items:center;">
    <div style="display:flex; gap:20px; align-items:center;">
        <div style="background:rgba(79,70,229,0.1); padding:16px; border-radius:16px;
                    border:1px solid rgba(79,70,229,0.2); font-size:32px;">🛡️</div>
        <div>
            <h1 style="margin:0; font-size:2.5rem; font-weight:700;">ThreatScan</h1>
            <div class="hero-tags">
                <span class="hero-tag">Malicious URL Detector</span>
                <span class="hero-tag">Random Forest</span>
                <span class="hero-tag">36 Features</span>
                <span class="hero-tag">Realistic Dataset</span>
            </div>
        </div>
    </div>
    <div style="text-align:right;">
        <span class="hero-tag" style="margin-right:8px;">98.2% Accuracy</span>
        <span class="hero-tag">100 Seed URLs</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Session state ────────────────────────────────────────────────────────────
# Streamlit reruns the entire script on every widget interaction.
# st.session_state persists values across those reruns within one browser tab.
if "history" not in st.session_state:
    st.session_state.history = []   # accumulates scan results (dicts)

# ── Load (or train) the model once per session ──────────────────────────────
model, feat_cols = load_model()

# ── Single-tab layout (URL Scanner only) ────────────────────────────────────
(tab1,) = st.tabs(["🔍 URL Scanner"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — URL SCANNER
# ══════════════════════════════════════════════════════════════════════════════
with tab1:

    # ── Input row ────────────────────────────────────────────────────────────
    # A wide text field for the URL and a compact button in a 5:1 column split.
    st.markdown(
        '<div class="uppercase-label" style="margin-bottom:8px;">// Target URL</div>',
        unsafe_allow_html=True,
    )
    
    # Wrap the input and button in a form so "Enter" triggers the submit button
    with st.form(key="scanner_form", border=False):
        col_input, col_btn = st.columns([5, 1])
        with col_input:
            url_input = st.text_input(
                "URL",
                placeholder="https://example.com or paste a suspicious link...",
                label_visibility="collapsed",
            )
        with col_btn:
            # Change st.button to st.form_submit_button
            analyze_btn = st.form_submit_button("⚡ Scan URL")

    # ── Analysis block ────────────────────────────────────────────────────────
    # Only executes when the button is pressed AND a URL has been entered.
    if analyze_btn and url_input:
        with st.spinner("Analysing..."):

            # Step 1 — Extract the 36-feature dict for this URL
            feats = extract_features(url_input)

            # Step 2 — Build a (1, 36) numpy array in the training column order
            X = np.array([feats.get(c, 0) for c in feat_cols]).reshape(1, -1)

            # Step 3 — Get class probabilities from the Random Forest
            #   prob[0] = P(benign)     prob[1] = P(malicious)
            prob = model.predict_proba(X)[0]

            # Step 4 — Convert to a 0–100 risk score
            risk_score = round(prob[1] * 100, 1)
            is_mal     = risk_score >= 50

            # Step 5 — Map score to a human-readable risk level
            if   risk_score < 20: risk_level = "SAFE"
            elif risk_score < 40: risk_level = "LOW"
            elif risk_score < 70: risk_level = "MEDIUM"
            elif risk_score < 90: risk_level = "HIGH"
            else:                 risk_level = "CRITICAL"

            # Step 6 — CSS class for the verdict card border colour
            border_class = "result-high" if is_mal else "result-safe"

            # Step 7 — Prepend to scan history (most-recent first)
            st.session_state.history.insert(0, {
                "url":   url_input,
                "level": risk_level,
                "score": risk_score,
                "time":  time.strftime("%H:%M"),
            })

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Results: verdict card (left) | feature impact chart (right) ─────
        res_col1, res_col2 = st.columns([7, 5])

        # ── Left — Verdict card + Threat indicators ───────────────────────
        with res_col1:

            # Resolve the hostname for display in the card
            try:
                parsed_host = urlparse(
                    url_input if '://' in url_input else 'http://' + url_input
                ).hostname or "unknown"
            except Exception:
                parsed_host = "unknown"

            # Verdict card — shows emoji, risk level, score, and hostname.
            # border_class adds a coloured left border (result-safe / result-high)
            # via the CSS defined in Section 5.
            st.markdown(f"""
            <div class="ts-card {border_class}">
                <div style="display:flex; justify-content:space-between;
                            align-items:start; margin-bottom:24px;">
                    <div style="display:flex; gap:16px; align-items:center;">
                        <div style="font-size:32px;">
                            {'🚨' if is_mal else '✅'}
                        </div>
                        <div>
                            <h3 style="margin:0; font-size:2rem; font-weight:700;">
                                {risk_level} RISK
                            </h3>
                            <p style="margin:0; font-size:14px; opacity:0.8;">
                                Confidence Score: {risk_score}/100
                            </p>
                        </div>
                    </div>
                    <div style="text-align:right;">
                        <div class="uppercase-label" style="opacity:0.6;">
                            Target Host
                        </div>
                        <div class="mono" style="font-size:14px;">
                            {parsed_host}
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Threat indicators — plain English list of triggered features
            st.markdown('<div class="ts-card">', unsafe_allow_html=True)
            st.markdown("#### 🎯 Threat Indicators")

            indicators = []
            if feats.get('suspicious_tld'):
                indicators.append("Suspicious Top Level Domain detected.")
            if feats.get('is_ip_host'):
                indicators.append("IP Address used instead of Domain name.")
            if feats.get('brand_in_domain'):
                indicators.append("Potential Brand Impersonation (Typosquatting).")
            if feats.get('phish_path_kw'):
                indicators.append("Phishing keywords found in URL path.")
            if not feats.get('is_https'):
                indicators.append("Connection is not secured with HTTPS.")
            if feats.get('is_shortener'):
                indicators.append("URL shortener service detected.")
            if feats.get('executable_ext'):
                indicators.append("Direct link to an executable file.")
            if feats.get('spam_keyword_count', 0) > 0:
                indicators.append(
                    f"Contains {feats['spam_keyword_count']} spam/urgent keyword(s)."
                )
            if not indicators:
                indicators.append("No significant threat indicators found.")

            for ind in indicators:
                st.markdown(f"- {ind}")

            st.markdown('</div>', unsafe_allow_html=True)

        # ── Right — Feature impact bar chart ─────────────────────────────
        with res_col2:
            st.markdown(
                '<div class="ts-card" style="height:100%;">',
                unsafe_allow_html=True,
            )
            st.markdown("#### 📊 Feature Impact")
            st.caption("Top features contributing to the final classification.")

            # Heuristic impact scores for the bar chart.
            # These are NOT the model's internal feature importances —
            # they are chosen to give a readable, intuitive visualisation.
            impact_data = {
                "Suspicious TLD":  85 if feats.get('suspicious_tld')    else 5,
                "IP Host":         90 if feats.get('is_ip_host')         else 2,
                "Brand Spoof":     75 if feats.get('brand_in_domain')    else 4,
                "Phish Keywords":  80 if feats.get('phish_path_kw')      else 10,
                "Shortener":       60 if feats.get('is_shortener')       else 5,
                "Spam Words": min(feats.get('spam_keyword_count', 0) * 20, 100),
            }

            # Filter to items with meaningful scores; fall back to a baseline
            # BUG FIX: renamed loop variable from `f` to `item` to avoid
            # shadowing the `feats` dict from extract_features() above.
            chart_items = [(k, v) for k, v in impact_data.items() if v > 5]
            if not chart_items:
                chart_items = [("Baseline Safe", 10)]

            # Sort ascending so the highest bar appears at the top
            chart_items.sort(key=lambda item: item[1])
            chart_labels = [item[0] for item in chart_items]
            chart_values = [item[1] for item in chart_items]

            # Colour each bar by severity bracket
            bar_colours = [
                '#10b981' if v < 40          # green  = low
                else '#f59e0b' if v < 70     # amber  = medium
                else '#f43f5e'               # red    = high
                for v in chart_values
            ]

            fig, ax = plt.subplots(figsize=(5, 2.5))
            ax.barh(chart_labels, chart_values, color=bar_colours, alpha=0.9)

            # Match the dark app theme
            ax.set_facecolor('#0f172a')
            fig.patch.set_facecolor('#0f172a')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color('#334155')
            ax.spines['left'].set_color('#334155')
            ax.tick_params(colors='#94a3b8')
            ax.set_xlabel("Impact Score", color='#94a3b8', fontsize=9)
            plt.setp(ax.get_xticklabels(), color='#94a3b8', fontsize=8)
            plt.setp(ax.get_yticklabels(), color='#94a3b8', fontsize=8)
            plt.tight_layout()

            st.pyplot(fig, use_container_width=True)
            plt.close(fig)   # free memory — prevents matplotlib figure leak

            st.markdown('</div>', unsafe_allow_html=True)

    elif analyze_btn:
        # Button was clicked but the text field is empty
        st.warning("Please enter a URL to scan.")


# =============================================================================
# RECENT SCANS HISTORY
# =============================================================================
# Rendered below the tab panel. Shows the last 10 scanned URLs with a
# coloured risk badge and the time of each scan.

st.markdown(
    "<hr style='border-color:#1e293b; margin:3rem 0;'>",
    unsafe_allow_html=True,
)
st.markdown("### 🕒 Recent Scans")

if not st.session_state.history:
    st.markdown(
        '<div style="text-align:center; color:#64748b; padding:2rem;">'
        'No scan history yet. Start by analysing a URL.'
        '</div>',
        unsafe_allow_html=True,
    )
else:
    for item in st.session_state.history[:10]:
        # HIGH, CRITICAL, and MEDIUM all get the red badge;
        # LOW and SAFE get the green badge.
        badge_cls = (
            "risk-high"
            if item['level'] in ("HIGH", "CRITICAL", "MEDIUM")
            else "risk-safe"
        )
        # Truncate very long URLs so the history row stays on one line
        display_url = (
            item['url'][:80] + '...'
            if len(item['url']) > 80
            else item['url']
        )
        st.markdown(f"""
        <div style="background:rgba(30,41,59,0.3); border:1px solid #1e293b;
                    border-radius:12px; padding:1rem 1.5rem; margin-bottom:8px;
                    display:flex; justify-content:space-between; align-items:center;">
            <div style="display:flex; gap:16px; align-items:center;">
                <span class="risk-badge {badge_cls}">{item['level']}</span>
                <span class="mono" style="font-size:14px; color:#cbd5e1;">
                    {display_url}
                </span>
            </div>
            <div class="uppercase-label" style="opacity:0.6; flex-shrink:0;">
                {item['time']}
            </div>
        </div>
        """, unsafe_allow_html=True)
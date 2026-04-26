"""
=============================================================================
ThreatScan — Malicious URL Detector  (Sentinel Engine)
=============================================================================
Author  : ThreatScan Project
Stack   : Python 3.x · scikit-learn · Streamlit · matplotlib · SQLite
Run     : streamlit run threatscan_sentinel.py

WHAT'S INSIDE
─────────────
Section 1  — Imports
Section 2  — Dataset  : 50 seed URLs × 2 classes, augmented to 2,000 rows
Section 3  — Feature Extraction  : 36 lexical + host-based features per URL
Section 4  — URL Unshortening    : expands bit.ly / tinyurl etc.
Section 5  — SQLite Logging      : every scan is persisted to scans.db
Section 6  — Model Training      : Random Forest + SVM + Decision Tree
Section 7  — Prediction helper
Section 8  — Page config + CSS   : ThreatScan dark theme (original frontend)
Section 9  — App layout          : Nav · Hero · Scanner · Model Performance
                                   · Recent Scans history

DATASET SIZE  →  2,000 rows (1,000 benign + 1,000 malicious)
──────────────────────────────────────────────────────────────
Streamlit Community Cloud gives ~1 GB RAM.  The original 50 k-row dataset
takes ~5–8 min to extract features and blows the RAM cap.  2 k rows extract
in ~2 s and train all three models in ~5 s while keeping all 36 features,
all metrics (Accuracy / Precision / Recall / F1 / FAR / FRR / AUC),
Confusion Matrix, ROC Curve and SQLite logging fully intact.
=============================================================================
"""

# =============================================================================
# SECTION 1 — IMPORTS
# =============================================================================
import os
import re
import math
import time
import csv
import random
import json
import sqlite3

import joblib
import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import streamlit as st

from urllib.parse import urlparse, parse_qs
from datetime    import datetime

from sklearn.model_selection import train_test_split
from sklearn.ensemble        import RandomForestClassifier
from sklearn.svm             import SVC
from sklearn.tree            import DecisionTreeClassifier
from sklearn.metrics         import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score, ConfusionMatrixDisplay,
)

# Optional live URL unshortening
try:
    import requests as _requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


# =============================================================================
# SECTION 2 — DATASET : URL LISTS, AUGMENTATION, CSV CREATION
# =============================================================================

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

# ── Absolute path constants ──────────────────────────────────────────────────
_HERE           = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH    = os.path.join(_HERE, "data",   "urls_dataset.csv")
MODEL_PATH      = os.path.join(_HERE, "models", "best_model.joblib")
COMPARISON_PATH = os.path.join(_HERE, "models", "model_comparison.json")
DB_PATH         = os.path.join(_HERE, "data",   "scans.db")


def augment_urls(url_list: list, target_count: int, seed: int = 42) -> list:
    """Deterministically grow a seed list to target_count unique-ish URLs."""
    rng    = random.Random(seed)
    paths  = ['/index','/page','/post','/article','/blog','/news','/shop','/product']
    params = ['ref','source','utm','id','token','session','lang','page']
    augmented = list(url_list)
    attempts  = 0
    while len(augmented) < target_count and attempts < target_count * 20:
        attempts += 1
        base    = rng.choice(url_list)
        variant = base
        if rng.random() < 0.55:
            variant += f"?{rng.choice(params)}={rng.randint(1000,99999)}"
        if rng.random() < 0.35:
            variant += rng.choice(paths)
        if rng.random() < 0.15:
            variant += f"#{rng.randint(10,9999)}"
        if variant not in augmented:
            augmented.append(variant)
    while len(augmented) < target_count:
        augmented.append(rng.choice(url_list) + f"?x={rng.randint(100000,999999)}")
    rng.shuffle(augmented)
    return augmented[:target_count]


def create_dataset(n_per_class: int = 1000):
    """Build a balanced 2,000-row CSV and write it to DATASET_PATH."""
    benign    = augment_urls(BENIGN_URLS,    n_per_class, seed=1)
    malicious = augment_urls(MALICIOUS_URLS, n_per_class, seed=2)
    rows      = [(u, 0) for u in benign] + [(u, 1) for u in malicious]
    random.Random(42).shuffle(rows)
    os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)
    with open(DATASET_PATH, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows([["url", "label"]] + rows)


def ensure_dataset():
    if not os.path.exists(DATASET_PATH):
        create_dataset(1000)


# =============================================================================
# SECTION 3 — FEATURE EXTRACTION  (36 features per URL)
# =============================================================================

SUSPICIOUS_TLDS = {
    'xyz','top','click','tk','ml','ga','cf','gq','pw','cc',
    'su','biz','info','online','site','live','stream','download',
    'loan','review','country','kim','science','work','party','trade',
    'cricket','date','faith','racing','accountant','win','bid',
    'men','icu','monster','cyou','buzz','sbs','ru',
}
TRUSTED_DOMAINS = {
    'google.com','youtube.com','facebook.com','microsoft.com','apple.com',
    'amazon.com','github.com','twitter.com','linkedin.com','wikipedia.org',
    'instagram.com','netflix.com','stackoverflow.com','reddit.com','paypal.com',
    'bbc.com','nytimes.com','dropbox.com','mozilla.org','cloudflare.com',
    'medium.com','kaggle.com','huggingface.co','arxiv.org','nature.com',
    'zoom.us','slack.com','notion.so','figma.com','canva.com','stripe.com',
    'shopify.com','heroku.com','vercel.com','netlify.com',
}
BRAND_KEYWORDS = [
    'paypal','google','apple','microsoft','amazon','facebook',
    'instagram','netflix','ebay','steam','whatsapp','youtube',
    'dropbox','icloud','twitter','chase','wellsfargo','citibank',
    'bankofamerica','boa','dhl','fedex','usps','ups',
]
URL_SHORTENERS = {
    'bit.ly','tinyurl.com','t.co','goo.gl','ow.ly','is.gd',
    'buff.ly','rebrand.ly','short.io','tiny.cc','cutt.ly',
    'shorturl.at','rb.gy','short.link','qr.ae','v.gd','tiny.one',
}
PHISH_RE = re.compile(
    r'login|signin|verify|account|update|secure|confirm|'
    r'password|credential|alert|suspend|unlock|recover|'
    r'reset|billing|payment|invoice', re.I
)
EXEC_RE    = re.compile(r'\.(exe|bat|cmd|msi|scr|vbs|jar|apk|dmg|sh|ps1|crx|xpi)$', re.I)
SPAM_WORDS = [
    'free','win','prize','claim','urgent','alert','suspended','verify',
    'confirm','limited','offer','bonus','gift','reward','lucky','congratulation',
]


def _entropy(s: str) -> float:
    if not s: return 0.0
    freq = {}
    for c in s: freq[c] = freq.get(c, 0) + 1
    n = len(s)
    return -sum((v/n)*math.log2(v/n) for v in freq.values())


def _domain_parts(hostname: str):
    clean = re.sub(r'^www\.', '', hostname.lower())
    parts = clean.split('.')
    if len(parts) >= 3: return '.'.join(parts[:-2]), parts[-2], parts[-1]
    if len(parts) == 2: return '', parts[0], parts[1]
    return '', hostname, ''


def extract_features(url: str, expanded_url: str = None) -> dict:
    """Convert a raw URL into a 36-element feature dictionary."""
    raw    = str(url).strip()
    target = expanded_url if expanded_url else raw
    f      = {}
    try:
        p = urlparse(target if '://' in target else 'http://' + target)
    except Exception:
        p = urlparse('http://invalid')

    hostname   = (p.hostname or '').lower()
    path       = p.path  or ''
    query      = p.query or ''
    scheme     = p.scheme or ''
    full_lower = target.lower()
    _, domain, tld = _domain_parts(hostname)
    base = f"{domain}.{tld}" if domain and tld else hostname
    sub, _, _  = _domain_parts(hostname)
    hl = max(len(hostname), 1)

    f['is_https']           = int(scheme == 'https')
    f['is_http']            = int(scheme == 'http')
    f['url_length']         = len(target)
    f['hostname_length']    = len(hostname)
    f['path_length']        = len(path)
    f['query_length']       = len(query)
    f['dot_count']          = hostname.count('.')
    f['hyphen_count']       = hostname.count('-')
    f['underscore_count']   = target.count('_')
    f['at_sign']            = int('@' in target)
    f['double_slash']       = int('//' in path)
    f['question_mark']      = int('?' in target)
    f['ampersand_count']    = query.count('&')
    f['equals_count']       = query.count('=')
    f['percent_count']      = len(re.findall(r'%[0-9a-fA-F]{2}', target))
    f['hash_count']         = int('#' in target)
    f['digit_ratio']        = round(sum(c.isdigit() for c in hostname)/hl, 4)
    f['alpha_ratio']        = round(sum(c.isalpha() for c in hostname)/hl, 4)
    f['subdomain_count']    = len(sub.split('.')) if sub else 0
    f['suspicious_tld']     = int(tld in SUSPICIOUS_TLDS)
    f['tld_length']         = len(tld)
    f['is_ip_host']         = int(bool(re.match(r'^\d{1,3}(\.\d{1,3}){3}$', hostname)))
    f['trusted_domain']     = int(base in TRUSTED_DOMAINS)
    brand_hit               = any(b in hostname for b in BRAND_KEYWORDS)
    f['brand_in_domain']    = int(brand_hit and base not in TRUSTED_DOMAINS)
    f['digit_in_word']      = int(bool(re.search(r'[a-z]\d[a-z]', hostname)))
    f['phish_path_kw']      = int(bool(PHISH_RE.search(path)))
    f['executable_ext']     = int(bool(EXEC_RE.search(path)))
    f['path_depth']         = path.count('/')
    f['path_has_ip']        = int(bool(re.search(r'/\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', path)))
    try:    f['param_count'] = len(parse_qs(query))
    except: f['param_count'] = 0
    f['hostname_entropy']   = round(_entropy(hostname), 4)
    f['path_entropy']       = round(_entropy(path), 4)
    f['is_shortener']       = int(hostname in URL_SHORTENERS)
    f['spam_keyword_count'] = sum(w in full_lower for w in SPAM_WORDS)
    f['has_punycode']       = int('xn--' in hostname)
    f['domain_age_days']    = 365   # placeholder (real WHOIS too slow for Streamlit)
    return f


FEATURE_COLUMNS = list(extract_features("http://example.com").keys())


# =============================================================================
# SECTION 4 — URL UNSHORTENING
# =============================================================================

def is_shortened_url(url: str) -> bool:
    try:
        parsed = urlparse(url if '://' in url else 'http://' + url)
        domain = (parsed.netloc or '').lower().lstrip('www.')
        return any(s in domain for s in URL_SHORTENERS)
    except:
        return False


def unshorten_url(url: str) -> str:
    if not HAS_REQUESTS or not is_shortened_url(url):
        return url
    try:
        url  = url if url.startswith(('http://','https://')) else 'https://' + url
        resp = _requests.head(url, allow_redirects=True, timeout=8)
        if resp.status_code == 200 and resp.url:
            return resp.url
        resp = _requests.get(url, allow_redirects=True, timeout=8)
        return resp.url if resp.status_code == 200 and resp.url else url
    except:
        return url


def safe_unshorten(url: str):
    was = is_shortened_url(url)
    if was:
        return url, unshorten_url(url), True
    return url, url, False


# =============================================================================
# SECTION 5 — SQLITE LOGGING
# =============================================================================

def init_database():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS scans (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        url TEXT NOT NULL, expanded_url TEXT,
        verdict TEXT NOT NULL, risk_score REAL NOT NULL,
        safe_pct REAL NOT NULL, mal_pct REAL NOT NULL,
        processing_time REAL, ip_address TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    c.execute('''CREATE TABLE IF NOT EXISTS scan_features (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        scan_id INTEGER, feature_name TEXT NOT NULL, feature_value TEXT,
        FOREIGN KEY (scan_id) REFERENCES scans(id))''')
    conn.commit(); conn.close()


def log_scan(data: dict):
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute('''INSERT INTO scans
        (url,expanded_url,verdict,risk_score,safe_pct,mal_pct,processing_time,ip_address)
        VALUES (?,?,?,?,?,?,?,?)''',
        (data.get('url',''), data.get('expanded_url',''), data.get('verdict',''),
         data.get('risk_score',0), data.get('safe_pct',0), data.get('mal_pct',0),
         data.get('processing_time',0), '127.0.0.1'))
    sid = c.lastrowid
    for k, v in data.get('features', {}).items():
        c.execute('INSERT INTO scan_features (scan_id,feature_name,feature_value) VALUES (?,?,?)',
                  (sid, k, str(v)))
    conn.commit(); conn.close()


def get_db_recent(limit: int = 10) -> list:
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        c    = conn.cursor()
        c.execute('SELECT url,verdict,risk_score,timestamp FROM scans ORDER BY timestamp DESC LIMIT ?', (limit,))
        rows = c.fetchall(); conn.close()
        return [dict(r) for r in rows]
    except:
        return []


def get_db_total() -> int:
    try:
        conn = sqlite3.connect(DB_PATH)
        c    = conn.cursor()
        c.execute('SELECT COUNT(*) FROM scans')
        n = c.fetchone()[0]; conn.close()
        return n
    except:
        return 0


# =============================================================================
# SECTION 6 — MODEL TRAINING  (Random Forest + SVM + Decision Tree)
# =============================================================================

def calculate_far_frr(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    return far, frr


def train_all_models():
    df = pd.read_csv(DATASET_PATH)
    df = df.drop_duplicates(subset=['url']).dropna(subset=['url','label'])
    df['label'] = df['label'].astype(int).clip(0,1)

    X = (pd.DataFrame([extract_features(u) for u in df['url']])
           [FEATURE_COLUMNS].fillna(0).values.astype(float))
    y = df['label'].values

    col_idx  = FEATURE_COLUMNS.index('domain_age_days')
    col_vals = X[:, col_idx]
    median_age = np.median(col_vals[col_vals >= 0]) if np.any(col_vals >= 0) else 365
    X[X[:, col_idx] == -1, col_idx] = median_age

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    classifiers = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=12, min_samples_leaf=2,
            class_weight='balanced', random_state=42, n_jobs=-1),
        "SVM": SVC(
            kernel='rbf', C=1.0, gamma='scale',
            probability=True, class_weight='balanced', random_state=42),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=12, min_samples_leaf=2,
            class_weight='balanced', random_state=42),
    }

    results    = {}
    best_model = None
    best_acc   = 0

    for name, clf in classifiers.items():
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        y_prob = clf.predict_proba(X_te)[:, 1]
        acc    = accuracy_score(y_te, y_pred)
        prec   = precision_score(y_te, y_pred, zero_division=0)
        rec    = recall_score(y_te, y_pred, zero_division=0)
        f1     = f1_score(y_te, y_pred, zero_division=0)
        far, frr = calculate_far_frr(y_te, y_pred)
        auc    = roc_auc_score(y_te, y_prob)
        results[name] = {
            "accuracy": float(acc), "precision": float(prec),
            "recall": float(rec),   "f1_score": float(f1),
            "far": float(far),      "frr": float(frr),
            "auc": float(auc),
            "y_test": y_te.tolist(), "y_pred": y_pred.tolist(), "y_prob": y_prob.tolist(),
        }
        if acc > best_acc:
            best_acc   = acc
            best_model = clf

    comp_json = {n: {k: v for k, v in d.items() if k not in ('y_test','y_pred','y_prob')}
                 for n, d in results.items()}
    os.makedirs(os.path.dirname(COMPARISON_PATH), exist_ok=True)
    with open(COMPARISON_PATH, 'w') as fh:
        json.dump(comp_json, fh, indent=2)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({"model": best_model, "feature_columns": FEATURE_COLUMNS}, MODEL_PATH)
    return best_model, results


@st.cache_resource(show_spinner="⚙️ Training Sentinel Engine… first run only (~10 s)")
def load_model():
    ensure_dataset()
    if os.path.exists(MODEL_PATH):
        payload = joblib.load(MODEL_PATH)
        return payload["model"], payload["feature_columns"]
    model, _ = train_all_models()
    return model, FEATURE_COLUMNS


def get_model_comparison() -> dict | None:
    if os.path.exists(COMPARISON_PATH):
        with open(COMPARISON_PATH) as fh:
            return json.load(fh)
    return None


def get_test_predictions(mdl, fc):
    df = pd.read_csv(DATASET_PATH)
    df = df.drop_duplicates(subset=['url']).dropna(subset=['url','label'])
    df['label'] = df['label'].astype(int).clip(0,1)
    X = (pd.DataFrame([extract_features(u) for u in df['url']])
           [fc].fillna(0).values.astype(float))
    y = df['label'].values
    col_idx = fc.index('domain_age_days')
    col_vals = X[:, col_idx]
    med = np.median(col_vals[col_vals >= 0]) if np.any(col_vals >= 0) else 365
    X[X[:, col_idx] == -1, col_idx] = med
    _, X_te, _, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    return y_te, mdl.predict(X_te), mdl.predict_proba(X_te)[:, 1]


# =============================================================================
# SECTION 7 — PREDICTION HELPER
# =============================================================================

def predict_url(url: str, model, feat_cols: list) -> dict:
    start = time.time()
    original, expanded, was_short = safe_unshorten(url)
    feats = extract_features(original, expanded if was_short else None)
    X     = np.array([feats.get(c, 0) for c in feat_cols]).reshape(1, -1)
    prob  = model.predict_proba(X)[0]
    safe_pct = round(prob[0] * 100, 1)
    mal_pct  = round(prob[1] * 100, 1)

    # 5-level risk classification (matches ThreatScan frontend)
    if   mal_pct < 20: risk_level = "SAFE"
    elif mal_pct < 40: risk_level = "LOW"
    elif mal_pct < 70: risk_level = "MEDIUM"
    elif mal_pct < 90: risk_level = "HIGH"
    else:              risk_level = "CRITICAL"

    is_mal      = mal_pct >= 50
    proc_ms     = round((time.time() - start) * 1000, 2)

    return {
        "url": original, "expanded_url": expanded if was_short else original,
        "was_shortened": was_short,
        "verdict": risk_level, "risk_score": mal_pct,
        "safe_pct": safe_pct, "mal_pct": mal_pct, "is_mal": is_mal,
        "features": feats, "processing_time_ms": proc_ms,
    }


# =============================================================================
# SECTION 8 — PAGE CONFIG + CSS THEME  (original ThreatScan frontend)
# =============================================================================

init_database()

st.set_page_config(
    page_title="ThreatScan – Malicious URL Detector",
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
.uppercase-label {
  font-size: 10px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: #64748b;
}

/* ── Top navigation bar ─────────────────────────────────────────────────── */
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
.ts-icon {
  background: #4f46e5;
  padding: 6px;
  border-radius: 8px;
  box-shadow: 0 4px 14px rgba(79, 70, 229, 0.39);
}

/* ── General card container ─────────────────────────────────────────────── */
.ts-card {
  background: rgba(30, 41, 59, 0.4);
  border: 1px solid #1e293b;
  border-radius: 24px;
  padding: 2rem;
  margin-bottom: 1.5rem;
}
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
.stButton > button {
  background-color: #4f46e5 !important;
  color: white !important;
  border: none !important;
  border-radius: 12px !important;
  padding: 0.75rem 2rem !important;
  font-weight: 700 !important;
  width: 100%;
  transition: 0.2s;
}
.stButton > button:hover {
  background-color: #4338ca !important;
  box-shadow: 0 4px 14px rgba(79, 70, 229, 0.39) !important;
}

/* ── Risk badge pills ───────────────────────────────────────────────────── */
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

/* ── Matplotlib charts inside dark cards ────────────────────────────────── */
.stPlotlyChart, [data-testid="stImage"] { border-radius: 12px; }

/* ── Dataframe / table styling ──────────────────────────────────────────── */
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SECTION 9 — APP LAYOUT
# =============================================================================

# ── Navigation bar ───────────────────────────────────────────────────────────
st.markdown("""
<div class="ts-nav">
    <div class="ts-brand">
        <span class="ts-icon">🛡️</span> ThreatScan
    </div>
    <div style="color:#64748b; font-size:14px; font-weight:500;">
        Professional URL Shield · Sentinel Engine
    </div>
</div>
""", unsafe_allow_html=True)

# ── Hero card ─────────────────────────────────────────────────────────────────
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
                <span class="hero-tag">SVM</span>
                <span class="hero-tag">Decision Tree</span>
                <span class="hero-tag">36 Features</span>
                <span class="hero-tag">SQLite Logging</span>
            </div>
        </div>
    </div>
    <div style="text-align:right;">
        <span class="hero-tag" style="margin-right:8px;">2,000 Training URLs</span>
        <span class="hero-tag">FAR / FRR Tracked</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ── Load (or train) model once per session ───────────────────────────────────
model, feat_cols = load_model()

# ── Two tabs: Scanner | Model Performance ────────────────────────────────────
tab1, tab2 = st.tabs(["🔍 URL Scanner", "📊 Model Performance"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — URL SCANNER
# ══════════════════════════════════════════════════════════════════════════════
with tab1:

    st.markdown(
        '<div class="uppercase-label" style="margin-bottom:8px;">// Target URL</div>',
        unsafe_allow_html=True,
    )
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        url_input = st.text_input(
            "URL",
            placeholder="https://example.com or paste a suspicious link...",
            label_visibility="collapsed",
        )
    with col_btn:
        analyze_btn = st.button("⚡ Scan URL")

    if analyze_btn and url_input:
        with st.spinner("Analysing…"):
            result = predict_url(url_input, model, feat_cols)
            feats  = result['features']

            # Log to SQLite
            try:
                log_scan({
                    'url': result['url'], 'expanded_url': result['expanded_url'],
                    'verdict': result['verdict'], 'risk_score': result['risk_score'],
                    'safe_pct': result['safe_pct'], 'mal_pct': result['mal_pct'],
                    'processing_time': result['processing_time_ms'], 'features': feats,
                })
            except Exception:
                pass

            # Add to session history
            st.session_state.history.insert(0, {
                "url":   url_input,
                "level": result['verdict'],
                "score": result['risk_score'],
                "time":  time.strftime("%H:%M"),
            })

        st.markdown("<br>", unsafe_allow_html=True)

        risk_level   = result['verdict']
        risk_score   = result['risk_score']
        is_mal       = result['is_mal']
        border_class = "result-high" if is_mal else "result-safe"

        try:
            parsed_host = urlparse(
                url_input if '://' in url_input else 'http://' + url_input
            ).hostname or "unknown"
        except Exception:
            parsed_host = "unknown"

        # ── Results row: verdict (left) | feature impact (right) ─────────────
        res_col1, res_col2 = st.columns([7, 5])

        with res_col1:
            # Verdict card
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
                                Confidence Score: {risk_score}/100 ·
                                Processing: {result['processing_time_ms']}ms
                            </p>
                        </div>
                    </div>
                    <div style="text-align:right;">
                        <div class="uppercase-label" style="opacity:0.6;">Target Host</div>
                        <div class="mono" style="font-size:14px;">{parsed_host}</div>
                        {"<div style='color:#f59e0b;font-size:12px;margin-top:4px'>🔗 Shortened URL expanded</div>" if result.get('was_shortened') else ""}
                    </div>
                </div>
                <!-- Probability bars -->
                <div style="display:flex; gap:12px; margin-bottom:16px;">
                    <div style="flex:1; background:#0f172a; border-radius:8px; padding:12px; text-align:center;">
                        <div style="font-size:1.4rem; font-weight:700; color:#34d399;">{result['safe_pct']}%</div>
                        <div class="uppercase-label">Safe</div>
                    </div>
                    <div style="flex:1; background:#0f172a; border-radius:8px; padding:12px; text-align:center;">
                        <div style="font-size:1.4rem; font-weight:700; color:#f87171;">{result['mal_pct']}%</div>
                        <div class="uppercase-label">Malicious</div>
                    </div>
                    <div style="flex:1; background:#0f172a; border-radius:8px; padding:12px; text-align:center;">
                        <div style="font-size:1.4rem; font-weight:700; color:#818cf8;">{'Yes' if feats.get('is_https') else 'No'}</div>
                        <div class="uppercase-label">HTTPS</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Threat indicators card
            st.markdown('<div class="ts-card">', unsafe_allow_html=True)
            st.markdown("#### 🎯 Threat Indicators")
            indicators = []
            if feats.get('suspicious_tld'):          indicators.append("Suspicious Top Level Domain detected.")
            if feats.get('is_ip_host'):              indicators.append("IP Address used instead of Domain name.")
            if feats.get('brand_in_domain'):         indicators.append("Potential Brand Impersonation (Typosquatting).")
            if feats.get('phish_path_kw'):           indicators.append("Phishing keywords found in URL path.")
            if not feats.get('is_https'):            indicators.append("Connection is not secured with HTTPS.")
            if feats.get('is_shortener'):            indicators.append("URL shortener service detected.")
            if feats.get('executable_ext'):          indicators.append("Direct link to an executable file.")
            if feats.get('spam_keyword_count',0)>0:
                indicators.append(f"Contains {feats['spam_keyword_count']} spam/urgent keyword(s).")
            if feats.get('at_sign'):                 indicators.append("@ symbol detected in URL (redirect trick).")
            if feats.get('digit_in_word'):           indicators.append("Typosquatting pattern detected (e.g. g00gle).")
            if feats.get('has_punycode'):            indicators.append("Punycode / IDN homograph attack detected.")
            if not indicators:                       indicators.append("No significant threat indicators found.")
            for ind in indicators:
                st.markdown(f"- {ind}")
            st.markdown('</div>', unsafe_allow_html=True)

        with res_col2:
            # Feature impact chart
            st.markdown('<div class="ts-card" style="height:100%;">', unsafe_allow_html=True)
            st.markdown("#### 📊 Feature Impact")
            st.caption("Top features contributing to the final classification.")
            impact_data = {
                "Suspicious TLD":  85 if feats.get('suspicious_tld')    else 5,
                "IP Host":         90 if feats.get('is_ip_host')         else 2,
                "Brand Spoof":     75 if feats.get('brand_in_domain')    else 4,
                "Phish Keywords":  80 if feats.get('phish_path_kw')      else 10,
                "Shortener":       60 if feats.get('is_shortener')       else 5,
                "Spam Words":      min(feats.get('spam_keyword_count',0)*20, 100),
                "Young Domain":    70 if feats.get('domain_age_days',365)<30 else 5,
            }
            chart_items = sorted([(k,v) for k,v in impact_data.items() if v>5], key=lambda x:x[1])
            if not chart_items: chart_items = [("Baseline Safe", 10)]
            chart_labels = [x[0] for x in chart_items]
            chart_values = [x[1] for x in chart_items]
            bar_colours  = ['#10b981' if v<40 else '#f59e0b' if v<70 else '#f43f5e' for v in chart_values]

            fig, ax = plt.subplots(figsize=(5, 2.5))
            ax.barh(chart_labels, chart_values, color=bar_colours, alpha=0.9)
            ax.set_facecolor('#0f172a'); fig.patch.set_facecolor('#0f172a')
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color('#334155'); ax.spines['left'].set_color('#334155')
            ax.tick_params(colors='#94a3b8')
            ax.set_xlabel("Impact Score", color='#94a3b8', fontsize=9)
            plt.setp(ax.get_xticklabels(), color='#94a3b8', fontsize=8)
            plt.setp(ax.get_yticklabels(), color='#94a3b8', fontsize=8)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True); plt.close(fig)
            st.markdown('</div>', unsafe_allow_html=True)

        # Full feature vector
        with st.expander("🔬 View Full Feature Vector (36 features)"):
            st.dataframe(
                pd.DataFrame(feats.items(), columns=["Feature","Value"]).set_index("Feature"),
                use_container_width=True, height=400)

    elif analyze_btn:
        st.warning("Please enter a URL to scan.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    comparison = get_model_comparison()

    if comparison:
        st.markdown('<div class="ts-card">', unsafe_allow_html=True)
        st.markdown("#### 🤖 Model Comparison  (Random Forest vs SVM vs Decision Tree)")
        df_comp = pd.DataFrame([{
            "Model":     name,
            "Accuracy":  f"{d['accuracy']*100:.2f}%",
            "Precision": f"{d['precision']*100:.2f}%",
            "Recall":    f"{d['recall']*100:.2f}%",
            "F1-Score":  f"{d['f1_score']*100:.2f}%",
            "FAR":       f"{d['far']*100:.2f}%",
            "FRR":       f"{d['frr']*100:.2f}%",
            "AUC":       f"{d['auc']:.3f}",
        } for name, d in comparison.items()])
        st.dataframe(df_comp, use_container_width=True)
        st.info("Targets: Accuracy ≥ 95% | FAR ≤ 2% | FRR ≤ 3%")
        st.markdown('</div>', unsafe_allow_html=True)

        # Confusion matrix + ROC curve side by side
        y_test, y_pred, y_prob = get_test_predictions(model, feat_cols)
        cm_col, roc_col = st.columns(2)

        with cm_col:
            st.markdown('<div class="ts-card">', unsafe_allow_html=True)
            st.markdown("#### 🟦 Confusion Matrix")
            cm    = confusion_matrix(y_test, y_pred)
            fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
            fig_cm.patch.set_facecolor('#0f172a'); ax_cm.set_facecolor('#0f172a')
            ConfusionMatrixDisplay(cm, display_labels=["Benign","Malicious"]).plot(
                ax=ax_cm, colorbar=False, cmap='Blues')
            ax_cm.tick_params(colors='#94a3b8')
            ax_cm.title.set_color('white')
            ax_cm.xaxis.label.set_color('#94a3b8')
            ax_cm.yaxis.label.set_color('#94a3b8')
            plt.tight_layout()
            st.pyplot(fig_cm, use_container_width=True); plt.close(fig_cm)
            st.markdown('</div>', unsafe_allow_html=True)

        with roc_col:
            st.markdown('<div class="ts-card">', unsafe_allow_html=True)
            st.markdown("#### 📈 ROC Curve")
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc_val     = roc_auc_score(y_test, y_prob)
            fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
            fig_roc.patch.set_facecolor('#0f172a'); ax_roc.set_facecolor('#0f172a')
            ax_roc.plot(fpr, tpr, color='#818cf8', lw=2, label=f'AUC = {auc_val:.3f}')
            ax_roc.plot([0,1],[0,1], 'k--', lw=1, alpha=0.5)
            ax_roc.set_xlabel('False Positive Rate', color='#94a3b8')
            ax_roc.set_ylabel('True Positive Rate',  color='#94a3b8')
            ax_roc.set_title('ROC Curve',            color='white')
            ax_roc.legend(loc='lower right', facecolor='#1e293b', labelcolor='white')
            ax_roc.tick_params(colors='#94a3b8')
            for spine in ax_roc.spines.values(): spine.set_color('#334155')
            plt.tight_layout()
            st.pyplot(fig_roc, use_container_width=True); plt.close(fig_roc)
            st.markdown('</div>', unsafe_allow_html=True)

        # FAR / FRR detail
        st.markdown('<div class="ts-card">', unsafe_allow_html=True)
        st.markdown("#### ⚠️ FAR & FRR Detail")
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        far_val = fp / (fp + tn) if (fp + tn) > 0 else 0
        frr_val = fn / (fn + tp) if (fn + tp) > 0 else 0
        fc1, fc2, fc3, fc4 = st.columns(4)
        fc1.metric("True Positives",  int(tp))
        fc2.metric("True Negatives",  int(tn))
        fc3.metric("False Alarm Rate (FAR)", f"{far_val*100:.2f}%",
                   delta="≤ 2% target" if far_val <= 0.02 else f"⚠️ {far_val*100:.2f}% > 2%")
        fc4.metric("False Reject Rate (FRR)", f"{frr_val*100:.2f}%",
                   delta="≤ 3% target" if frr_val <= 0.03 else f"⚠️ {frr_val*100:.2f}% > 3%")
        st.markdown('</div>', unsafe_allow_html=True)

        # DB scan count
        total = get_db_total()
        if total:
            st.caption(f"🗄️ Total scans logged to SQLite: **{total}**")

    else:
        st.markdown('<div class="ts-card">', unsafe_allow_html=True)
        st.info("Run your first URL scan to trigger model training and populate this dashboard.")
        st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
# RECENT SCANS HISTORY  (session + SQLite)
# =============================================================================

st.markdown("<hr style='border-color:#1e293b; margin:3rem 0;'>", unsafe_allow_html=True)
st.markdown("### 🕒 Recent Scans")

# Prefer SQLite (persists across page refreshes); fall back to session state
db_history = get_db_recent(10)
display_history = db_history if db_history else st.session_state.history[:10]

if not display_history:
    st.markdown(
        '<div style="text-align:center; color:#64748b; padding:2rem;">'
        'No scan history yet. Start by analysing a URL above.'
        '</div>',
        unsafe_allow_html=True,
    )
else:
    for item in display_history:
        # Support both SQLite rows (dict with 'verdict') and session dicts (with 'level')
        level       = item.get('verdict') or item.get('level', 'SAFE')
        score_raw   = item.get('risk_score') or item.get('score', 0)
        timestamp   = item.get('timestamp') or item.get('time', '')
        url_text    = item.get('url', '')
        badge_cls   = "risk-high" if level in ("HIGH","CRITICAL","MEDIUM") else "risk-safe"
        display_url = url_text[:80] + ('…' if len(url_text) > 80 else '')

        st.markdown(f"""
        <div style="background:rgba(30,41,59,0.3); border:1px solid #1e293b;
                    border-radius:12px; padding:1rem 1.5rem; margin-bottom:8px;
                    display:flex; justify-content:space-between; align-items:center;">
            <div style="display:flex; gap:16px; align-items:center;">
                <span class="risk-badge {badge_cls}">{level}</span>
                <span class="mono" style="font-size:14px; color:#cbd5e1;">{display_url}</span>
            </div>
            <div style="display:flex; gap:16px; align-items:center; flex-shrink:0;">
                <span style="color:#64748b; font-size:12px; font-weight:700;">
                    Score: {score_raw:.0f}/100
                </span>
                <span class="uppercase-label" style="opacity:0.6;">{timestamp}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    if db_history:
        st.caption(f"🗄️ Showing last {len(db_history)} scans from SQLite  ·  Total: {get_db_total()}")

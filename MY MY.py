"""
=============================================================================
SENTINEL - Malicious URL Detector
=============================================================================
Single-file Streamlit application.
Includes:
- 10,000 URL dataset generation (Optimized for Streamlit Cloud RAM)
- 36 lexical + host-based features (with real WHOIS domain age)
- URL unshortening (expands bit.ly, t.co, etc.)
- Training: Random Forest, SVM, Decision Tree
- Metrics: Accuracy, Precision, Recall, F1, FAR, FRR, AUC
- Visualizations: Confusion Matrix, ROC Curve, Feature Impact
- SQLite database logging of all scans
- Sentinel dark UI with recent scans history
=============================================================================
"""

import os
import re
import math
import time
import csv
import random
import json
import sqlite3
import requests
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from urllib.parse import urlparse, parse_qs
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score, ConfusionMatrixDisplay
)

# =============================================================================
# 1. DATASET GENERATION
# =============================================================================

REALISTIC_BENIGN = [
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

REALISTIC_MALICIOUS = [
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

_HERE = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(_HERE, "data", "urls_dataset.csv")
MODEL_PATH = os.path.join(_HERE, "models", "best_model.joblib")
COMPARISON_PATH = os.path.join(_HERE, "models", "model_comparison.json")
DB_PATH = os.path.join(_HERE, "data", "scans.db")

def augment_urls(url_list, target_count):
    if len(url_list) >= target_count:
        return random.sample(url_list, target_count)
    augmented = list(url_list)
    paths = ['/index', '/page', '/post', '/article', '/blog', '/news', '/shop', '/product']
    while len(augmented) < target_count:
        base = random.choice(url_list)
        variant = base
        if random.random() < 0.5:
            variant += f"?{random.choice(['ref', 'source', 'utm', 'id'])}={random.randint(1000, 99999)}"
        if random.random() < 0.3:
            variant += random.choice(paths)
        if variant not in augmented:
            augmented.append(variant)
    random.shuffle(augmented)
    return augmented[:target_count]

def create_dataset(n_benign=5000, n_malicious=5000): # FIX: Reduced from 50k to prevent SVM freezing
    print(f"Generating {n_benign} benign URLs...")
    benign_urls = augment_urls(REALISTIC_BENIGN, n_benign)
    print(f"Generating {n_malicious} malicious URLs...")
    malicious_urls = augment_urls(REALISTIC_MALICIOUS, n_malicious)
    rows = [(url, 0) for url in benign_urls] + [(url, 1) for url in malicious_urls]
    random.shuffle(rows)
    os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)
    with open(DATASET_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["url", "label"])
        writer.writerows(rows)
    print(f"Dataset created: {len(rows)} rows")

def ensure_dataset():
    if not os.path.exists(DATASET_PATH):
        create_dataset(5000, 5000)

# =============================================================================
# 2. FEATURE EXTRACTION (36 features + real WHOIS)
# =============================================================================

try:
    import whois
    HAS_WHOIS = True
except ImportError:
    HAS_WHOIS = False

SUSPICIOUS_TLDS = {
    'xyz','top','click','tk','ml','ga','cf','gq','pw','cc','su','biz','info',
    'online','site','live','stream','download','loan','review','country','kim',
    'science','work','party','trade','cricket','date','faith','racing',
    'accountant','win','bid','men','icu','monster','cyou','buzz','sbs','ru'
}
TRUSTED_DOMAINS = {
    'google.com','youtube.com','facebook.com','microsoft.com','apple.com',
    'amazon.com','github.com','twitter.com','linkedin.com','wikipedia.org',
    'instagram.com','netflix.com','stackoverflow.com','reddit.com','paypal.com',
    'bbc.com','nytimes.com','dropbox.com','mozilla.org','cloudflare.com',
    'medium.com','kaggle.com','huggingface.co','arxiv.org','nature.com',
    'zoom.us','slack.com','notion.so','figma.com','canva.com','stripe.com',
    'shopify.com','heroku.com','vercel.com','netlify.com'
}
BRAND_KEYWORDS = [
    'paypal','google','apple','microsoft','amazon','facebook','instagram',
    'netflix','ebay','steam','whatsapp','youtube','dropbox','icloud','twitter',
    'chase','wellsfargo','citibank','bankofamerica','boa','dhl','fedex','usps','ups'
]
URL_SHORTENERS = {
    'bit.ly','tinyurl.com','t.co','goo.gl','ow.ly','is.gd','buff.ly','rebrand.ly',
    'short.io','tiny.cc','cutt.ly','shorturl.at','rb.gy','short.link','qr.ae','v.gd','tiny.one'
}
PHISH_RE = re.compile(
    r'login|signin|verify|account|update|secure|confirm|password|credential|alert|suspend|unlock|recover|reset|billing|payment|invoice', re.I
)
EXEC_RE = re.compile(r'\.(exe|bat|cmd|msi|scr|vbs|jar|apk|dmg|sh|ps1|crx|xpi)$', re.I)
SPAM_WORDS = [
    'free','win','prize','claim','urgent','alert','suspended','verify',
    'confirm','limited','offer','bonus','gift','reward','lucky','congratulation'
]

def _entropy(s):
    if not s: return 0.0
    freq = {}
    for c in s: freq[c] = freq.get(c, 0) + 1
    n = len(s)
    return -sum((v/n)*math.log2(v/n) for v in freq.values())

def _domain_parts(hostname):
    clean = re.sub(r'^www\.', '', hostname.lower())
    parts = clean.split('.')
    if len(parts) >= 3:
        return '.'.join(parts[:-2]), parts[-2], parts[-1]
    if len(parts) == 2:
        return '', parts[0], parts[1]
    return '', hostname, ''

def get_domain_age(domain):
    if not HAS_WHOIS or not domain:
        return -1
    try:
        w = whois.whois(domain)
        creation = w.creation_date
        if creation is None:
            return -1
        if isinstance(creation, list):
            creation = creation[0]
        return max((datetime.now() - creation).days, 0)
    except Exception:
        return -1

def extract_features(url, expanded_url=None, skip_whois=False): # FIX: Added skip_whois flag
    raw = str(url).strip()
    target_url = expanded_url if expanded_url else raw
    f = {}
    try:
        p = urlparse(target_url if '://' in target_url else 'http://' + target_url)
    except Exception:
        p = urlparse('http://invalid')
    hostname = (p.hostname or '').lower()
    path = p.path or ''
    query = p.query or ''
    scheme = p.scheme or ''
    full_lower = target_url.lower()
    _, domain, tld = _domain_parts(hostname)
    base = f"{domain}.{tld}" if domain and tld else hostname
    sub, _, _ = _domain_parts(hostname)
    hl = max(len(hostname), 1)

    f['is_https'] = int(scheme == 'https')
    f['is_http'] = int(scheme == 'http')
    f['url_length'] = len(target_url)
    f['hostname_length'] = len(hostname)
    f['path_length'] = len(path)
    f['query_length'] = len(query)
    f['dot_count'] = hostname.count('.')
    f['hyphen_count'] = hostname.count('-')
    f['underscore_count'] = target_url.count('_')
    f['at_sign'] = int('@' in target_url)
    f['double_slash'] = int('//' in path)
    f['question_mark'] = int('?' in target_url)
    f['ampersand_count'] = query.count('&')
    f['equals_count'] = query.count('=')
    f['percent_count'] = len(re.findall(r'%[0-9a-fA-F]{2}', target_url))
    f['hash_count'] = int('#' in target_url)
    f['digit_ratio'] = round(sum(c.isdigit() for c in hostname) / hl, 4)
    f['alpha_ratio'] = round(sum(c.isalpha() for c in hostname) / hl, 4)
    f['subdomain_count'] = len(sub.split('.')) if sub else 0
    f['suspicious_tld'] = int(tld in SUSPICIOUS_TLDS)
    f['tld_length'] = len(tld)
    f['is_ip_host'] = int(bool(re.match(r'^\d{1,3}(\.\d{1,3}){3}$', hostname)))
    f['trusted_domain'] = int(base in TRUSTED_DOMAINS)
    brand_hit = any(b in hostname for b in BRAND_KEYWORDS)
    f['brand_in_domain'] = int(brand_hit and base not in TRUSTED_DOMAINS)
    f['digit_in_word'] = int(bool(re.search(r'[a-z]\d[a-z]', hostname)))
    f['phish_path_kw'] = int(bool(PHISH_RE.search(path)))
    f['executable_ext'] = int(bool(EXEC_RE.search(path)))
    f['path_depth'] = path.count('/')
    f['path_has_ip'] = int(bool(re.search(r'/\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', path)))
    try:
        f['param_count'] = len(parse_qs(query))
    except:
        f['param_count'] = 0
    f['hostname_entropy'] = round(_entropy(hostname), 4)
    f['path_entropy'] = round(_entropy(path), 4)
    f['is_shortener'] = int(hostname in URL_SHORTENERS)
    f['spam_keyword_count'] = sum(w in full_lower for w in SPAM_WORDS)
    f['has_punycode'] = int('xn--' in hostname)
    
    # FIX: Only run WHOIS if skip_whois is False, preventing 10k API calls during training
    if skip_whois:
        f['domain_age_days'] = -1
    else:
        domain_age = get_domain_age(base)
        f['domain_age_days'] = domain_age if domain_age >= 0 else 365
        
    return f

FEATURE_COLUMNS = list(extract_features("http://example.com", skip_whois=True).keys())

# =============================================================================
# 3. URL UNSHORTENING
# =============================================================================

def is_shortened_url(url):
    try:
        parsed = urlparse(url if '://' in url else 'http://' + url)
        domain = parsed.netloc.lower()
        if domain.startswith('www.'):
            domain = domain[4:]
        return any(shortener in domain for shortener in URL_SHORTENERS)
    except:
        return False

def unshorten_url(url):
    if not is_shortened_url(url):
        return url
    try:
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        resp = requests.head(url, allow_redirects=True, timeout=10)
        if resp.status_code == 200 and resp.url:
            return resp.url
        resp = requests.get(url, allow_redirects=True, timeout=10)
        if resp.status_code == 200 and resp.url:
            return resp.url
        return url
    except:
        return url

def safe_unshorten(url):
    was = is_shortened_url(url)
    if was:
        expanded = unshorten_url(url)
        return url, expanded, True
    return url, url, False

# =============================================================================
# 4. DATABASE (SQLite)
# =============================================================================

def init_database():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT NOT NULL,
            expanded_url TEXT,
            verdict TEXT NOT NULL,
            risk_score REAL NOT NULL,
            safe_pct REAL NOT NULL,
            mal_pct REAL NOT NULL,
            processing_time REAL,
            ip_address TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS scan_features (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_id INTEGER,
            feature_name TEXT NOT NULL,
            feature_value TEXT,
            FOREIGN KEY (scan_id) REFERENCES scans(id)
        )
    ''')
    conn.commit()
    conn.close()

def log_prediction(data):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT INTO scans (url, expanded_url, verdict, risk_score, safe_pct, mal_pct, processing_time, ip_address)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        data.get('url',''), data.get('expanded_url',''), data.get('verdict',''),
        data.get('risk_score',0), data.get('safe_pct',0), data.get('mal_pct',0),
        data.get('processing_time',0), data.get('ip_address','')
    ))
    scan_id = c.lastrowid
    if 'features' in data:
        for fname, fval in data['features'].items():
            c.execute('INSERT INTO scan_features (scan_id, feature_name, feature_value) VALUES (?,?,?)',
                      (scan_id, fname, str(fval)))
    conn.commit()
    conn.close()
    return scan_id

def get_recent_scans(limit=10):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT id, url, verdict, risk_score, timestamp FROM scans ORDER BY timestamp DESC LIMIT ?', (limit,))
    rows = c.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_total_scans():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT COUNT(*) FROM scans')
    count = c.fetchone()[0]
    conn.close()
    return count

# =============================================================================
# 5. MODEL TRAINING (3 models + FAR/FRR)
# =============================================================================

def calculate_far_frr(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    far = fp / (fp + tn) if (fp + tn) > 0 else 0
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0
    return far, frr

def train_all_models():
    df = pd.read_csv(DATASET_PATH)
    df = df.drop_duplicates(subset=['url']).dropna(subset=['url', 'label'])
    df['label'] = df['label'].astype(int).clip(0,1)

    # FIX: Skip WHOIS during bulk training to prevent network lockouts
    X = pd.DataFrame([extract_features(u, skip_whois=True) for u in df['url']])[FEATURE_COLUMNS].fillna(0).values.astype(float)
    y = df['label'].values

    col_idx = FEATURE_COLUMNS.index('domain_age_days')
    col_vals = X[:, col_idx]
    median_age = np.median(col_vals[col_vals >= 0]) if np.any(col_vals >= 0) else 365
    X[X[:, col_idx] == -1, col_idx] = median_age

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=2,
                                                class_weight='balanced', random_state=42, n_jobs=-1),
        "SVM": SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, class_weight='balanced', random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=12, min_samples_leaf=2, class_weight='balanced', random_state=42)
    }

    results = {}
    best_model = None
    best_acc = 0

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        far, frr = calculate_far_frr(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob) if y_prob is not None else 0

        results[name] = {
            "accuracy": float(acc), "precision": float(prec), "recall": float(rec),
            "f1_score": float(f1), "far": float(far), "frr": float(frr), "auc": float(auc),
            "y_test": y_test.tolist(), "y_pred": y_pred.tolist(), "y_prob": y_prob.tolist() if y_prob is not None else []
        }
        if acc > best_acc:
            best_acc = acc
            best_model = model

    comparison_json = {name: {k:v for k,v in results[name].items() if k not in ['y_test','y_pred','y_prob']}
                       for name in results}
    os.makedirs(os.path.dirname(COMPARISON_PATH), exist_ok=True)
    with open(COMPARISON_PATH, 'w') as f:
        json.dump(comparison_json, f, indent=2)

    joblib.dump({"model": best_model, "feature_columns": FEATURE_COLUMNS}, MODEL_PATH)
    return best_model, results

@st.cache_resource(show_spinner="Loading ThreatScan Engine...")
def load_model():
    ensure_dataset()
    if not os.path.exists(MODEL_PATH):
        model, _ = train_all_models()
        return model, FEATURE_COLUMNS
    payload = joblib.load(MODEL_PATH)
    return payload["model"], payload["feature_columns"]

def get_model_comparison():
    if os.path.exists(COMPARISON_PATH):
        with open(COMPARISON_PATH, 'r') as f:
            return json.load(f)
    return None

# =============================================================================
# 6. PREDICTION FUNCTION
# =============================================================================

def predict_url(url, model, feat_cols, log_to_db=True, db_callback=None):
    start = time.time()
    original, expanded, was_short = safe_unshorten(url)
    feats = extract_features(original, expanded if was_short else None, skip_whois=False) # Only does WhoIs for single URL scans
    X = np.array([feats.get(c,0) for c in feat_cols]).reshape(1,-1)
    prob = model.predict_proba(X)[0]
    safe_pct = round(prob[0]*100,1)
    mal_pct = round(prob[1]*100,1)
    if mal_pct >= 50:
        verdict = "MALICIOUS"
    elif mal_pct >= 30:
        verdict = "SUSPICIOUS"
    else:
        verdict = "SAFE"
    proc_time = round((time.time()-start)*1000,2)

    signals = []
    if feats.get('is_https'): signals.append(("✅ Uses HTTPS", "good"))
    else: signals.append(("⚠️ No HTTPS", "bad"))
    if feats.get('is_ip_host'): signals.append(("⚠️ IP address as host", "bad"))
    if feats.get('suspicious_tld'): signals.append(("⚠️ Suspicious TLD", "bad"))
    if feats.get('brand_in_domain'): signals.append(("⚠️ Brand impersonation", "bad"))
    if feats.get('digit_in_word'): signals.append(("⚠️ Typosquatting", "bad"))
    if feats.get('phish_path_kw'): signals.append(("⚠️ Phishing keywords in path", "bad"))
    if feats.get('is_shortener'): signals.append(("⚠️ URL shortener used", "bad"))
    if feats.get('at_sign'): signals.append(("⚠️ @ symbol in URL", "bad"))
    if feats.get('has_punycode'): signals.append(("⚠️ Punycode / IDN attack", "bad"))
    if feats.get('executable_ext'): signals.append(("⚠️ Executable file extension", "bad"))
    if feats.get('trusted_domain'): signals.append(("✅ Trusted domain", "good"))
    if feats.get('subdomain_count',0) >= 3:
        signals.append((f"⚠️ Deep subdomains ({feats['subdomain_count']})", "bad"))
    if feats.get('hyphen_count',0) >= 3:
        signals.append((f"⚠️ Many hyphens ({feats['hyphen_count']})", "bad"))
    if feats.get('url_length',0) > 100:
        signals.append((f"⚠️ Long URL ({feats['url_length']} chars)", "bad"))
    if feats.get('spam_keyword_count',0) >= 2:
        signals.append((f"⚠️ Spam keywords ({feats['spam_keyword_count']})", "bad"))
    if not any(k=="bad" for _,k in signals):
        signals.append(("✅ No suspicious signals found", "good"))
    if was_short:
        signals.append((f"🔗 Shortened URL expanded from {original[:50]}...", "info"))

    result = {
        "url": original, "expanded_url": expanded if was_short else original,
        "was_shortened": was_short, "verdict": verdict, "risk_score": mal_pct,
        "safe_pct": safe_pct, "mal_pct": mal_pct, "signals": signals,
        "features": feats, "processing_time_ms": proc_time
    }
    if log_to_db and db_callback:
        try:
            db_callback(result)
        except:
            pass
    return result

def get_hostname(url):
    try:
        parsed = urlparse(url if '://' in url else 'http://' + url)
        return parsed.hostname or "unknown"
    except:
        return "unknown"

# =============================================================================
# 7. STREAMLIT UI
# =============================================================================

init_database()

st.set_page_config(page_title="Sentinel - Malicious URL Detector", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
.stApp { background-color: #0b0f19 !important; color: #e2e8f0 !important; }
header, #MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 1rem !important; padding-bottom: 2rem !important; max-width: 1200px !important; }
.sentinel-nav { display: flex; justify-content: space-between; align-items: center; padding: 1rem 0; border-bottom: 1px solid #1e293b; margin-bottom: 2rem; }
.nav-brand { font-size: 1.5rem; font-weight: 700; color: #ffffff; display: flex; align-items: center; gap: 10px; }
.nav-sub { color: #94a3b8; font-size: 0.9rem; font-weight: 500; }
.inspect-card { background-color: #111827; border: 1px solid #1e293b; border-radius: 16px; padding: 2.5rem; margin-bottom: 2rem; text-align: center; }
.inspect-title { font-size: 1.6rem; font-weight: 600; margin-bottom: 0.5rem; color: white; }
.inspect-desc { color: #94a3b8; font-size: 0.9rem; max-width: 500px; margin: 0 auto 1.5rem auto; }
.stTextInput > div > div > input { background-color: #0b0f19 !important; border: 1px solid #1e293b !important; color: white !important; border-radius: 8px !important; padding: 0.8rem 1rem !important; }
.stTextInput > div > div > input:focus { border-color: #4f46e5 !important; box-shadow: none !important; }
.stButton > button { background-color: #1e293b !important; color: #e2e8f0 !important; border: none !important; border-radius: 8px !important; height: 45px !important; width: 100% !important; font-weight: 600 !important; }
.stButton > button:hover { background-color: #334155 !important; color: white !important; }
.result-card { background: #111827; border: 1px solid #1e293b; border-radius: 16px; padding: 1.5rem; margin: 1rem 0; }
.result-safe { border-left: 4px solid #10b981; }
.result-suspicious { border-left: 4px solid #f59e0b; }
.result-malicious { border-left: 4px solid #ef4444; }
.verdict { font-size: 1.8rem; font-weight: 700; }
.verdict-safe { color: #10b981; }
.verdict-suspicious { color: #f59e0b; }
.verdict-malicious { color: #ef4444; }
section[data-testid="stSidebar"] { background-color: #0b0f19 !important; border-right: 1px solid #1e293b !important; }
.history-item { background: #111827; border-radius: 8px; padding: 0.8rem; margin-bottom: 0.6rem; border: 1px solid #1e293b; }
.history-verdict { font-weight: 700; font-size: 0.8rem; margin-bottom: 4px; }
.history-verdict-safe { color: #10b981; }
.history-verdict-mal { color: #ef4444; }
.history-url { color: #94a3b8; font-size: 0.75rem; word-break: break-all; }
.metric-box { background: #111827; border: 1px solid #1e293b; border-radius: 12px; padding: 1rem; text-align: center; }
.metric-value { font-size: 1.8rem; font-weight: 700; }
.metric-label { font-size: 0.7rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="sentinel-nav">
    <div class="nav-brand"><span style="font-size:1.5rem;">🛡️</span> SENTINEL</div>
    <div class="nav-sub">Professional URL Shield · Malicious URL Detector</div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🕒 RECENT SCANS")
    st.markdown("---")
    recent = get_recent_scans(10)
    if not recent:
        st.markdown('<div style="color:#64748b; text-align:center; padding:1rem;">No scan history yet.</div>', unsafe_allow_html=True)
    else:
        for scan in recent:
            vc = "history-verdict-mal" if scan['verdict'] in ["MALICIOUS","CRITICAL","HIGH"] else "history-verdict-safe"
            st.markdown(f'<div class="history-item"><div class="history-verdict {vc}">{scan["verdict"]}</div><div class="history-url">{scan["url"][:50]}...</div></div>', unsafe_allow_html=True)
    st.markdown("---")
    st.caption(f"Total scans logged: {get_total_scans()}")
    st.caption("Powered by Random Forest | 36 Features")

st.markdown("""
<div class="inspect-card">
    <div style="font-size: 2.5rem; margin-bottom: 1rem;">🛡️</div>
    <div class="inspect-title">Ready for Inspection</div>
    <div class="inspect-desc">Enter a URL below to analyze its intent using our Random Forest classifier.</div>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([4,1])
with col1:
    url_input = st.text_input("URL", placeholder="https://example.com/login", label_visibility="collapsed")
with col2:
    analyze_btn = st.button("🔍 Analyze", use_container_width=True)

model, feat_cols = load_model()

if analyze_btn and url_input:
    with st.spinner("Analyzing threat signatures..."):
        def log_cb(res):
            log_prediction({
                'url': res['url'], 'expanded_url': res.get('expanded_url',''),
                'verdict': res['verdict'], 'risk_score': res['risk_score'],
                'safe_pct': res['safe_pct'], 'mal_pct': res['mal_pct'],
                'processing_time': res.get('processing_time_ms',0),
                'ip_address': '127.0.0.1', 'features': res['features']
            })
        result = predict_url(url_input, model, feat_cols, log_to_db=True, db_callback=log_cb)
        st.markdown("<br>", unsafe_allow_html=True)
        verdict = result['verdict']
        result_class = f"result-{verdict.lower()}"
        verdict_class = f"verdict-{verdict.lower()}"
        st.markdown(f"""
        <div class="result-card {result_class}">
            <div style="display: flex; justify-content: space-between; align-items: start; flex-wrap: wrap;">
                <div>
                    <div class="verdict {verdict_class}">{verdict}</div>
                    <div style="color: #94a3b8; font-size: 0.85rem; margin-top: 0.25rem;">
                        Confidence: {result['risk_score']}% | Processing: {result.get('processing_time_ms',0)}ms
                    </div>
                </div>
                <div style="text-align: right;">
                    <div style="color: #64748b; font-size: 0.7rem;">Target Host</div>
                    <div style="font-family: monospace; font-size: 0.8rem;">{get_hostname(url_input)}</div>
                    {f'<div style="color: #f59e0b; font-size: 0.7rem; margin-top: 4px;">🔗 Shortened URL expanded</div>' if result.get('was_shortened') else ''}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a: st.metric("Safe Probability", f"{result['safe_pct']}%")
        with col_b: st.metric("Malicious Probability", f"{result['mal_pct']}%")
        with col_c: st.metric("HTTPS Secured", "Yes" if result['features'].get('is_https') else "No")
        with col_d: st.metric("Domain Age", f"{result['features'].get('domain_age_days',365)} days")

        st.markdown("### 🎯 Threat Indicators")
        indicators = []
        if result['features'].get('suspicious_tld'): indicators.append("Suspicious Top Level Domain detected")
        if result['features'].get('is_ip_host'): indicators.append("IP Address used instead of Domain name")
        if result['features'].get('brand_in_domain'): indicators.append("Potential Brand Impersonation (Typosquatting)")
        if result['features'].get('phish_path_kw'): indicators.append("Phishing keywords found in URL path")
        if not result['features'].get('is_https'): indicators.append("Connection is not secured with HTTPS")
        if result['features'].get('is_shortener'): indicators.append("URL shortener service detected")
        if result['features'].get('executable_ext'): indicators.append("Direct link to an executable file")
        if result['features'].get('spam_keyword_count',0) > 0:
            indicators.append(f"Contains {result['features']['spam_keyword_count']} spam/urgent keywords")
        if result['features'].get('domain_age_days',365) < 30:
            indicators.append(f"Very young domain ({result['features']['domain_age_days']} days old)")
        if not indicators: indicators.append("No significant threat indicators found")
        for ind in indicators: st.markdown(f"- {ind}")

        st.markdown("### 📊 Feature Impact")
        impact = {
            "Suspicious TLD": 85 if result['features'].get('suspicious_tld') else 5,
            "IP Host": 90 if result['features'].get('is_ip_host') else 2,
            "Brand Spoof": 75 if result['features'].get('brand_in_domain') else 4,
            "Phish Keywords": 80 if result['features'].get('phish_path_kw') else 10,
            "Shortener": 60 if result['features'].get('is_shortener') else 5,
            "Young Domain": 70 if result['features'].get('domain_age_days',365) < 30 else 5,
            "Spam Words": min(result['features'].get('spam_keyword_count',0)*20,100),
        }
        items = [(k,v) for k,v in impact.items() if v > 5]
        if not items: items = [("Baseline Safe",10)]
        items.sort(key=lambda x: x[1])
        labels, values = zip(*items)
        colors = ['#10b981' if v<40 else '#f59e0b' if v<70 else '#ef4444' for v in values]
        fig, ax = plt.subplots(figsize=(6,3))
        fig.patch.set_facecolor('#111827')
        ax.set_facecolor('#111827')
        ax.barh(labels, values, color=colors, alpha=0.9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#334155')
        ax.spines['left'].set_color('#334155')
        ax.tick_params(colors='#94a3b8')
        ax.set_xlabel("Impact Score", color='#94a3b8')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

elif analyze_btn and not url_input:
    st.warning("Please enter a URL to analyze.")

# =============================================================================
# MODEL PERFORMANCE SECTION 
# =============================================================================

st.markdown("---")
st.markdown("## 📊 Model Performance")

# FIX: Added @st.cache_data to prevent 10,000 URLs from being processed on every button click
@st.cache_data(show_spinner="Evaluating model performance...")
def get_test_predictions(_model, feat_cols):
    df = pd.read_csv(DATASET_PATH)
    df = df.drop_duplicates(subset=['url']).dropna(subset=['url','label'])
    df['label'] = df['label'].astype(int).clip(0,1)
    
    # Extract features safely, skipping WHOIS lookups
    X = pd.DataFrame([extract_features(u, skip_whois=True) for u in df['url']])[feat_cols].fillna(0).values.astype(float)
    y = df['label'].values
    
    col_idx = feat_cols.index('domain_age_days')
    col_vals = X[:, col_idx]
    median_age = np.median(col_vals[col_vals >= 0]) if np.any(col_vals >= 0) else 365
    X[X[:, col_idx] == -1, col_idx] = median_age
    
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    y_pred = _model.predict(X_test)
    y_prob = _model.predict_proba(X_test)[:,1]
    return y_test, y_pred, y_prob

comparison = get_model_comparison()
if comparison:
    df_comp = pd.DataFrame([
        {
            "Model": name,
            "Accuracy": f"{data['accuracy']*100:.2f}%",
            "Precision": f"{data['precision']*100:.2f}%",
            "Recall": f"{data['recall']*100:.2f}%",
            "F1-Score": f"{data['f1_score']*100:.2f}%",
            "FAR": f"{data['far']*100:.2f}%",
            "FRR": f"{data['frr']*100:.2f}%",
            "AUC": f"{data['auc']:.3f}"
        } for name, data in comparison.items()
    ])
    st.dataframe(df_comp, use_container_width=True)
    st.info("Performance Targets: Accuracy ≥ 95%, FAR ≤ 2%, FRR ≤ 3%")

    y_test, y_pred, y_prob = get_test_predictions(model, feat_cols)
    col_cm, col_roc = st.columns(2)
    with col_cm:
        st.markdown("### Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(5,4))
        fig_cm.patch.set_facecolor('#111827')
        ax_cm.set_facecolor('#111827')
        ConfusionMatrixDisplay(cm, display_labels=["Benign","Malicious"]).plot(ax=ax_cm, colorbar=False, cmap='Blues')
        ax_cm.tick_params(colors='white')
        ax_cm.title.set_color('white')
        st.pyplot(fig_cm)
        plt.close(fig_cm)
    with col_roc:
        st.markdown("### ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        fig_roc, ax_roc = plt.subplots(figsize=(5,4))
        fig_roc.patch.set_facecolor('#111827')
        ax_roc.set_facecolor('#111827')
        ax_roc.plot(fpr, tpr, color='#4f46e5', lw=2, label=f'AUC = {auc:.3f}')
        ax_roc.plot([0,1],[0,1], 'k--', lw=1, alpha=0.5)
        ax_roc.set_xlabel('False Positive Rate', color='white')
        ax_roc.set_ylabel('True Positive Rate', color='white')
        ax_roc.set_title('ROC Curve', color='white')
        ax_roc.legend(loc='lower right')
        ax_roc.tick_params(colors='white')
        st.pyplot(fig_roc)
        plt.close(fig_roc)
else:
    st.info("Run model training first to see performance metrics. This will happen automatically on first scan.")

if analyze_btn and url_input:
    with st.expander("View Full Feature Vector (36 features)"):
        st.dataframe(pd.DataFrame(result['features'].items(), columns=["Feature","Value"]).set_index("Feature"),
                     use_container_width=True, height=400)
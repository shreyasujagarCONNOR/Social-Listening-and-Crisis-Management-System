"""ZipBite core logic: Gemini classifier, spread models, RAG reply drafter.

Used by app.py. Nothing in here posts anything: it only classifies, predicts and drafts.
"""
import json
import re
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

HERE = Path(__file__).parent
MODEL = "gemini-3.5-flash-lite"        # change here if Google retires this model name
EMBED_MODEL = "gemini-embedding-001"   # change here if Google retires this model name

TOPICS = ["Delivery Logistics", "Rider Interaction", "Food Quality & Safety", "Order Accuracy",
          "Packaging & Spillage", "App & UX Experience", "Pricing & Promos", "Customer Support"]
LLM_FIELDS = ["language", "sentiment_from_text_only", "sarcasm", "high_arousal_emotion", "urgency",
              "topic", "needs_reply", "relevancy_to_todays_events"]
INT_FIELDS = {"sarcasm", "high_arousal_emotion", "relevancy_to_todays_events"}
ALLOWED = {
    "language": {"English", "Hinglish", "Hindi"},
    "sentiment_from_text_only": {"negative", "neutral", "positive"},
    "sarcasm": {0, 1}, "high_arousal_emotion": {0, 1},
    "urgency": {"urgent", "not urgent"},
    "topic": set(TOPICS),
    "needs_reply": {"yes", "no"},
    "relevancy_to_todays_events": {0, 1},
}
CATS = {
    "sentiment_from_text_only": ["negative", "neutral", "positive"],
    "topic": TOPICS,
    "language": ["English", "Hinglish", "Hindi"],
    "platform": ["X", "Instagram", "Facebook", "Reddit", "YouTube", "LinkedIn"],
}

# What is happening "today". Editable in the app's sidebar. These are placeholder events.
DEFAULT_EVENTS = """- Diwali / festive rush: 14 Oct - 27 Oct 2026
- Evening cricket match nights (city-wide ordering spikes)
- Heavy rain and waterlogging in big cities
- FSSAI food-safety raids on delivery kitchens: 22 Sep - 6 Oct 2026
- Delivery partners' strike: 1 Oct - 9 Oct 2026
- ZipBite's new Diwali ad campaign: 20 Oct - 27 Oct 2026"""


def classifier_prompt(events):
    return f"""You are a social-media analyst for ZipBite, an Indian food delivery app.
You read ONE public post at a time. Posts are in English, Hinglish (Hindi in Roman letters) or Hindi (Devanagari).

Return ONLY a JSON object with exactly these keys:
- "language": "English" | "Hinglish" | "Hindi"
- "sentiment_from_text_only": "negative" | "neutral" | "positive"
- "sarcasm": 0 or 1
- "high_arousal_emotion": 0 or 1
- "urgency": "urgent" | "not urgent"
- "topic": one of {TOPICS}
- "needs_reply": "yes" | "no"
- "relevancy_to_todays_events": 0 or 1

Definitions:
- language: "Hindi" = written in Devanagari script; "Hinglish" = Hindi words written in Roman letters (alone or mixed with English); "English" = English. Judge the main language of the post.
- sentiment_from_text_only: judge the LITERAL words only. If a post uses praise words to mock the brand, the literal sentiment is "positive" (do NOT correct it; sarcasm is flagged separately).
- sarcasm = 1 only when the wording is positive but the real meaning is a complaint or mockery.
- high_arousal_emotion = 1 when the post shows strong emotion: anger, rage, fear, disgust, outrage, shock, or very excited/hysterical laughter (many capitals, "!!!", angry emojis). A calm complaint is 0.
- urgency = "urgent" only for health or safety risk, food poisoning, harassment, privacy/data leak, discrimination or legal threats. A late or wrong order is "not urgent".
- topic: the main subject of the post.
- needs_reply = "yes" if a real customer has a complaint or a question the brand should answer. "no" for praise and plain statements.
- relevancy_to_todays_events = 1 if the post refers to one of these current events (by name or clearly), otherwise 0:
{events}
  Use the post time given with the post. A generic complaint with no link to an event is 0.

Rules: judge only what the post says; do not invent facts. Output the JSON object and nothing else.

Examples (hand-written, not from the dataset):
Post time: 2026-10-20 21:00
Post: Wah @ZipBite, 3 ghante se order nahi aaya. Kya speed hai, dil jeet liya 🙄
{{"language":"Hinglish","sentiment_from_text_only":"positive","sarcasm":1,"high_arousal_emotion":0,"urgency":"not urgent","topic":"Delivery Logistics","needs_reply":"yes","relevancy_to_todays_events":0}}

Post time: 2026-10-22 20:30
Post: Diwali ki raat aur @ZipBite ke khane mein baal nikla. Sab ko ulti ho rahi hai 🤬 #FoodSafety
{{"language":"Hinglish","sentiment_from_text_only":"negative","sarcasm":0,"high_arousal_emotion":1,"urgency":"urgent","topic":"Food Quality & Safety","needs_reply":"yes","relevancy_to_todays_events":1}}

Post time: 2026-08-14 13:10
Post: Does @ZipBite deliver near Pune station?
{{"language":"English","sentiment_from_text_only":"neutral","sarcasm":0,"high_arousal_emotion":0,"urgency":"not urgent","topic":"Delivery Logistics","needs_reply":"yes","relevancy_to_todays_events":0}}
"""


DRAFT_SYSTEM = """You write PUBLIC reply drafts for ZipBite's social media team. A human reviews every draft before anything is posted.
Use ONLY the SOURCES block for tone and for anything the brand may promise or do.
Hard rules:
- Write the reply in REPLY LANGUAGE exactly: English; Hinglish = Hindi in Roman letters mixed with English (do NOT write plain English); Hindi = Devanagari script.
- Never promise refunds, credits, compensation, amounts, timelines or outcomes unless a SOURCE explicitly allows saying it publicly.
- Do not use speed or outcome words such as 'immediately', 'right away', 'sort this out', 'resolved', 'तुरंत'. Say 'we are looking into it' or 'our team will review' instead.
- When you ask the customer to send something, say it in plain words ('your order ID'). Use a placeholder like [NAME] ONLY for a detail ZipBite itself must fill in.
- Never invent order details. Never give medical advice beyond 'please see a doctor if you feel unwell'. Never admit legal liability, blame the customer, the rider or anyone, or argue.
- Keep it under 280 characters.
- If the post is sarcastic, answer the real complaint calmly. Do not mirror the sarcasm.
- If ESCALATE is true, follow the crisis and safety sources and move the conversation to DM.
Return ONLY JSON: {"reply": "...", "sources_used": ["id", ...], "placeholders": ["..."]}"""

PROMISE_WORDS = re.compile(
    r"refund|credit|compensat|\bfree\b|guarantee|voucher|discount|₹|\brs\.?\s*\d|immediate|right away|\basap\b|"
    r"within \d|definitely|for sure|sort (this|it|that) out|resolve|रिफंड|मुफ़्त|मुफ्त|वापस मिल|तुरंत|फौरन|"
    r"अभी के अभी|गारंटी|turant|abhi ke abhi|jaldi hi", re.I)

LANG_NOTE = {"English": "English",
             "Hinglish": "Hinglish (Hindi in Roman letters mixed with English)",
             "Hindi": "Hindi (Devanagari script)"}

# ----------------------------------------------------------------------------- Gemini helpers
_lock, _last = threading.Lock(), [0.0]


def throttle(min_gap=1.0):
    """Keep at least `min_gap` seconds between Gemini calls."""
    with _lock:
        wait = _last[0] + min_gap - time.time()
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()


def make_client(api_key):
    from google import genai
    return genai.Client(api_key=api_key)


def _config(system, temperature):
    from google.genai import types
    return types.GenerateContentConfig(system_instruction=system, temperature=temperature,
                                       response_mime_type="application/json")


def _json(raw):
    return json.loads(raw.strip().removeprefix("```json").removesuffix("```").strip())


def parse_labels(raw):
    """Validate the classifier's reply. Never raises."""
    try:
        data = _json(raw)
        out = {}
        for k in LLM_FIELDS:
            v = data.get(k)
            if k in INT_FIELDS:
                v = int(v)
            if k == "language" and v not in ALLOWED[k]:
                v = "English"                      # some other language: answer in English
            if v not in ALLOWED[k]:
                raise ValueError(f"bad value for {k}: {v}")
            out[k] = v
        return {"ok": True, "labels": out}
    except Exception as e:
        return {"ok": False, "error": str(e)[:120]}


def classify_post(client, timestamp, text, events=DEFAULT_EVENTS, retries=4):
    """Step 1: Gemini labels one post. Only the post time and text are sent."""
    msg = f"Post time: {timestamp}\nPost: {text}"
    last = {"ok": False, "error": "api_failed"}
    for attempt in range(retries):
        try:
            throttle()
            r = client.models.generate_content(model=MODEL, contents=msg,
                                               config=_config(classifier_prompt(events), 0))
            parsed = parse_labels(r.text)
            if parsed["ok"]:
                return parsed
            last = parsed
        except Exception as e:
            last = {"ok": False, "error": str(e)[:120]}
            time.sleep(3 * 2 ** attempt)
    return last


# ----------------------------------------------------------------------------- spread models
def build_features(d):
    """Real-time features: Gemini's labels + author/post details (no likes, no comments)."""
    t = pd.to_datetime(d["timestamp"])
    X = pd.DataFrame(index=d.index)
    X["is_night_post"] = ((t.dt.hour >= 22) | (t.dt.hour < 6)).astype(int)
    X["is_weekend"] = (t.dt.dayofweek >= 5).astype(int)
    X["urgent"] = (d["urgency"] == "urgent").astype(int)
    X["needs_reply"] = (d["needs_reply"] == "yes").astype(int)
    for c in ["sarcasm", "high_arousal_emotion", "relevancy_to_todays_events",
              "author_verified", "has_picture", "has_profile_picture"]:
        X[c] = d[c].astype(int)
    for col, cats in CATS.items():
        for c in cats:
            X[f"{col}={c}"] = (d[col] == c).astype(int)
    X["log10_followers"] = np.log10(d["author_followers"].clip(lower=0) + 1)
    return X


def build_features_1h(d):
    """One-hour features: the real-time ones PLUS first-hour likes and comments."""
    X = build_features(d)
    X["log1p_likes_1h"] = np.log1p(d["likes_1h"])
    X["log1p_comments_1h"] = np.log1p(d["comments_1h"])
    return X


class SimpleLogit:
    """A logistic regression rebuilt from plain numbers (coefficients, intercept, scaling).

    It gives the same answers as the scikit-learn model it was exported from, without needing
    scikit-learn (or any particular version of it) to load the model.
    """

    def __init__(self, spec):
        self.features = spec["features"]
        self.coef = np.array(spec["coef"], dtype=float)
        self.intercept = float(spec["intercept"])
        self.scaled = [self.features.index(c) for c in spec["scaled_columns"]]
        self.mean = np.array(spec["mean"], dtype=float)
        self.scale = np.array(spec["scale"], dtype=float)

    def predict_proba(self, X):
        x = X[self.features].to_numpy(dtype=float).copy()
        x[:, self.scaled] = (x[:, self.scaled] - self.mean) / self.scale
        p = 1.0 / (1.0 + np.exp(-(x @ self.coef + self.intercept)))
        return np.column_stack([1 - p, p])


def _find(base, name):
    """The model files may sit in a models/ folder or directly next to app.py."""
    for p in (base / "models" / name, base / name):
        if p.exists():
            return p
    raise FileNotFoundError(f"{name} not found in {base} or {base / 'models'}")


def is_negative(labels):
    """Truly negative: negative words, or praise words used sarcastically."""
    s_ = labels["sentiment_from_text_only"]
    return s_ == "negative" or (s_ == "positive" and labels["sarcasm"] == 1)


def load_assets(base=HERE):
    kb = json.load(open(base / "knowledge_base.json", encoding="utf-8"))
    cfg = json.load(open(base / "config.json", encoding="utf-8"))
    b = json.load(open(base / "models.json", encoding="utf-8"))   # plain numbers: no scikit-learn version problems
    return {"viral": SimpleLogit(b["viral"]), "viral_1h": SimpleLogit(b["viral_1h"]),
            "columns": b["columns"], "columns_1h": b["columns_1h"], "kb": kb,
            "cutoff_viral": float(cfg["cutoff_viral"]), "cutoff_crisis": float(cfg["cutoff_crisis"])}


def _row(labels, post, likes=None, comments=None):
    row = dict(labels, timestamp=post["timestamp"], platform=post["platform"],
               author_followers=post["author_followers"], author_verified=int(post["author_verified"]),
               has_picture=int(post["has_picture"]), has_profile_picture=int(post["has_profile_picture"]))
    if likes is not None:
        row.update(likes_1h=likes, comments_1h=comments)
    return pd.DataFrame([row])


def _verdict(assets, labels, pv):
    """A crisis is a post that spreads AND is negative. So: crisis chance = viral chance, but only for a
    negative post. A post that cannot spread cannot be a crisis, and a neutral or happy post is never one."""
    neg = is_negative(labels)
    pc = pv if neg else 0.0
    return {"p_viral": pv, "p_crisis": pc, "negative": neg,
            "will_go_viral": pv >= assets["cutoff_viral"], "is_crisis": neg and pc >= assets["cutoff_crisis"]}


def predict_realtime(assets, labels, post):
    """Step 2, at posting time."""
    X = build_features(_row(labels, post))[assets["columns"]]
    return _verdict(assets, labels, float(assets["viral"].predict_proba(X)[0, 1]))


def predict_one_hour(assets, labels, post, likes, comments):
    """Step 2b, one hour later: same labels, plus the real engagement."""
    X = build_features_1h(_row(labels, post, likes, comments))[assets["columns_1h"]]
    return _verdict(assets, labels, float(assets["viral_1h"].predict_proba(X)[0, 1]))


# ----------------------------------------------------------------------------- RAG drafter
class Retriever:
    """Finds the tone-guide and playbook entries that fit a post."""

    def __init__(self, client, kb):
        self.client, self.kb = client, kb
        texts = [f"{k['title']}. {k['text']}" for k in kb]
        try:
            self.vec, self.mode = self._embed(texts), "embeddings"
        except Exception:
            self.tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)).fit(texts)
            self.vec, self.mode = self.tfidf.transform(texts), "keyword fallback"

    def _embed(self, texts):
        r = self.client.models.embed_content(model=EMBED_MODEL, contents=texts)
        return np.array([e.values for e in r.embeddings])

    def _encode(self, texts):
        return self._embed(texts) if self.mode == "embeddings" else self.tfidf.transform(texts)

    def retrieve(self, post_text, labels, escalate, extra=3):
        query = f"{post_text}\nTopic: {labels['topic']}. Sentiment: {labels['sentiment_from_text_only']}."
        try:
            sims = cosine_similarity(self._encode([query]), self.vec)[0]
        except Exception:
            sims = np.zeros(len(self.kb))
        must = ["tone_voice", "tone_language"]
        if escalate:
            must += ["playbook_crisis"] + (["playbook_safety"] if labels["urgency"] == "urgent" else [])
        if labels["sarcasm"] == 1:
            must += ["playbook_sarcasm"]
        ranked = [self.kb[i]["id"] for i in np.argsort(-sims)]
        ids = must + [i for i in ranked if i not in must][:extra]
        return [k for k in self.kb if k["id"] in ids]


def draft_reply(client, retriever, post_text, labels, escalate, language, retries=4):
    """Step 3: draft a reply from the retrieved playbook entries. A person still approves it."""
    sources = retriever.retrieve(post_text, labels, escalate)
    src_block = "\n".join(f"[{k['id']}] ({k['kind']}: {k['title']}) {k['text']}" for k in sources)
    msg = (f"POST: {post_text}\nREPLY LANGUAGE: {LANG_NOTE.get(language, language)}\n"
           f"LABELS: {json.dumps(labels, ensure_ascii=False)}\nESCALATE: {escalate}\n\nSOURCES:\n{src_block}")
    for attempt in range(retries):
        try:
            throttle()
            r = client.models.generate_content(model=MODEL, contents=msg, config=_config(DRAFT_SYSTEM, 0.3))
            d = _json(r.text)
            reply = d["reply"].strip()
            word = "ऑर्डर आईडी" if language == "Hindi" else "order ID"
            reply = re.sub(r"\[\s*order[ _]?id\s*\]", word, reply, flags=re.I)
            left = re.findall(r"\[[A-Za-z_ ]+\]", reply)
            flags = []
            if len(reply) > 280:
                flags.append("TOO_LONG")
            if PROMISE_WORDS.search(reply):
                flags.append("CHECK_PROMISE")
            if left:
                flags.append("FILL_PLACEHOLDERS")
            return {"reply": reply, "flags": flags, "sources_used": d.get("sources_used", []),
                    "retrieved": [k["id"] for k in sources]}
        except Exception:
            time.sleep(3 * 2 ** attempt)
    return {"reply": None, "flags": ["DRAFT_FAILED"], "sources_used": [], "retrieved": []}


# ----------------------------------------------------------------------------- the whole pipeline
def analyse(client, retriever, assets, post, events=DEFAULT_EVENTS):
    """Classify -> predict -> (if a reply is needed) draft. Returns everything the app shows."""
    out = {"post": post}                     # the post's language is read by the classifier, not typed in
    c = classify_post(client, post["timestamp"], post["text"], events)
    if not c["ok"]:
        out.update(status="NEEDS_HUMAN_REVIEW", error=c["error"])
        return out
    L = c["labels"]
    rt = predict_realtime(assets, L, post)
    urgent = L["urgency"] == "urgent"
    alert = bool(rt["is_crisis"] or urgent)          # urgent posts always escalate to a person
    reply_needed = L["needs_reply"] == "yes" or urgent
    out.update(labels=L, realtime=rt, urgent=urgent, alert=alert, reply_needed=reply_needed,
               priority=("URGENT" if alert else "NORMAL") if reply_needed else ("ALERT" if rt["is_crisis"] else "NONE"),
               draft=None, status="NO_REPLY_NEEDED")
    if reply_needed:
        d = draft_reply(client, retriever, post["text"], L, alert, L["language"])
        out.update(draft=d, status="PENDING_APPROVAL" if d["reply"] else "NEEDS_HUMAN_REVIEW")
    return out

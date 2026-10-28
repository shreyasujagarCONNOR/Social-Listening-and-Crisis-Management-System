"""ZipBite: AI-powered social listening and crisis response (demo).

Flow: Gemini classifies a post -> two models predict "will it go viral?" and "is it a crisis?"
-> if a reply is needed, the RAG drafter writes one from the tone guide and playbook
-> a person approves, edits or rejects it. Nothing is ever posted automatically.
"""
import datetime as dt
import os

import pandas as pd
import streamlit as st

import zipbite_core as zc

st.set_page_config(page_title="ZipBite Social Listening", page_icon="🛵", layout="wide")

DEMOS = {
    "Routine mention": dict(
        text="Ordered a thali from @ZipBite for lunch. Arrived hot and right on time. 👍",
        language="English", platform="Instagram", followers=800, verified=False, pic=False, profpic=True),
    "Complaint needing a reply": dict(
        text="@ZipBite my biryani order is 75 mins late and the support chat is not opening. Rs 450 already paid. Please respond.",
        language="English", platform="X", followers=2500, verified=False, pic=False, profpic=True),
    "Crisis (food safety)": dict(
        text="Diwali week and @ZipBite sends me a cockroach in my thali. My kid already ate half. 📸 Photo attached. Tagging @FSSAI, please RT. #FoodSafety",
        language="English", platform="X", followers=18000, verified=False, pic=True, profpic=True),
    "Crisis (Hindi, hospital)": dict(
        text="@ZipBite के खाने में कीड़ा निकला! बच्चे को उल्टी हो रही है, अस्पताल जा रहे हैं 😡😡 #FoodSafety",
        language="Hindi", platform="X", followers=25000, verified=True, pic=False, profpic=True),
    "Edge case: sarcasm (Hinglish)": dict(
        text="Wah @ZipBite, 2 ghante se 'arriving soon'. Kya service hai bhai, dil jeet liya 🙄",
        language="Hinglish", platform="X", followers=1200, verified=False, pic=False, profpic=True),
    "Edge case: mixed feelings (Hinglish)": dict(
        text="@ZipBite pizza theek-thaak tha par rider bhaiya ko raise do, barish mein 20 min tak gate pe khade rahe 🙏",
        language="Hinglish", platform="Facebook", followers=600, verified=False, pic=False, profpic=True),
}

FLAG_HELP = {
    "CHECK_PROMISE": "The draft may promise money, speed or an outcome. Check it against the playbook before approving.",
    "TOO_LONG": "The draft is over 280 characters. Shorten it.",
    "FILL_PLACEHOLDERS": "The draft still has a [PLACEHOLDER] that a person must fill in.",
    "DRAFT_FAILED": "The drafter did not return a reply. A person must write one.",
}
PRIORITY_HELP = {
    "URGENT": "URGENT: look at this first", "NORMAL": "NORMAL", "ALERT": "ALERT: crisis predicted, no reply needed",
    "NONE": "No action needed",
}

# ------------------------------------------------------------------ state
for k, v in dict(f_text="", f_lang="English", f_plat="X", f_fol=1000, f_ver=False, f_pic=False, f_pp=True,
                 f_date=dt.date.today(), f_time=dt.datetime.now().time().replace(second=0, microsecond=0),
                 result=None, log=[], rid=0, events=zc.DEFAULT_EVENTS, demo_choice="Custom post").items():
    st.session_state.setdefault(k, v)


def load_demo():
    name = st.session_state.demo_choice
    if name in DEMOS:
        d = DEMOS[name]
        st.session_state.update(f_text=d["text"], f_lang=d["language"], f_plat=d["platform"], f_fol=d["followers"],
                                f_ver=d["verified"], f_pic=d["pic"], f_pp=d["profpic"])
        st.session_state.result = None


def api_key():
    try:
        key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        key = None
    return key or os.environ.get("GEMINI_API_KEY")


@st.cache_resource(show_spinner="Loading the models...")
def get_assets():
    return zc.load_assets()


@st.cache_resource(show_spinner="Connecting to Gemini and indexing the playbook...")
def get_services(key):
    client = zc.make_client(key)
    return client, zc.Retriever(client, get_assets()["kb"])


def record_decision(res, decision, reviewer, final_text, reason=""):
    row = {"decided_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "reviewer": reviewer, "decision": decision,
           "post": res["post"]["text"], "priority": res["priority"],
           "viral_prob": round(res["realtime"]["p_viral"], 3), "crisis_prob": round(res["realtime"]["p_crisis"], 3),
           "draft": res["draft"]["reply"], "final_text": final_text if decision == "APPROVED" else "",
           "flags": ", ".join(res["draft"]["flags"]), "reject_reason": reason}
    st.session_state.log.append(row)
    res["decision"] = {"decision": decision, "reviewer": reviewer}
    try:
        pd.DataFrame(st.session_state.log).to_csv("approval_log.csv", index=False, encoding="utf-8-sig")
    except Exception:
        pass   # file saving can fail on read-only hosts; the in-app log still works


# ------------------------------------------------------------------ sidebar
assets = None
try:
    assets = get_assets()
except Exception as e:
    st.sidebar.error(f"Could not load the model files: {e}")

with st.sidebar:
    st.header("Try a demo post")
    st.selectbox("Pick a post", ["Custom post"] + list(DEMOS), key="demo_choice", on_change=load_demo)
    st.divider()
    st.subheader("Today's events")
    st.caption("The classifier uses this list to decide if a post is about something happening now. Edit it for the day.")
    st.text_area("Events", key="events", height=210, label_visibility="collapsed")
    if assets:
        st.divider()
        st.caption(f"Alert cutoffs: viral {assets['cutoff_viral']:.0%}, crisis {assets['cutoff_crisis']:.0%}")

st.title("ZipBite Social Listening")
st.caption("Prototype on synthetic data. A person approves every reply; nothing is posted automatically.")

tab_main, tab_log, tab_about = st.tabs(["Analyse a post", "Approval log", "How it works"])

# ------------------------------------------------------------------ main tab
with tab_main:
    left, right = st.columns([2, 3], gap="large")

    with left:
        st.subheader("1. The post")
        st.text_area("Post text", key="f_text", height=140, placeholder="Paste a post that mentions @ZipBite")
        c1, c2 = st.columns(2)
        c1.selectbox("Language", ["English", "Hinglish", "Hindi"], key="f_lang")
        c2.selectbox("Platform", zc.CATS["platform"], key="f_plat")
        c3, c4 = st.columns(2)
        c3.date_input("Date", key="f_date")
        c4.time_input("Time", key="f_time")
        st.number_input("Author followers", min_value=0, step=100, key="f_fol")
        c5, c6, c7 = st.columns(3)
        c5.checkbox("Verified", key="f_ver")
        c6.checkbox("Has picture", key="f_pic")
        c7.checkbox("Profile photo", key="f_pp")
        go = st.button("Analyse post", type="primary", width="stretch")

    if go:
        key = api_key()
        if not st.session_state.f_text.strip():
            st.warning("Type or pick a post first.")
        elif not key:
            st.error("No Gemini API key found. Add GEMINI_API_KEY to the app's secrets.")
        elif assets is None:
            st.error("The model files did not load, so predictions are unavailable.")
        else:
            post = dict(text=st.session_state.f_text.strip(), language=st.session_state.f_lang,
                        platform=st.session_state.f_plat,
                        timestamp=f"{st.session_state.f_date} {st.session_state.f_time:%H:%M}",
                        author_followers=int(st.session_state.f_fol), author_verified=bool(st.session_state.f_ver),
                        has_picture=bool(st.session_state.f_pic), has_profile_picture=bool(st.session_state.f_pp))
            with st.spinner("Classifying, predicting and drafting..."):
                client, retriever = get_services(key)
                res = zc.analyse(client, retriever, assets, post, st.session_state.events)
            st.session_state.rid += 1
            res.update(rid=st.session_state.rid, decision=None, one_hour=None)
            st.session_state.result = res

    with right:
        res = st.session_state.result
        if res is None:
            st.info("Pick a demo post in the sidebar or paste your own, then press Analyse post.")
        elif "labels" not in res:
            st.error("The classifier could not read this post. Please send it to a person to review.")
            st.caption(f"Reason: {res.get('error')}")
        else:
            L, rt = res["labels"], res["realtime"]

            # --- alert banner
            if rt["is_crisis"]:
                st.error("CRISIS ALERT: this post is negative and likely to spread. Escalate to a person now."
                         + (" It is also marked urgent." if res["urgent"] else ""))
            elif res["urgent"]:
                st.error("URGENT: possible health, safety or legal issue. A person must look at this now, "
                         "even though the spread score is low.")
            elif rt["will_go_viral"]:
                st.warning("Likely to go viral, but not predicted as a crisis. Keep an eye on it.")
            else:
                st.success("Routine: no alert.")

            # --- step 1
            st.subheader("2. What the classifier read (Gemini)")
            st.table(pd.DataFrame({"Field": ["Sentiment (words only)", "Sarcasm", "Strong emotion", "Urgency",
                                             "Topic", "Needs a reply", "About today's events"],
                                   "Value": [L["sentiment_from_text_only"], "yes" if L["sarcasm"] else "no",
                                             "yes" if L["high_arousal_emotion"] else "no", L["urgency"],
                                             L["topic"], L["needs_reply"],
                                             "yes" if L["relevancy_to_todays_events"] else "no"]}).set_index("Field"))

            # --- step 2
            st.subheader("3. Predictions at posting time")
            m1, m2 = st.columns(2)
            for col, label, p, yes, cut in [(m1, "Will it go viral?", rt["p_viral"], rt["will_go_viral"], assets["cutoff_viral"]),
                                            (m2, "Is it a crisis?", rt["p_crisis"], rt["is_crisis"], assets["cutoff_crisis"])]:
                col.metric(label, f"{p:.0%}")
                col.progress(min(max(p, 0.0), 1.0))
                col.caption(f"{'YES' if yes else 'No'} (alert cutoff {cut:.0%})")

            # --- step 3
            st.subheader("4. Reply draft for a person to approve")
            st.caption(f"Priority: {PRIORITY_HELP[res['priority']]}")
            d = res["draft"]
            if not res["reply_needed"]:
                st.info("No reply needed for this post.")
                if res["priority"] == "ALERT":
                    st.warning("A crisis is predicted, so a person should still look at this post.")
            elif d is None or not d["reply"]:
                st.error("The drafter did not return a reply. A person must write one.")
            else:
                decided = res["decision"]
                edited = st.text_area("Draft reply (edit freely before approving)", value=d["reply"],
                                      key=f"draft_{res['rid']}", height=130, disabled=bool(decided))
                st.caption(f"{len(edited)} / 280 characters")
                for f in d["flags"]:
                    st.warning(f"{f}: {FLAG_HELP.get(f, '')}")
                with st.expander("Which playbook pages the drafter read"):
                    titles = {k["id"]: f"{k['title']} ({k['kind']})" for k in assets["kb"]}
                    for rid_ in d["retrieved"]:
                        st.write(("**" if rid_ in d["sources_used"] else "") + titles.get(rid_, rid_)
                                 + ("**  (used)" if rid_ in d["sources_used"] else ""))
                if decided:
                    if decided["decision"] == "APPROVED":
                        st.success(f"Approved by {decided['reviewer']}. Copy the text and post it yourself; the app never posts.")
                    else:
                        st.info(f"Rejected by {decided['reviewer']}.")
                else:
                    st.text_input("Your name (needed to approve or reject)", key="reviewer")
                    reviewer = st.session_state.get("reviewer", "").strip()
                    b1, b2 = st.columns(2)
                    if b1.button("Approve this reply", type="primary", disabled=not reviewer or not edited.strip(),
                                 width="stretch"):
                        record_decision(res, "APPROVED", reviewer, edited.strip())
                        st.rerun()
                    reason = st.text_input("Reason for rejecting (optional)", key=f"reason_{res['rid']}")
                    if b2.button("Reject", disabled=not reviewer, width="stretch"):
                        record_decision(res, "REJECTED", reviewer, "", reason)
                        st.rerun()

            # --- one hour later
            with st.expander("One hour later: re-check with likes and comments"):
                e1, e2 = st.columns(2)
                likes = e1.number_input("Likes in the first hour", min_value=0, step=10, key=f"lk_{res['rid']}")
                comments = e2.number_input("Comments in the first hour", min_value=0, step=5, key=f"cm_{res['rid']}")
                if st.button("Re-check", key=f"recheck_{res['rid']}"):
                    res["one_hour"] = zc.predict_one_hour(assets, L, res["post"], int(likes), int(comments))
                oh = res["one_hour"]
                if oh:
                    st.table(pd.DataFrame({"Viral": [f"{rt['p_viral']:.0%}", f"{oh['p_viral']:.0%}"],
                                           "Crisis": [f"{rt['p_crisis']:.0%}", f"{oh['p_crisis']:.0%}"]},
                                          index=["At posting time", "After one hour"]))
                    if oh["is_crisis"] and not rt["is_crisis"]:
                        st.error("Now flagged as a crisis: it is spreading. Escalate to a person.")
                    elif oh["will_go_viral"] and not rt["will_go_viral"]:
                        st.warning("Now flagged as likely to go viral.")
                    elif rt["is_crisis"] and not oh["is_crisis"]:
                        st.info("The one-hour score is lower, but the alert from posting time stays until a person lowers it.")
                    else:
                        st.success("No change in alert status.")

# ------------------------------------------------------------------ log tab
with tab_log:
    st.subheader("Approval log (this session)")
    if st.session_state.log:
        log_df = pd.DataFrame(st.session_state.log)
        st.dataframe(log_df, width="stretch", hide_index=True)
        st.download_button("Download log (CSV)", log_df.to_csv(index=False).encode("utf-8-sig"),
                           file_name="approval_log.csv", mime="text/csv")
        st.caption("On a hosted demo this log resets when the app restarts. Download it to keep it.")
    else:
        st.info("No decisions yet. Approve or reject a draft and it will appear here.")

# ------------------------------------------------------------------ about tab
with tab_about:
    st.markdown("""
**What happens to a post**

1. **Gemini reads the post** and labels sentiment, sarcasm, strong emotion, urgency, topic, whether a reply is needed, and whether it is about today's events.
2. **Two logistic-regression models** use those labels plus author details to estimate *will it go viral?* and *is it a crisis?* at posting time. One hour later, a second pair of models also uses likes and comments.
3. **If a reply is needed**, the drafter reads the matching pages of the team's tone guide and crisis playbook and writes a draft. Urgent posts always get a draft and always go to a person first.
4. **A person approves, edits or rejects** the draft. Nothing is posted automatically.

**Limits to keep in mind**

- The data is synthetic, so the scores look better than real-world accuracy would.
- Alerts have false alarms by design: it is safer to look at an extra post than to miss a crisis.
- The tone guide and playbook are sample text for a fictional company.
- Gemini can be wrong, especially on sarcasm and mixed-language posts. That is why a person reads every draft.
""")

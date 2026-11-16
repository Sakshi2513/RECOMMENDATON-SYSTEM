"""Repair Recommendation: Streamlit app.  Run:  streamlit run app.py

Settings (environment variables, or Streamlit secrets on Streamlit Cloud):
  APP_PASSWORD     shared password. Required on Streamlit Cloud: without it the app refuses to open.
  ADMIN_PASSWORD   optional; enables the Administrator panel (download of the correction log)
  APP_DATA_DIR     folder for the correction database
"""

import hmac
import json
import random
import time

import streamlit as st

from recsys import config
from recsys.actions import LABEL, STEP
from recsys.classifier import load_model
from recsys.storage import CorrectionLog

PASSWORD = config.setting("APP_PASSWORD")
ADMIN_PASSWORD = config.setting("ADMIN_PASSWORD")

st.set_page_config(page_title="Repair Recommendation", layout="centered", initial_sidebar_state="collapsed")
st.markdown(
    """
<style>
  .block-container {max-width: 980px; padding-top: 2.2rem;}
  #MainMenu, footer, header [data-testid="stToolbar"] {visibility: hidden;}
  h1 {font-weight: 650; letter-spacing: -0.01em; margin-bottom: 0.2rem;}
  h3 {font-weight: 600; margin-top: 1.6rem;}
  .sub {color: #6b7280; margin-bottom: 1.4rem; font-size: 0.98rem;}
  .note {color: #6b7280; font-size: 0.82rem; margin-top: 1.8rem; border-top: 1px solid #e5e7eb; padding-top: 0.8rem;}
  div[data-testid="stTextArea"] textarea {font-size: 0.95rem;}
  div.stButton > button {border-radius: 6px; font-weight: 550;}
</style>
""",
    unsafe_allow_html=True,
)


def require_login():
    """Stop the page until the shared password is entered. Never run open on a public host."""
    if config.on_streamlit_cloud() and not PASSWORD:
        st.error("This app is not configured. The administrator must set APP_PASSWORD in the app's secrets.")
        st.stop()
    if not PASSWORD or st.session_state.get("authed"):
        return
    st.title("Repair Recommendation")
    st.markdown('<div class="sub">Sign in to continue.</div>', unsafe_allow_html=True)
    with st.form("login"):
        password = st.text_input("Password", type="password")
        if st.form_submit_button("Sign in", type="primary"):
            if hmac.compare_digest(password.encode(), PASSWORD.encode()):
                st.session_state["authed"] = True
                st.rerun()
            else:
                time.sleep(1)  # slow down guessing
                st.error("Incorrect password.")
    st.stop()


@st.cache_resource(show_spinner="Loading model...")
def get_model():
    return load_model(config.MODEL_PATH, config.TYPO_PATH)


@st.cache_resource
def get_log():
    return CorrectionLog(config.data_dir())


@st.cache_data
def get_examples():
    return json.loads(config.EXAMPLES_PATH.read_text(encoding="utf-8"))


def fill_example():
    example = random.choice(get_examples())
    st.session_state["neg"], st.session_state["pos"] = example["negative"], example["positive"]
    st.session_state.pop("result", None)


def render_actions(model, result):
    """Checklist of suggested actions; returns the ones currently ticked."""
    chosen = []

    def row(action, score):
        threshold = model.thresholds[action]
        left, right = st.columns([2, 3], vertical_alignment="center")
        if left.checkbox(LABEL[action], value=score >= threshold, key=f"act_{result['id']}_{action}"):
            chosen.append(action)
        right.progress(min(score, 1.0), text=f"{score:.0%}  ·  suggested from {threshold:.0%}")

    ranked = model.ranked(result["scores"])
    st.markdown("### Suggested repair actions")
    for action, score in ranked[:5]:
        row(action, score)
    with st.expander(f"Other actions ({len(ranked) - 5})"):
        for action, score in ranked[5:]:
            row(action, score)
    if not chosen:
        st.info("No action reached its threshold. Select actions manually if required.")
    return chosen


def render_recommendation(model, result, chosen):
    st.markdown("### Recommendation")
    default = "\n".join(f"{i}. {STEP[a]}" for i, a in enumerate(chosen, 1))
    final = st.text_area(
        "Recommendation text (editable)",
        value=default,
        height=170,
        max_chars=config.MAX_CHARS,
        key=f"final_{result['id']}_{'_'.join(sorted(chosen))}",
        label_visibility="collapsed",
    )
    surveyor = st.text_input("Surveyor name or initials (optional)", key="surveyor", max_chars=60)
    col_a, col_b, _ = st.columns([1.2, 1.6, 3])
    for col, label, status in ((col_a, "Approve", "accepted"), (col_b, "Save with edits", "edited")):
        if col.button(label, key=f"{status}_{result['id']}"):
            try:
                get_log().add(
                    surveyor=surveyor,
                    negative=result["neg"],
                    positive=result["pos"],
                    actions=chosen,
                    shown_text=default,
                    final_text=default if status == "accepted" else final,
                    status=status,
                    model_version=model.version,
                )
                st.success("Saved to the correction log.")
            except Exception:
                st.error("Could not save. Please copy the text and tell the administrator.")


def render_admin():
    with st.expander("Administrator"):
        password = st.text_input("Administrator password", type="password", key="admin_pw")
        if password and hmac.compare_digest(password.encode(), ADMIN_PASSWORD.encode()):
            log = get_log()
            st.write(f"{log.count()} saved corrections.")
            if config.on_streamlit_cloud():
                st.warning(
                    "On Streamlit Cloud this log is erased whenever the app restarts or is redeployed. "
                    "Download it regularly."
                )
            st.download_button("Download corrections (CSV)", log.to_csv(), file_name="corrections.csv", mime="text/csv")
        elif password:
            st.error("Incorrect password.")


def main():
    require_login()
    model = get_model()

    st.title("Repair Recommendation")
    st.markdown(
        '<div class="sub">Enter the observations for an impacted area to get suggested repair actions. '
        "Review and edit the suggestion before it is used.</div>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2, gap="large")
    neg = col1.text_area(
        "Negative side (where the damage shows)",
        key="neg",
        height=140,
        max_chars=config.MAX_CHARS,
        placeholder="e.g. Skirting level dampness in the master bedroom",
    )
    pos = col2.text_area(
        "Positive side (suspected source)",
        key="pos",
        height=140,
        max_chars=config.MAX_CHARS,
        placeholder="e.g. Tile joints open and hollowness in the common bathroom, gaps around the nahani trap",
    )
    st.caption("Do not enter customer names, phone numbers or addresses.")

    col_go, col_example, _ = st.columns([2, 1.4, 3])
    go = col_go.button("Generate recommendation", type="primary")
    col_example.button("Load example", on_click=fill_example)

    if go:
        if not (neg.strip() or pos.strip()):
            st.warning("Enter at least one description.")
        else:
            st.session_state["result"] = {
                "id": time.time_ns(),
                "neg": neg,
                "pos": pos,
                "scores": model.predict(neg, pos),
            }

    result = st.session_state.get("result")
    if result:
        chosen = render_actions(model, result)
        render_recommendation(model, result, chosen)

    st.markdown(
        '<div class="note">Suggestions come from a model trained on past inspection reports. '
        "They are not a substitute for the surveyor's judgement. Approved and edited versions are logged "
        "(phone numbers and e-mail addresses are removed) to improve the model.</div>",
        unsafe_allow_html=True,
    )
    if ADMIN_PASSWORD:
        render_admin()


main()

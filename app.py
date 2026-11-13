import pandas as pd
import streamlit as st

from config import DATA_PATH, MODELS_DIR, OUTPUTS_DIR, TARGET_COLUMN
from src.evaluate import evaluate_models
from src.predict import batch_predict, predict_record
from src.preprocessing import dataset_summary, load_data, make_split, prepare_features
from src.train import save_models, train_models
from src.url_analysis import analyze_url
from src.visualization import (class_distribution, confusion_matrix_plot, correlation_heatmap,
                               feature_distribution, feature_importance, model_comparison, roc_plot)

st.set_page_config(page_title="PhishGuard AI", page_icon="🛡️", layout="wide")
st.markdown("""<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
  :root { --dp-color-ground:#07090e; --dp-color-ground-tint:#101522; --dp-color-panel:#10141c; --dp-color-accent:#ff951f; --dp-color-value:#edcd8c; --dp-color-ink:#f4f5f7; --dp-color-ink-muted:#c9ced8; --dp-color-line:#303139; }
  .stApp { background:var(--dp-color-ground); color:var(--dp-color-ink); font-family:'IBM Plex Sans','Financial Plex Sans',sans-serif; }
  header[data-testid='stHeader'], [data-testid='stToolbar'], [data-testid='stAppDeployButton'], .stAppDeployButton { display:none!important; }
  .block-container { padding-top:2.5rem; }
  [data-testid='stSidebar'] { background:#0c0f16; border-right:1px solid var(--dp-color-line); }
  [data-testid='stSidebar'] .stRadio label { border-radius:0; padding:8px 8px; color:#c9ced8; border-left:2px solid transparent; }
  [data-testid='stSidebar'] .stRadio label:hover { color:#f4f5f7; background:#151a24; border-left-color:#edcd8c; }
  [data-testid='stSidebar'] [data-testid='stAlert'] { background:#1a1712; border:1px solid #5b4829; color:#edcd8c; border-radius:5px; }
  h1,h2,h3 { color:var(--dp-color-ink)!important; letter-spacing:-.03em; } h1 { font-weight:600!important; } p, li { color:var(--dp-color-ink-muted); line-height:1.6; }
  .hero { position:relative; display:grid; grid-template-columns:minmax(0,1.35fr) minmax(230px,.65fr); align-items:center; gap:34px; padding:34px; border:1px solid var(--dp-color-line); border-radius:5px; background:#10141c; margin:0 0 24px; overflow:visible; }
  .hero:before,.hero:after { content:''; position:absolute; inset:8px -8px -8px 8px; border:1px solid #3e3324; border-radius:5px; z-index:-1; } .hero:after { inset:15px -15px -15px 15px; border-color:#2d2923; }
  .eyebrow { color:var(--dp-color-accent); font-size:.74rem; font-weight:700; text-transform:uppercase; letter-spacing:.14em; margin-bottom:10px; }
  .hero-title { font-size:clamp(2.2rem,4vw,4.6rem); color:var(--dp-color-ink); font-weight:600; line-height:.98; letter-spacing:-.05em; margin:0 0 13px; max-width:760px; }
  .hero-copy { max-width:680px; color:var(--dp-color-ink-muted); font-size:1rem; margin:0; }
  .hero-object { position:relative; background:radial-gradient(circle at 55% 110%,rgba(255,149,31,.26),transparent 58%),#15171b; border:1px solid #5a4a32; border-radius:5px; padding:22px; min-height:166px; display:flex; flex-direction:column; justify-content:space-between; box-shadow:0 20px 35px rgba(0,0,0,.32); }
  .object-label { color:#c9ced8; font-size:.7rem; letter-spacing:.13em; text-transform:uppercase; font-weight:600; }.object-value { color:var(--dp-color-value); font-family:'IBM Plex Mono','Financial Plex Mono',monospace; font-size:clamp(1.7rem,3vw,2.7rem); line-height:1; font-weight:500; font-variant-numeric:tabular-nums; }.object-note { color:#a9abb1; font-family:'IBM Plex Mono','Financial Plex Mono',monospace; font-size:.72rem; }
  .kpi { background:var(--dp-color-panel); border:1px solid var(--dp-color-line); border-radius:5px; padding:17px 18px; margin:5px 0; min-height:98px; }
  .kpi p { margin:0; color:#aeb4bf; font-size:.72rem; text-transform:uppercase; letter-spacing:.1em; font-weight:600; }.kpi h2 { margin:10px 0 0; color:var(--dp-color-value); font-family:'IBM Plex Mono','Financial Plex Mono',monospace; font-size:1.35rem; font-variant-numeric:tabular-nums; }
  .section-label { color:var(--dp-color-accent); text-transform:uppercase; letter-spacing:.12em; font-size:.72rem; font-weight:700; margin:32px 0 3px; }.section-copy { color:#b9bec7; margin:0 0 14px; }
  .risk { display:inline-block; padding:6px 13px; border-radius:40px; font-family:'IBM Plex Mono','Financial Plex Mono',monospace; font-size:.78rem; font-weight:600; letter-spacing:.04em; margin-bottom:10px; }.risk-high { background:#311719; color:#ffb0a9; border:1px solid #733b34; }.risk-medium { background:#312715; color:#f3cf88; border:1px solid #6c562b; }.risk-low { background:#142a24; color:#aee2cb; border:1px solid #326956; }
  [data-testid='stDataFrame'] { border:1px solid var(--dp-color-line); border-radius:5px; } .stButton>button, .stDownloadButton>button { border-radius:40px; border:1px solid var(--dp-color-accent); background:var(--dp-color-accent); color:#16120c; font-weight:700; padding:.45rem 1rem; transition:background 160ms ease,border-color 160ms ease; }.stButton>button:hover,.stDownloadButton>button:hover { background:#f5ac4a; border-color:#f5ac4a; color:#16120c; }
  .stButton>button:focus-visible,.stDownloadButton>button:focus-visible,button:focus-visible { outline:2px solid var(--dp-color-value)!important; outline-offset:3px; } [data-testid='stExpander'] { border:1px solid var(--dp-color-line); border-radius:5px; background:var(--dp-color-panel); }
  @media (max-width:800px) { .hero { grid-template-columns:1fr; gap:20px; padding:25px; }.hero:after { inset:8px -8px -8px 8px; }.hero:before { display:none; } .hero-title { font-size:2.5rem; } }
  @media (prefers-reduced-motion:reduce) { * { transition:none!important; } }
</style>""", unsafe_allow_html=True)

@st.cache_data
def get_data():
    return load_data(DATA_PATH)

@st.cache_resource
def get_experiment():
    data = get_data()
    X_train, X_test, y_train, y_test, constants = make_split(data)
    models = train_models(X_train, y_train)
    metrics, details = evaluate_models(models, X_test, y_test)
    return models, metrics, details, X_train.columns.tolist(), constants

def kpi(title, value):
    st.markdown(f"<div class='kpi'><p>{title}</p><h2>{value}</h2></div>", unsafe_allow_html=True)


def section(title, copy=None):
    st.markdown(f"<div class='section-label'>{title}</div>", unsafe_allow_html=True)
    if copy:
        st.markdown(f"<p class='section-copy'>{copy}</p>", unsafe_allow_html=True)


def hero(title, copy, eyebrow="PhishGuard AI | Security analytics", value=None, label="MODEL SIGNAL", note="HELD-OUT EVALUATION"):
    object_html = "" if value is None else f"<div class='hero-object'><div class='object-label'>{label}</div><div class='object-value'>{value}</div><div class='object-note'>{note}</div></div>"
    st.markdown(f"<div class='hero'><div><div class='eyebrow'>{eyebrow}</div><div class='hero-title'>{title}</div><p class='hero-copy'>{copy}</p></div>{object_html}</div>", unsafe_allow_html=True)

try:
    data = get_data()
except FileNotFoundError:
    st.error(f"Dataset not found at {DATA_PATH}. Copy the supplied CSV into the data folder.")
    st.stop()

models, metrics, details, features, constants = get_experiment()
best_name = metrics.iloc[0]["Model"]
best_model = models[best_name]
summary = dataset_summary(data)

st.sidebar.title("🛡️ PhishGuard AI")
st.sidebar.caption("Threat intelligence workspace")
st.sidebar.markdown("---")
page = st.sidebar.radio("WORKSPACE", ["Dashboard", "Dataset Explorer", "Exploratory Data Analysis", "Model Training & Evaluation", "Phishing Detection", "About Project"])
st.sidebar.markdown("---")
st.sidebar.caption("Model status")
st.sidebar.success(f"{best_name} active")
st.sidebar.caption("Prediction only uses pre-extracted dataset features.")

if page == "Dashboard":
    hero("Detect website risk with confidence.", "A decision-support workspace that classifies websites from extracted URL and webpage characteristics. It does not crawl or scan live URLs.", value=f"{metrics.iloc[0]['F1-score']:.2%}", label="BEST MODEL F1 SCORE", note=f"{best_name.upper()} | HELD-OUT TEST SET")
    a, b, c, d = st.columns(4)
    with a: kpi("Dataset records", f"{summary['rows']:,}")
    with b: kpi("Legitimate websites", f"{summary['class_counts'].get(0, 0):,}")
    with c: kpi("Phishing websites", f"{summary['class_counts'].get(1, 0):,}")
    with d: kpi("Best actual model", best_name)
    section("Model intelligence", "Comparing all three classifiers on the same held-out test split.")
    st.plotly_chart(model_comparison(metrics), use_container_width=True)
    st.info("The best model is selected from the held-out test-set F1-score. All displayed metrics come from this run, not hardcoded results.")

elif page == "Dataset Explorer":
    hero("Dataset Explorer", "Inspect the 10,000 labelled website records that power this demonstration.", "Data intelligence", value=f"{summary['rows']:,}", label="LABELLED RECORDS", note="BALANCED BINARY DATASET")
    st.dataframe(data.head(100), use_container_width=True, height=360)
    a, b, c, d = st.columns(4)
    with a: kpi("Shape", f"{data.shape[0]} × {data.shape[1]}")
    with b: kpi("Missing values", summary["missing_values"])
    with c: kpi("Duplicate rows", summary["duplicate_rows"])
    with d: kpi("Constant features", ", ".join(constants) or "None")
    st.plotly_chart(class_distribution(data), use_container_width=True)
    section("Column information", "Feature names, data types, and variation across the supplied dataset.")
    st.dataframe(pd.DataFrame({"Feature": data.columns, "Data type": data.dtypes.astype(str).values, "Unique values": data.nunique().values}), use_container_width=True)

elif page == "Exploratory Data Analysis":
    hero("Exploratory Data Analysis", "Find patterns that distinguish legitimate websites from phishing websites.", "Pattern discovery", value=f"{len(features)}", label="ACTIVE FEATURES", note="CONSTANT FEATURE REMOVED")
    st.plotly_chart(correlation_heatmap(data, features), use_container_width=True)
    options = [x for x in ["UrlLength", "NumDots", "PctExtHyperlinks", "NumNumericChars", "PctExtResourceUrls"] if x in features]
    selected = st.selectbox("Select a numerical feature", options)
    st.plotly_chart(feature_distribution(data, selected), use_container_width=True)
    st.plotly_chart(feature_importance(models["Random Forest"], features), use_container_width=True)
    st.caption("Feature importance measures contribution to this fitted model; it does not demonstrate causal relationships.")

elif page == "Model Training & Evaluation":
    hero("Model Training & Evaluation", "Actual predictions, transparent comparisons, and saved model artifacts.", "Validation lab", value=f"{metrics.iloc[0]['ROC-AUC']:.4f}", label="BEST ROC-AUC", note="ACTUAL TEST-SET PREDICTIONS")
    st.write("All three classifiers use the same stratified 80:20 split (random_state=42). Scaling is confined to Logistic Regression's pipeline, preventing test-data leakage.")
    if st.button("Save trained models to disk"):
        save_models(models, features, constants, MODELS_DIR)
        OUTPUTS_DIR.mkdir(exist_ok=True)
        metrics.to_csv(OUTPUTS_DIR / "model_metrics.csv", index=False)
        st.success("Models and actual evaluation metrics saved.")
    formatted = metrics.copy()
    for col in formatted.columns[1:]: formatted[col] = formatted[col].map("{:.4f}".format)
    st.dataframe(formatted, use_container_width=True)
    st.plotly_chart(model_comparison(metrics), use_container_width=True)
    st.plotly_chart(roc_plot(details), use_container_width=True)
    for name, item in details.items():
        with st.expander(f"{name}: confusion matrix and classification report"):
            st.plotly_chart(confusion_matrix_plot(item["confusion_matrix"], name), use_container_width=True)
            st.dataframe(pd.DataFrame(item["report"]).transpose().round(4), use_container_width=True)
    st.info("False positives are legitimate sites incorrectly flagged, which can disrupt users. False negatives are phishing sites missed by the model, creating a security risk.")

elif page == "Phishing Detection":
    hero("Phishing Detection", f"Run a record-level or batch assessment using the current best model: {best_name}.", "Risk assessment", value=best_name, label="ACTIVE MODEL", note="CLASSIFICATION DECISION SUPPORT")
    method = st.radio("Prediction method", ["Single existing record", "Batch CSV upload", "URL triage (experimental)"], horizontal=True)
    if method == "Single existing record":
        row_id = st.selectbox("Choose a dataset record", data["id"].tolist())
        record = data.loc[data["id"] == row_id].iloc[0]
        input_frame = record[features].to_frame().T
        st.dataframe(input_frame, use_container_width=True)
        label, probability, risk = predict_record(best_model, input_frame)
        risk_class = risk.lower()
        st.markdown(f"<span class='risk risk-{risk_class}'>{risk.upper()} RISK</span>", unsafe_allow_html=True)
        left, mid, right = st.columns(3)
        with left: kpi("Prediction", "Phishing" if label else "Legitimate")
        with mid: kpi("Phishing probability", f"{probability:.1%}")
        with right: kpi("Risk category", risk)
        st.write("The model found patterns in the extracted features that are more consistent with " + ("phishing websites." if label else "legitimate websites.") + " This is a decision-support result, not proof of website safety.")
    elif method == "Batch CSV upload":
        uploaded = st.file_uploader("Upload a CSV with the required feature columns", type="csv")
        if uploaded:
            try:
                result = batch_predict(best_model, pd.read_csv(uploaded), features)
                st.dataframe(result, use_container_width=True)
                csv = result.to_csv(index=False).encode("utf-8")
                st.download_button("Download predictions CSV", csv, "phishguard_predictions.csv", "text/csv")
            except Exception as exc:
                st.error(f"Could not process this file: {exc}")
    else:
        section("URL triage", "This local checker reads the URL text only. It does not open, crawl, or verify the linked website.")
        raw_url = st.text_input("Website URL", placeholder="https://example.com/login")
        if st.button("Analyse URL", type="primary"):
            try:
                result = analyze_url(raw_url)
                risk_class = result["risk"].lower()
                st.markdown(f"<span class='risk risk-{risk_class}'>{result['risk'].upper()} URL RISK</span>", unsafe_allow_html=True)
                one, two, three = st.columns(3)
                with one: kpi("URL risk score", f"{result['risk_score']:.0%}")
                with two: kpi("Host", result["host"])
                with three: kpi("Assessment", result["risk"])
                st.write(result["verdict"] + ".")
                if result["signals"]:
                    st.dataframe(pd.DataFrame(result["signals"])[["signal", "explanation"]], use_container_width=True, hide_index=True)
                else:
                    st.success("No suspicious lexical signals were found in the URL text.")
                st.warning("This is a transparent URL-pattern check, not the trained machine-learning prediction. A safe URL can still be malicious, and a flagged URL can still be legitimate.")
            except ValueError as exc:
                st.error(str(exc))

else:
    hero("About PhishGuard AI", "A beginner-friendly DMBI mini-project built as a practical cybersecurity decision-support prototype.", "Project brief", value="3", label="MODELS COMPARED", note="LOGISTIC | TREE | FOREST")
    st.markdown("""**Problem statement:** Phishing websites imitate trusted services to steal sensitive information.\n\n**Business objective:** Identify suspicious websites early to support allow, warn, or review decisions.\n\n**Data mining task:** Supervised binary classification: `0 = Legitimate`, `1 = Phishing`. The public Kaggle phishing dataset has 10,000 records and extracted website features.\n\n**Algorithms:** Logistic Regression is a transparent baseline; Decision Tree is interpretable and non-linear; Random Forest captures interactions and provides feature importance.\n\n**Technology:** Python, Streamlit, pandas, NumPy, scikit-learn, Plotly, Matplotlib and joblib.\n\n**Limitations:** The model uses historical, engineered features; it cannot inspect arbitrary live URLs. Results may not generalize to newer attacks or different real-world class proportions.\n\n**Future improvements:** External validation, newer data, cross-validation, calibrated thresholds, model monitoring, and a separate live feature-extraction service.""")

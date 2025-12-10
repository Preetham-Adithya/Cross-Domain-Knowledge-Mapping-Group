# mental_health_dashboard_admin_only.py
import streamlit as st
import pandas as pd
import numpy as np
import re
import tempfile
import networkx as nx
from pyvis.network import Network
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import spacy
from datetime import datetime
import os
import json

ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "Admin@123"

#ktPage config
st.set_page_config(page_title="Mental Health Detection", layout="wide")
#CSS
st.markdown("""
<style>

:root {
  --primary: #4f46e5;
  --secondary: #0ea5e9;
  --bg-light: #eef3ff;
  --card-bg: rgba(255,255,255,0.65);
  --glass: rgba(255,255,255,0.35);
}

/* GLOBAL APP */
.stApp {
    background: linear-gradient(135deg, #c7d2fe, #e0f2fe);
    background-attachment: fixed;
    font-family: 'Inter', sans-serif;
    color: #1e293b !important;
}

/* CARD (GLASS EFFECT) */
.card {
    background: var(--card-bg);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    padding: 20px;
    border-radius: 16px;
    box-shadow: 0 12px 28px rgba(0,0,0,0.12);
    transition: transform .15s ease;
    border: 1px solid rgba(255,255,255,0.45);
    margin-bottom: 12px;
}
.card:hover {
    transform: translateY(-3px);
}

/* TITLES */
h1, h2, h3, h4 {
    font-weight: 700 !important;
    color: #111827 !important;
}

/* INPUT FIELDS */
input, textarea, select {
    background: rgba(255,255,255,0.78) !important;
    padding: 10px 14px !important;
    border-radius: 10px !important;
    border: 1px solid #c7d2fe !important;
    transition: .2s;
}
input:focus, textarea:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 6px rgba(79,70,229,0.38);
}

/* BUTTONS */
.stButton > button {
    background: linear-gradient(135deg, var(--primary), var(--secondary)) !important;
    padding: 9px 20px !important;
    border-radius: 10px !important;
    color: white !important;
    border: none !important;
    font-weight: 600 !important;
    transition: .18s;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 18px rgba(0,0,0,0.18);
}

/* TABS */
.stTabs [data-baseweb="tab"] {
    font-size: 15px !important;
    font-weight: 700 !important;
    padding: 10px !important;
}

/* DATAFRAME */
.dataframe {
    background: white;
    border-radius: 12px;
    overflow: hidden;
}

/* PYVIS GRAPH */
#graph-container {
    border-radius: 12px;
    padding: 10px;
    background: white;
    box-shadow: 0 6px 16px rgba(0,0,0,0.12);
}

</style>
""", unsafe_allow_html=True)

#Session state initialization
if "users" not in st.session_state:
# precreate admin user for convenience (password checked on login)
    st.session_state.users = {ADMIN_USERNAME: ADMIN_PASSWORD}
if "auth" not in st.session_state:
    st.session_state.auth = False
if "current_user" not in st.session_state:
    st.session_state.current_user = None
if "df" not in st.session_state:
    st.session_state.df = None
if "df_name" not in st.session_state:
    st.session_state.df_name = None
if "vectorizer" not in st.session_state:
    st.session_state.vectorizer = None
if "model" not in st.session_state:
    st.session_state.model = None
if "trained" not in st.session_state:
    st.session_state.trained = False
if "model_classes" not in st.session_state:
    st.session_state.model_classes = []
if "kg" not in st.session_state:
    st.session_state.kg = nx.DiGraph()
if "pred_history" not in st.session_state:
    st.session_state.pred_history = []
if "feedback" not in st.session_state:
    st.session_state.feedback = []
if "feedback_key" not in st.session_state:
    st.session_state.feedback_key = 0
if "feedback_id" not in st.session_state:
    st.session_state.feedback_id = 1
if "pipeline_status" not in st.session_state:
    st.session_state.pipeline_status = {
        "ETL Pipeline": "Idle",
        "Graph Updater": "Idle"
    }

DISORDER_KB = {
    "depression": ["sadness", "loss of interest", "fatigue", "sleep issues", "low appetite", "hopelessness"],
    "anxiety": ["worry", "restlessness", "panic", "sleep issues", "irritability"],
    "bipolar": ["mania", "depression", "mood swings", "impulsivity"],
    "adhd": ["inattention", "hyperactivity", "impulsivity"],
    "ocd": ["obsessions", "compulsions", "anxiety"]
}

@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except Exception:
        return spacy.blank("en")

nlp = load_spacy_model()

def clean_text(s):
    if pd.isna(s):
        return ""
    s = str(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def add_dataset_to_session(df, name="dataset"):
    st.session_state.df = df
    st.session_state.df_name = name

def build_graph_from_dataset(sample_n=200):
    if st.session_state.df is None:
        return
    G = st.session_state.kg
    df_sample = st.session_state.df.sample(min(len(st.session_state.df), sample_n), random_state=1)

    text_props = {"color": "#60a5fa", "size": 15, "shape": "dot", "type": "text"}
    label_props = {"color": "#ef4444", "size": 30, "shape": "box", "type": "label"}
    concept_props = {"color": "#10b981", "size": 20, "shape": "circle", "type": "concept"}

    for _, row in df_sample.iterrows():
        t = clean_text(row["text"])[:200]
        l = str(row["label"])
        l_lower = l.lower()
        if t and not G.has_node(t):
            G.add_node(t, **text_props, title=t)
        if not G.has_node(l):
            G.add_node(l, **label_props, title=l)
        if t:
            G.add_edge(t, l, relation="CONTAINS")
        if l_lower in DISORDER_KB:
            for rel in DISORDER_KB[l_lower]:
                if not G.has_node(rel):
                    G.add_node(rel, **concept_props, title=rel)
                G.add_edge(l, rel, relation="HAS_SYMPTOM")
    st.session_state.kg = G


def show_pyvis(G, height=650):
    net = Network(
        height=f"{height}px",
        width="100%",
        directed=True,
        bgcolor="#ffffff",
        font_color="#1e293b"
    )

 
    net.barnes_hut(
        gravity=-40000,
        central_gravity=0.001,
        spring_length=150,
        spring_strength=0.009,
        damping=0.8,
        overlap=0.2
    )


    for node, data in G.nodes(data=True):
        color = data.get("color", "#60a5fa")
        size = data.get("size", 18)
        shape = data.get("shape", "dot")
        title = data.get("title", str(node))
        border = "#1e40af"

        net.add_node(
            node,
            label=str(node),
            title=title,
            color={"background": color, "border": border},
            borderWidth=2,
            shape=shape,
            size=size
        )


    for u, v, attr in G.edges(data=True):
        net.add_edge(
            u,
            v,
            title=attr.get("relation", ""),
            color="#64748b",
            width=2,
            smooth=True
        )

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    net.save_graph(tmp.name)
    return tmp.name

# Model Training 
def train_and_store(df):
    X = df["text"].astype(str)
    y = df["label"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vect = TfidfVectorizer(stop_words="english", max_features=5000)
    X_train_vec = vect.fit_transform(X_train)
    X_test_vec = vect.transform(X_test)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_vec, y_train)

    y_pred = clf.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)

    st.session_state.vectorizer = vect
    st.session_state.model = clf
    st.session_state.trained = True
    st.session_state.model_classes = list(clf.classes_)

    fig, ax = plt.subplots(figsize=(6, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=False)
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    return acc, fig

#Prediction + Graph
def predict_and_update_graph(text):
    if not st.session_state.trained:
        return None

    vec = st.session_state.vectorizer.transform([text])
    pred = st.session_state.model.predict(vec)[0]

    G = st.session_state.kg
    text_node = clean_text(text)[:200]
    pred_lower = pred.lower()

    text_props = {"color": "#60a5fa", "size": 15, "shape": "dot", "type": "text", "title": text_node}
    label_props = {"color": "#ef4444", "size": 30, "shape": "box", "type": "label", "title": pred}
    concept_props = {"color": "#10b981", "size": 20, "shape": "circle", "type": "concept"}

    if not G.has_node(text_node):
        G.add_node(text_node, **text_props)
    if not G.has_node(pred):
        G.add_node(pred, **label_props)
    G.add_edge(text_node, pred, relation="PREDICTS")

    if pred_lower in DISORDER_KB:
        for rel in DISORDER_KB[pred_lower]:
            if not G.has_node(rel):
                G.add_node(rel, **concept_props, title=rel)
            G.add_edge(pred, rel, relation="HAS_SYMPTOM")

    st.session_state.kg = G

    st.session_state.pred_history.append({
        "user": st.session_state.current_user,
        "text": text,
        "predicted": pred,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

    return pred

#Triplet Extraction
def extract_triplets(text):
    doc = nlp(text)
    triplets = []
    for sent in doc.sents:
        subj = obj = verb = ""
        root = next((t for t in sent if t.dep_ == 'ROOT' or t.pos_ == 'VERB'), None)
        if not root:
            continue
        verb = root.lemma_
        subject_tokens = [t.text for t in sent if "subj" in t.dep_]
        subj = " ".join(subject_tokens)
        object_tokens = [t.text for t in sent if "obj" in t.dep_ or t.dep_ == 'attr' or t.dep_ == 'acomp']
        obj = " ".join(object_tokens)
        if subj and verb and obj:
            triplets.append((subj, verb, obj))
    return triplets

#Signup & Login UI 
def signup_ui():
    st.title("🛡️ Mental Health Dashboard")
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Create New Account")
    username = st.text_input("Username", key="signup_username")
    password = st.text_input("Password", type="password", key="signup_password")

    if st.button("Create Account", key="signup_create_btn"):
        if not username or not password:
            st.error("Enter username and password.")
            return
        if username == ADMIN_USERNAME:
            st.error("This username is reserved for the administrator.")
            return
        if username in st.session_state.users:
            st.error("Username exists.")
            return
        st.session_state.users[username] = password
        st.session_state.auth = True
        st.session_state.current_user = username
        st.success("Account created. Logged in.")
    st.markdown("</div>", unsafe_allow_html=True)

def login_ui():
    st.title("🛡️ Mental Health Dashboard")
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Login to Dashboard")
    username = st.text_input("Username", key="login_username")
    password = st.text_input("Password", type="password", key="login_password")

    if st.button("Login", key="login_btn"):
        # Admin login check: require ADMIN_PASSWORD
        if username == ADMIN_USERNAME:
            if password == ADMIN_PASSWORD:
                st.session_state.auth = True
                st.session_state.current_user = username
                st.success("Admin login successful.")
            else:
                st.error("Incorrect admin password.")
        else:
            if username in st.session_state.users and st.session_state.users[username] == password:
                st.session_state.auth = True
                st.session_state.current_user = username
                st.success("Welcome!")
            else:
                st.error("Invalid credentials.")
    st.markdown("</div>", unsafe_allow_html=True)

#Admin helpers 
def merge_nodes_in_graph(G, target, source):
    if source not in G or target not in G or source == target:
        return False, "Invalid nodes."
    for pred in list(G.predecessors(source)):
        if pred != target:
            existing = G.get_edge_data(pred, source) or {}
            G.add_edge(pred, target, **existing)
    for succ in list(G.successors(source)):
        if succ != target:
            existing = G.get_edge_data(source, succ) or {}
            G.add_edge(target, succ, **existing)
    if G.nodes[source]:
        for k, v in dict(G.nodes[source]).items():
            if k not in G.nodes[target]:
                G.nodes[target][k] = v
    G.remove_node(source)
    return True, f"Merged '{source}' into '{target}'."

def list_edges_with_attrs(G):
    rows = []
    for u, v, attrs in G.edges(data=True):
        rows.append({
            "source": u,
            "target": v,
            "relation": attrs.get("relation", ""),
            "confirmed": attrs.get("confirmed", False)
        })
    return pd.DataFrame(rows)

# Dashboard
def dashboard_ui():
    st.header("🧠 Mental Health Detection Dashboard")
    st.markdown("""
    <div style="font-size:18px; padding:6px 0; color:#4f46e5;">
    <b>AI-powered mental health text analysis and knowledge graph system</b>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("📥 Upload Your Dataset Here (CSV)")
    uploaded = st.file_uploader("Upload CSV", type=["csv"], key="upload_csv")

    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            df = df.iloc[:, :2]
            df.columns = ["text", "label"]
            df["text"] = df["text"].astype(str).apply(clean_text)
            df = df.dropna()
            add_dataset_to_session(df, name=uploaded.name)
            build_graph_from_dataset()
            st.success(f"Dataset uploaded successfully — *{len(df)}* rows")
        except Exception as e:
            st.error(f"Invalid CSV format: {e}")

    st.markdown("---")

    #Admin tab appears only for the admin user
    tab_names = [
        "📄 Dataset",
        "🔗 Triplet Extraction",
        "⚙️ Train & Detect",
        "📈 Charts",
        "🌐 Graph",
        "🔎 Semantic Search"
    ]

# Add admin tab only if the current user is admin
    if st.session_state.current_user == ADMIN_USERNAME:
        tab_names.append("💻 Admin")

    tab_names.append("💬 Feedback")

    tabs = st.tabs(tab_names)

    # Mapping for convenience (find index of a tab name)
    def tab_index(name):
        try:
            return tab_names.index(name)
        except ValueError:
            return None

    # Dataset Tab
    idx = tab_index("📄 Dataset")
    with tabs[idx]:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Dataset Preview")
        if st.session_state.df is None:
            st.info("Upload a dataset first.")
        else:
            st.markdown(f"*Current Dataset:* {st.session_state.df_name}")
            rows_to_show = st.slider("Preview rows", 5, 200, 20, key="dataset_preview_rows")
            st.dataframe(st.session_state.df.head(rows_to_show))
        st.markdown("</div>", unsafe_allow_html=True)

    # Triplet Extraction
    idx = tab_index("🔗 Triplet Extraction")
    with tabs[idx]:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("NLP Triplet Extraction")
        txt = st.text_area("Enter text for relationship extraction (Subject-Verb-Object)", key="triplet_text_input", height=140)
        if st.button("Extract Triplets", key="extract_triplets_btn"):
            if txt.strip():
                result = extract_triplets(txt)
                if result:
                    st.success(f"Found {len(result)} triplets!")
                    st.dataframe(pd.DataFrame(result, columns=["subject", "relation", "object"]))
                else:
                    st.info("No triplets found.")
        st.markdown("</div>", unsafe_allow_html=True)

    # Train & Detect
    idx = tab_index("⚙️ Train & Detect")
    with tabs[idx]:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Model Training")
        if st.button("Start Training (Logistic Regression)", key="train_model_btn"):
            if st.session_state.df is None:
                st.error("Upload dataset first.")
            else:
                try:
                    acc, fig = train_and_store(st.session_state.df)
                    st.success(f"Model trained successfully! Test Accuracy: *{acc:.3f}*")
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Training failed: {e}")

        st.subheader("Real-time Prediction")
        text = st.text_area("Enter text to predict mental health category", key="dashboard_text_input", height=140)
        if st.button("Predict Disorder Category", key="predict_disorder_btn"):
            if not st.session_state.trained:
                st.error("Train model first.")
            else:
                if not text.strip():
                    st.warning("Enter text to predict.")
                else:
                    try:
                        pred = predict_and_update_graph(text)
                        if pred:
                            st.success(f"Model Prediction: *{pred}*")
                        else:
                            st.error("Prediction failed.")
                    except Exception as e:
                        st.error(f"Prediction error: {e}")

        st.markdown("---")
        st.markdown("*Prediction History (latest 50)*")
        if st.session_state.pred_history:
            st.dataframe(pd.DataFrame(list(reversed(st.session_state.pred_history))).head(50))
        else:
            st.info("No prediction history yet.")
        st.markdown("</div>", unsafe_allow_html=True)

    # Charts
    idx = tab_index("📈 Charts")
    with tabs[idx]:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Data Visualization")
        if st.session_state.df is None:
            st.info("Upload dataset first.")
        else:
            st.markdown("### Category Distribution (Count)")
            counts = st.session_state.df["label"].value_counts()
            st.bar_chart(counts)

            st.markdown("### Category Distribution (Percentage)")
            figp, axp = plt.subplots()
            axp.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90)
            axp.axis("equal")
            st.pyplot(figp)
        st.markdown("</div>", unsafe_allow_html=True)

    # Graph
    idx = tab_index("🌐 Graph")
    with tabs[idx]:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Knowledge Graph Visualization (Interactive)")
        G = st.session_state.kg
        st.info(f"Current Graph Status: *{len(G.nodes()):,} Nodes, *{len(G.edges()):,} Edges**")

        st.markdown("> The visualization is expressive: **Red Boxes** are Categories/Disorders, **Light Blue Dots** are Text Snippets, and **Emerald Green Circles** represent Symptoms/Concepts.")
        if st.button("Generate & Show Full Graph", key="show_full_graph_btn"):
            if len(G.nodes()) == 0:
                st.info("Graph empty. Upload a dataset and sample it first.")
            else:
                html = show_pyvis(G)
                st.components.v1.html(open(html, "r", encoding="utf-8").read(), height=650, scrolling=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Semantic Search
    idx = tab_index("🔎 Semantic Search")
    with tabs[idx]:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Graph-based Semantic Search & Subgraph Generation")
        method = st.radio("Search method", ["Keyword (match node text)", "Semantic (model similarity)"], index=0, key="semantic_method_radio")
        query = st.text_input("Search text / query", key="semantic_search_input")
        topk = st.slider("Top K (for semantic)", 1, 10, 3, key="semantic_topk_slider")
        expand_depth = st.slider("Subgraph neighbor expansion depth", 0, 2, 1, key="semantic_expand_depth")

        if st.button("Search & Visualize Subgraph", key="semantic_search_btn"):
            if method == "Semantic (model similarity)" and not st.session_state.trained:
                st.error("Train model first for semantic similarity.")
            elif not query.strip():
                st.warning("Enter query text.")
            else:
                matched_nodes = []
                G = st.session_state.kg
                if method == "Keyword (match node text)":
                    matched_nodes = [n for n in G.nodes() if query.lower() in str(n).lower()]
                    if len(matched_nodes) == 0 and st.session_state.df is not None:
                        matched_rows = st.session_state.df[st.session_state.df["text"].str.contains(query, case=False, na=False)]
                        matched_nodes = list(matched_rows["text"].head(topk).values)
                else:
                    vecq = st.session_state.vectorizer.transform([query])
                    docs = st.session_state.vectorizer.transform(st.session_state.df["text"])
                    sims = (vecq @ docs.T).toarray().flatten()
                    idxs = sims.argsort()[-topk:][::-1]
                    matched_nodes = list(st.session_state.df.iloc[idxs]["text"].values)

                if not matched_nodes:
                    st.info("No matches found.")
                else:
                    st.success(f"Found *{len(matched_nodes)}* matching seed nodes.")
                    nodeset = set()
                    for n in matched_nodes:
                        if n in G:
                            nodeset.add(n)
                            if expand_depth >= 1:
                                nodeset.update(G.predecessors(n))
                                nodeset.update(G.successors(n))
                            if expand_depth >= 2:
                                for nb in list(G.predecessors(n)) + list(G.successors(n)):
                                    nodeset.update(n_n for n_n in G.predecessors(nb) if n_n in G)
                                    nodeset.update(n_n for n_n in G.successors(nb) if n_n in G)
                        else:
                            nodeset.add(n)
                            if st.session_state.df is not None:
                                row = st.session_state.df[st.session_state.df["text"] == n]
                                if not row.empty:
                                    lbl = str(row.iloc[0]["label"])
                                    nodeset.add(lbl)

                    valid_nodes_for_subgraph = [x for x in nodeset if x in G.nodes()]
                    sub = G.subgraph(valid_nodes_for_subgraph).copy()

                    if sub.number_of_nodes() == 0:
                        st.info("Subgraph is empty. No nodes found in the knowledge graph matching your query.")
                    else:
                        html = show_pyvis(sub, height=600)
                        st.components.v1.html(open(html, "r", encoding="utf-8").read(), height=600)
                        st.write(f"Subgraph size: *{len(sub.nodes())} nodes, *{len(sub.edges())} edges**")
        st.markdown("</div>", unsafe_allow_html=True)

    # ADMIN tab - only render if admin is logged in
    if st.session_state.current_user == ADMIN_USERNAME:
        idx = tab_index("💻 Admin")
        with tabs[idx]:
            # Double-check access server-side
            if st.session_state.current_user != ADMIN_USERNAME:
                st.error("Access denied. Admin only.")
                st.stop()

            st.subheader("Admin Dashboard — Monitor pipelines, datasets & refine graph")

            admin_tabs = st.tabs([
                "📊 Pipelines",
                "📁 Dataset Viewer",
                "🔧 Graph Refinement",
                "📚 Edge List & Confirm",
                "🛠 Advanced Tools",
                "📨 User Feedback"
            ])

            # 1. Pipelines
            with admin_tabs[0]:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.write("### Processing Pipelines Overview")
                for pname, status in st.session_state.pipeline_status.items():
                    color = "green" if status == "Running" else "red" if status == "Stopped" else "blue"
                    st.markdown(f"<b>{pname}</b> — <span style='color:{color}; font-weight:600;'>{status}</span>", unsafe_allow_html=True)
                colp1, colp2, colp3 = st.columns(3)
                if colp1.button("Start ETL Pipeline", key="admin_start_etl_btn"):
                    st.session_state.pipeline_status["ETL Pipeline"] = "Running"
                    st.success("ETL Pipeline marked as Running (simulated).")
                if colp2.button("Stop ETL Pipeline", key="admin_stop_etl_btn"):
                    st.session_state.pipeline_status["ETL Pipeline"] = "Stopped"
                    st.warning("ETL Pipeline marked as Stopped (simulated).")
                if colp3.button("Refresh Status", key="admin_refresh_pipelines_btn"):
                    st.session_state.pipeline_status["Graph Updater"] = "Idle"
                    st.success("Pipeline statuses refreshed (simulated).")
                st.markdown("</div>", unsafe_allow_html=True)

            # 2. Dataset Viewer
            with admin_tabs[1]:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.write("### Dataset Viewer")
                if st.session_state.df is None:
                    st.info("No dataset uploaded.")
                else:
                    st.write(f"*Dataset:* {st.session_state.df_name} — Rows: {len(st.session_state.df)}")
                    search_text = st.text_input("Search rows", placeholder="Type to filter dataset", key="admin_search_text")
                    df_filtered = st.session_state.df[
                        st.session_state.df.apply(lambda row: row.astype(str).str.contains(search_text, case=False).any(), axis=1)
                    ] if search_text else st.session_state.df
                    n_preview = st.number_input("Rows to preview", min_value=1, max_value=min(1000, len(df_filtered)), value=20, key="admin_dataset_preview_rows")
                    st.dataframe(df_filtered.head(n_preview))
                    if st.button("Download full CSV", key="admin_download_dataset_btn"):
                        csv_bytes = st.session_state.df.to_csv(index=False).encode("utf-8")
                        st.download_button("Click to download", data=csv_bytes, file_name=f"{st.session_state.df_name}", mime="text/csv", key="admin_download_btn_inner")
                st.markdown("</div>", unsafe_allow_html=True)

            # 3. Graph Refinement
            with admin_tabs[2]:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.write("### Graph Refinement Tools")
                G = st.session_state.kg
                if len(G.nodes()) == 0:
                    st.info("Graph empty. Build graph from dataset or predictions first.")
                else:
                    all_nodes = sorted(list(G.nodes()))
                    colA, colB = st.columns(2)
                    with colA:
                        node_source = st.selectbox("Node to merge (source)", all_nodes, key="merge_source_select")
                    with colB:
                        node_target = st.selectbox("Merge into (target)", all_nodes, key="merge_target_select")
                    if st.button("Merge Selected Nodes", key="admin_merge_nodes_btn"):
                        ok, msg = merge_nodes_in_graph(G, node_target, node_source)
                        if ok:
                            st.session_state.kg = G
                            st.success(msg)
                        else:
                            st.error(msg)

                    st.markdown("---")
                    st.write("#### Rename Node")
                    edit_node = st.selectbox("Select node", all_nodes, key="admin_edit_node_select")
                    new_label = st.text_input("New label", key="admin_new_label_input")
                    if st.button("Rename Node", key="admin_rename_node_btn"):
                        if not new_label.strip():
                            st.warning("Enter a new label.")
                        else:
                            try:
                                nx.relabel_nodes(G, {edit_node: new_label}, copy=False)
                                st.success(f"Renamed '{edit_node}' to '{new_label}'.")
                                st.session_state.kg = G
                            except Exception as e:
                                st.error(f"Rename failed: {e}")

                    st.markdown("---")
                    st.write("#### Delete Node")
                    del_node = st.selectbox("Select node to delete", all_nodes, key="admin_delete_node_select")
                    if st.button("Delete Node", key="admin_delete_node_btn"):
                        try:
                            G.remove_node(del_node)
                            st.success(f"Deleted node '{del_node}'.")
                            st.session_state.kg = G
                        except Exception as e:
                            st.error(f"Deletion failed: {e}")

                    st.markdown("---")
                    if st.button("Show Graph Preview", key="admin_graph_preview_btn"):
                        html = show_pyvis(G, height=500)
                        st.components.v1.html(open(html, "r", encoding="utf-8").read(), height=500)
                st.markdown("</div>", unsafe_allow_html=True)

            # 4. Edge List Editor
            with admin_tabs[3]:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.write("### Edge list & Confirm relations")
                G = st.session_state.kg
                if len(G.edges()) == 0:
                    st.info("No edges in graph.")
                else:
                    edges_df = list_edges_with_attrs(G)
                    st.dataframe(edges_df)
                    edge_choices = [f"{r['source']} -> {r['target']} (relation={r['relation']}, confirmed={r['confirmed']})" for _, r in edges_df.iterrows()]
                    sel_edge = st.selectbox("Select an edge", [""] + edge_choices, key="admin_select_edge")
                    if sel_edge:
                        left = sel_edge.split(" -> ")[0]
                        right = sel_edge.split(" -> ")[1].split(" (")[0]
                        cur_relation = G.get_edge_data(left, right).get("relation", "")
                        cur_confirmed = G.get_edge_data(left, right).get("confirmed", False)
                        st.write(f"Current relation: '{cur_relation}', confirmed: {cur_confirmed}")
                        new_relation = st.text_input("Edit relation", value=cur_relation, key="admin_edit_relation_input")
                        if st.button("Confirm / Update relation", key="admin_confirm_relation_btn"):
                            G[left][right]["relation"] = new_relation
                            G[left][right]["confirmed"] = True
                            st.success(f"Edge {left} -> {right} updated and confirmed.")
                            st.session_state.kg = G
                st.markdown("</div>", unsafe_allow_html=True)

            # 5. Advanced Tools
            with admin_tabs[4]:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.write("### Advanced Graph & Data Tools")
                colA, colB = st.columns(2)
                with colA:
                    if st.button("Export Graph as JSON", key="admin_export_json_btn"):
                        graph_json = nx.readwrite.json_graph.node_link_data(st.session_state.kg)
                        st.download_button("Download Graph JSON", json.dumps(graph_json, indent=2), file_name="knowledge_graph.json", mime="application/json")
                with colB:
                    if st.button("Reset Entire Graph", key="admin_reset_graph_btn"):
                        st.session_state.kg = nx.DiGraph()
                        st.success("Knowledge graph reset successfully.")
                st.markdown("</div>", unsafe_allow_html=True)

            # 6. User Feedback (Admin view)
            with admin_tabs[5]:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.write("### All User Feedback")
                if len(st.session_state.feedback) == 0:
                    st.info("No feedback submitted yet.")
                else:
                    fb_df = pd.DataFrame(st.session_state.feedback)
                    st.dataframe(fb_df)
                    if st.button("Download Feedback as CSV", key="admin_download_feedback_btn"):
                        csv_bytes = fb_df.to_csv(index=False).encode("utf-8")
                        st.download_button("Download Now", data=csv_bytes, file_name="all_feedback.csv", mime="text/csv", key="admin_dl_feedback")
                st.markdown("</div>", unsafe_allow_html=True)

    # Feedback tab (user)
    idx = tab_index("💬 Feedback")
    with tabs[idx]:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("User Feedback")
        current_feedback_key = st.session_state.feedback_key
        feedback_text = st.text_area("Your Feedback / Suggestions", height=150, key=f"user_feedback_input_{current_feedback_key}")
        feedback_type = st.selectbox("Feedback Type", ["Bug Report", "Feature Request", "General Comment"], key="feedback_type")
        if st.button("Submit Feedback", key="submit_feedback_btn"):
            if feedback_text.strip():
                st.session_state.feedback.append({
                    "id": st.session_state.feedback_id,
                    "user": st.session_state.current_user,
                    "type": feedback_type,
                    "text": feedback_text,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                st.session_state.feedback_id += 1
                st.success("Thank you! Your feedback has been recorded.")
                st.session_state.feedback_key += 1
                st.rerun()
            else:
                st.warning("Please enter your feedback before submitting.")
        st.markdown("---")
        st.markdown("*Recent Feedback (Admin View)*")
        if st.session_state.feedback:
            st.dataframe(pd.DataFrame(list(reversed(st.session_state.feedback))).head(10))
        else:
            st.info("No feedback submitted yet.")
        st.markdown("</div>", unsafe_allow_html=True)

#Main App 
if not st.session_state.auth:
    st.sidebar.title("Authentication")
    auth_mode = st.sidebar.radio("Mode", ["Signup", "Login"], index=0, key="auth_mode_radio")
    if auth_mode == "Signup":
        signup_ui()
    else:
        login_ui()
else:
    st.sidebar.success(f"Logged in as *{st.session_state.current_user}*")
    if st.sidebar.button("Logout", key="logout_btn"):
        st.session_state.auth = False
        st.session_state.current_user = None
        st.rerun()
    dashboard_ui()

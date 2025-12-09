import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import tempfile
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import speech_recognition as sr
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
import io

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="Mental Health Dashboard", layout="wide")
st.sidebar.header("Dataset Upload")

# -------------------------
# Load spaCy NLP
# -------------------------
nlp = spacy.load("en_core_web_sm")

# -------------------------
# Dataset loader
# -------------------------
def load_dataset(path):
    try:
        df = pd.read_csv(path)
        # ensure body & category consistency
        if 'body' not in df.columns or 'category' not in df.columns:
            st.error("CSV must contain 'body' and 'category' columns.")
            return None
        df['body'] = df['body'].fillna('').astype(str)
        df = df.dropna(subset=['category'])
        df['category'] = df['category'].astype(str)
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

uploaded_file = st.sidebar.file_uploader("Upload CSV file", type="csv")
if uploaded_file:
    df = load_dataset(uploaded_file)
else:
    DATA_PATH = r"C:\Users\91984\Downloads\Mental Health Disorder Detection Dataset.csv"
    df = load_dataset(DATA_PATH)

if df is not None:
    st.sidebar.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

# -------------------------
# Initialize session state keys
# -------------------------
if "model" not in st.session_state:
    st.session_state.model = None
if "vectorizer" not in st.session_state:
    st.session_state.vectorizer = None
if "accuracy" not in st.session_state:
    st.session_state.accuracy = None
if "y_test" not in st.session_state:
    st.session_state.y_test = None
if "y_pred" not in st.session_state:
    st.session_state.y_pred = None
if "predictions" not in st.session_state:
    st.session_state.predictions = []
if "feedback" not in st.session_state:
    st.session_state.feedback = []
if "nlp_logs" not in st.session_state:
    st.session_state.nlp_logs = []  # Each entry: {"text":..., "entities":int, "triplets":int}

# -------------------------
# ML model functions
# -------------------------
def train_ml_model(df):
    X = df['body']
    y = df['category']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    return model, vectorizer, accuracy, y_test, y_pred

def predict_disorder(model, vectorizer, text):
    text = str(text)
    vec = vectorizer.transform([text])
    return model.predict(vec)[0]

# -------------------------
# Voice assistant
# -------------------------
def recognize_speech():
    r = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            st.info("Listening... Speak now.")
            audio = r.listen(source, phrase_time_limit=5)
    except Exception as e:
        st.error(f"Microphone error: {e}")
        return ""
    try:
        text = r.recognize_google(audio)
        return text
    except Exception as e:
        st.error(f"Could not recognize speech: {e}")
        return ""

# -------------------------
# Triplet extraction
# -------------------------
def extract_triplets(text):
    doc = nlp(text)
    triplets = []
    for sent in doc.sents:
        subj, verb, obj = None, None, None
        for token in sent:
            # dependency labels contain 'subj' or 'obj', check membership
            if "subj" in token.dep_:
                subj = token.text
            if "obj" in token.dep_:
                obj = token.text
            if token.pos_ == "VERB":
                verb = token.text
        if subj and verb and obj:
            triplets.append((subj, verb, obj))
    return triplets

# -------------------------
# Graph functions
# -------------------------
def build_full_graph(df):
    G = nx.Graph()
    categories = df['category'].unique()
    G.add_nodes_from(categories)
    for row in df['category']:
        for other in categories:
            if other != row:
                if G.has_edge(row, other):
                    G[row][other]['weight'] += 1
                else:
                    G.add_edge(row, other, weight=1)
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
               "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
               "#bcbd22", "#17becf"]
    color_map = {cat: palette[i % len(palette)] for i, cat in enumerate(categories)}
    net = Network(height="500px", width="100%", notebook=False, bgcolor="#f0f0f0", font_color="black")
    for node in G.nodes():
        net.add_node(node, label=node, color=color_map[node], size=20)
    for u, v, data in G.edges(data=True):
        net.add_edge(u, v, value=data['weight'], color="#999999")
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    tmp_file.close()
    net.save_graph(tmp_file.name)
    return tmp_file.name

def build_subgraph_for_category(df, category):
    G = nx.Graph()
    if category not in df['category'].unique():
        return None
    G.add_node(category)
    for other in df['category'].unique():
        if other != category:
            co_count = df[df['category'] == category].shape[0]
            if co_count > 0:
                G.add_node(other)
                G.add_edge(category, other, weight=co_count)
    color_map = {category: "red"}
    for node in G.nodes():
        if node != category:
            color_map[node] = "#1f77b4"
    net = Network(height="500px", width="100%", notebook=False, bgcolor="#f0f0f0", font_color="black")
    for node in G.nodes():
        net.add_node(node, label=node, color=color_map[node], size=30 if node==category else 20)
    for u, v, data in G.edges(data=True):
        net.add_edge(u, v, value=data['weight'], color="#999999")
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    tmp_file.close()
    net.save_graph(tmp_file.name)
    return tmp_file.name

# -------------------------
# Horizontal tabs
# -------------------------
tab_labels = ["Dataset Loader", "NLP Pipeline", "Disorder Detection",
              "Charts & Insights", "Subgraph Visualizer", "Full Visualization Graph", "Admin Dashboard"]
tabs = st.tabs(tab_labels)

# -------------------------
# 1️⃣ Dataset Loader
# -------------------------
with tabs[0]:
    st.header("Dataset Loader")
    if df is not None:
        st.dataframe(df.head())
        st.success(f"Dataset has {df.shape[0]} rows and {df.shape[1]} columns.")
    else:
        st.warning("Upload CSV from the sidebar.")

# -------------------------
# 2️⃣ NLP Pipeline
# -------------------------
with tabs[1]:
    st.header("NLP Pipeline")
    if df is not None:
        row_index = st.number_input("Dataset row index (optional)", min_value=0, max_value=len(df)-1, value=0)
        default_text = df.loc[row_index, 'body']
    else:
        default_text = ""
    text_input = st.text_area("Enter text for NLP analysis (editable):", value=default_text, height=150)

    st.subheader("Named Entities")
    doc = nlp(text_input)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    if entities:
        for ent_text, ent_label in entities:
            st.write(f"{ent_text} → {ent_label}")
    else:
        st.write("No entities detected.")

    st.subheader("Triplet Extraction (Subject-Verb-Object)")
    triplets = extract_triplets(text_input)
    if triplets:
        for subj, verb, obj in triplets:
            st.write(f"({subj}, {verb}, {obj})")
    else:
        st.write("No triplets detected.")

    # Buttons to explicitly log this NLP run to admin monitoring (avoids duplicate logs on reruns)
    col1, col2 = st.columns([1,1])
    with col1:
        if st.button("Log this NLP Input"):
            st.session_state.nlp_logs.append({
                "text": text_input,
                "entities": len(entities),
                "triplets": len(triplets)
            })
            st.success("NLP input logged for monitoring.")
    with col2:
        if st.button("Clear NLP Logs"):
            st.session_state.nlp_logs = []
            st.info("NLP logs cleared.")

# -------------------------
# 3️⃣ Disorder Detection
# -------------------------
with tabs[2]:
    st.header("Disorder Detection (Train Model)")
    if df is not None:
        if st.button("Train Model"):
            (st.session_state.model, st.session_state.vectorizer,
             st.session_state.accuracy, st.session_state.y_test,
             st.session_state.y_pred) = train_ml_model(df)
            st.success(f"Model trained! Accuracy: {st.session_state.accuracy:.2f}")

            # Show confusion matrix immediately
            try:
                cm_fig, cm_ax = plt.subplots(figsize=(8,6))
                labels = df['category'].unique()
                cm = confusion_matrix(st.session_state.y_test, st.session_state.y_pred, labels=labels)
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                            xticklabels=labels,
                            yticklabels=labels,
                            ax=cm_ax)
                cm_ax.set_xlabel("Predicted")
                cm_ax.set_ylabel("Actual")
                st.pyplot(cm_fig)
                plt.close(cm_fig)
            except Exception as e:
                st.error(f"Error plotting confusion matrix: {e}")

        if st.session_state.model is not None:
            st.info(f"Model is trained! Accuracy: {st.session_state.accuracy:.2f}")

        # Text input for prediction
        text_input2 = st.text_area("Enter text to predict disorder:", height=100)
        if st.button("Predict"):
            if st.session_state.model is None:
                st.warning("Train the model first!")
            elif text_input2.strip() != "":
                prediction = predict_disorder(st.session_state.model, st.session_state.vectorizer, text_input2)
                st.write(f"Predicted Category: {prediction}")
                st.session_state.predictions.append((text_input2, prediction))

        # Voice input
        st.subheader("Or use voice input:")
        if st.button("Predict via Voice"):
            if st.session_state.model is None:
                st.warning("Train the model first!")
            else:
                spoken_text = recognize_speech()
                if spoken_text:
                    st.write(f"Recognized Text: {spoken_text}")
                    prediction = predict_disorder(st.session_state.model, st.session_state.vectorizer, spoken_text)
                    st.write(f"Predicted Category: {prediction}")
                    st.session_state.predictions.append((spoken_text, prediction))

# -------------------------
# 4️⃣ Charts & Insights
# -------------------------
with tabs[3]:
    st.header("Charts & Insights")
    if df is not None:
        category_counts = df['category'].value_counts()

        st.subheader("Category Distribution (Bar Chart)")
        fig1, ax1 = plt.subplots()
        ax1.bar(category_counts.index, category_counts.values)
        ax1.set_xlabel("Category"); ax1.set_ylabel("Count")
        ax1.set_title("Category Distribution")
        plt.xticks(rotation=45)
        st.pyplot(fig1)

        st.subheader("Category Distribution (Pie Chart)")
        fig2, ax2 = plt.subplots()
        ax2.pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%', startangle=140)
        ax2.set_title("Category Percentage")
        st.pyplot(fig2)

        st.subheader("Category Distribution (Horizontal Bar Chart)")
        fig3, ax3 = plt.subplots()
        ax3.barh(category_counts.index, category_counts.values)
        ax3.set_xlabel("Count"); ax3.set_ylabel("Category")
        ax3.set_title("Horizontal Bar Chart")
        st.pyplot(fig3)
    else:
        st.warning("Upload dataset to see charts.")

# -------------------------
# 5️⃣ Subgraph Visualizer
# -------------------------
with tabs[4]:
    st.header("Subgraph Visualizer (Filtered by Last Prediction)")
    if df is not None:
        if st.session_state.predictions:
            last_prediction = st.session_state.predictions[-1][1]
            st.info(f"Showing subgraph for predicted category: {last_prediction}")
            graph_file = build_subgraph_for_category(df, last_prediction)
            if graph_file:
                with open(graph_file, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                st.components.v1.html(html_content, height=500)
            else:
                st.warning("No data for this category.")
        else:
            st.warning("Predict a category first to see subgraph.")
    else:
        st.warning("Upload dataset first.")

# -------------------------
# 6️⃣ Full Visualization Graph
# -------------------------
with tabs[5]:
    st.header("Full Visualization Graph")
    if df is not None:
        graph_file = build_full_graph(df)
        with open(graph_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=500)
    else:
        st.warning("Upload dataset first.")

# -------------------------
# 7️⃣ Admin Dashboard
# -------------------------
with tabs[6]:
    st.header("Admin Dashboard")
    if df is not None:
        st.subheader("Dataset Info")
        st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
        st.write("Missing values per column:")
        st.write(df.isnull().sum())

    if st.session_state.model is not None:
        st.subheader("Model Info")
        st.write(f"Accuracy: {st.session_state.accuracy:.2f}")
        st.write(f"Model type: {type(st.session_state.model).__name__}")

    # -------------------------
    # Recent Predictions
    # -------------------------
    st.subheader("Recent Predictions")
    if st.button("Clear Predictions"):
        st.session_state.predictions = []
    if st.session_state.predictions:
        for i, (text, pred) in enumerate(st.session_state.predictions[-10:]):
            st.write(f"{i+1}. Text: {text} → Predicted: {pred}")
    else:
        st.write("No predictions yet.")

    # -------------------------
    # NLP Pipeline Monitoring
    # -------------------------
    st.header("NLP Pipeline Monitoring")
    if st.session_state.nlp_logs:
        st.subheader("Recent NLP Inputs (latest 5)")
        for log in st.session_state.nlp_logs[-5:]:
            preview = log['text'][:180].replace("\n", " ")
            st.write(f"• Text preview: {preview}{'...' if len(log['text'])>180 else ''}")
            st.write(f"  - Entities detected: {log['entities']}; Triplets extracted: {log['triplets']}")
            st.write("---")

        # Summary stats
        total_inputs = len(st.session_state.nlp_logs)
        avg_entities = sum(x["entities"] for x in st.session_state.nlp_logs) / total_inputs
        avg_triplets = sum(x["triplets"] for x in st.session_state.nlp_logs) / total_inputs

        st.metric("Total NLP Inputs Logged", total_inputs)
        st.metric("Average Entities per Text", round(avg_entities, 2))
        st.metric("Average Triplets per Text", round(avg_triplets, 2))

        # Download NLP logs as CSV
        try:
            logs_df = pd.DataFrame(st.session_state.nlp_logs)
            csv_buf = logs_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download NLP Logs CSV", data=csv_buf, file_name="nlp_logs.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Error preparing NLP logs download: {e}")
    else:
        st.write("No NLP logs available yet. Use the 'Log this NLP Input' button in the NLP Pipeline tab to add logs.")

    # -------------------------
    # Feedback Section
    # -------------------------
    st.header("User Feedback")
    feedback_text = st.text_area("Submit feedback about predictions, UI, NLP results, etc.", height=120)
    col_fb1, col_fb2 = st.columns([1,1])
    with col_fb1:
        if st.button("Submit Feedback"):
            if feedback_text.strip() != "":
                st.session_state.feedback.append(feedback_text.strip())
                st.success("Thank you! Feedback submitted.")
            else:
                st.warning("Feedback cannot be empty.")
    with col_fb2:
        if st.button("Clear Feedback"):
            st.session_state.feedback = []
            st.info("Feedback cleared.")

    st.subheader("All Feedback (latest first)")
    if st.session_state.feedback:
        for i, fb in enumerate(st.session_state.feedback[::-1], 1):
            st.write(f"**{i}.** {fb}")
        # allow download of feedback
        try:
            fb_df = pd.DataFrame({"feedback": st.session_state.feedback[::-1]})
            fb_csv = fb_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Feedback CSV", data=fb_csv, file_name="feedback.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Error preparing feedback download: {e}")
    else:
        st.write("No feedback yet.")

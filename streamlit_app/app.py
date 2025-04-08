import streamlit as st
import time
import numpy as np
import plotly.express as px
from transformers import AutoTokenizer, AutoModel
from gensim.models import Word2Vec, FastText
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from sklearn.manifold import TSNE
import torch
import os
import requests
from streamlit_lottie import st_lottie

# Monkey patch for torch if needed
if not hasattr(torch._classes, '__path__'):
    torch._classes.__path__ = []

st.title("Animated Visualization: Internal Computations in Word Embedding Models")
st.markdown("""
This demo visualizes the internal computations of different word embedding models.
- **Static Models (Word2Vec, FastText, GloVe):** Process text with fixed embeddings.
- **Contextual Models (BERT, GPT):** Use dynamic, context-aware embeddings via transformers.
Use the sidebar to select a model and compare side by side! Supports sentences or paragraphs.
""")

# --- Helper Functions for Animations and Lottie Integration ---

def load_lottieurl(url: str):
    """Load Lottie animation from URL"""
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Example Lottie URLs (replace with actual URLs or remove if not accessible)
lottie_token_flow = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_x62chJ.json")
lottie_attention = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_3rwasyjy.json")

def animate_progress(label: str, duration: float = 2.0):
    """Animate a text-based progress indicator over a set duration"""
    progress_placeholder = st.empty()
    increments = 20
    for i in range(increments):
        time.sleep(duration / increments)
        progress_percent = int(100 * (i + 1) / increments)
        progress_placeholder.markdown(f"**{label}: {progress_percent}%**")
    progress_placeholder.empty()

# --- Model-Specific Simulation Functions ---

def simulate_word2vec(sentences):
    """Simulate Word2Vec: token-by-token processing, sentence by sentence."""
    placeholder = st.empty()
    flow = ""
    for i, tokens in enumerate(sentences, 1):
        sentence_flow = ""
        for token in tokens:
            sentence_flow += token + " → "
            flow = f"**Processing Sentence {i}:** {sentence_flow}"
            placeholder.markdown(flow)
            time.sleep(0.3)
        flow += f"(End of Sentence {i}) → "
    placeholder.markdown(f"**Final Sequence:** {flow[:-3]}")

def simulate_fasttext(sentences):
    """Simulate FastText: token-level with subword splitting, sentence by sentence."""
    placeholder = st.empty()
    for i, tokens in enumerate(sentences, 1):
        st.markdown(f"**Sentence {i}:**")
        for token in tokens:
            if len(token) > 4:
                mid = len(token) // 2
                sub1, sub2 = token[:mid], token[mid:]
                placeholder.markdown(f"**{token}** → *Subwords:* `{sub1}` + `##{sub2}`")
            else:
                placeholder.markdown(f"**{token}** → *No subword splitting needed*")
            time.sleep(0.6)

def simulate_glove(sentences):
    """Simulate GloVe: global context of all sentences first, then token extraction."""
    placeholder = st.empty()
    full_text = " ".join(" ".join(tokens) for tokens in sentences)
    placeholder.markdown(f"**Global Context Processed:** _{full_text}_")
    time.sleep(1.5)
    for i, tokens in enumerate(sentences, 1):
        st.markdown(f"**Extracting from Sentence {i}:**")
        for token in tokens:
            placeholder.markdown(f"**Global Context Extraction:** *Found token:* `{token}`")
            time.sleep(0.4)

def simulate_bert(sentences):
    """Simulate BERT: subword tokenization and bidirectional processing for all sentences."""
    placeholder = st.empty()
    processed_tokens = []
    for tokens in sentences:
        for token in tokens:
            if len(token) > 4:
                mid = len(token) // 2
                processed_tokens.extend([token[:mid], "##" + token[mid:]])
            else:
                processed_tokens.append(token)
        processed_tokens.append("[SEP]")
    processed_tokens = processed_tokens[:-1]
    placeholder.markdown(f"**BERT Tokenization (with subwords):** {', '.join(processed_tokens)}")
    time.sleep(1)
    left_context = " ".join(["←"] * len(processed_tokens))
    right_context = " ".join(["→"] * len(processed_tokens))
    placeholder.markdown(f"**Left Context:** {left_context}")
    time.sleep(0.8)
    placeholder.markdown(f"**Right Context:** {right_context}")
    time.sleep(0.8)
    placeholder.markdown(f"**Bidirectional Processing for:** {', '.join(processed_tokens)}")

def simulate_gpt(sentences):
    """Simulate GPT: left-to-right generation across sentences."""
    placeholder = st.empty()
    generated = ""
    for i, tokens in enumerate(sentences, 1):
        st.markdown(f"**Generating Sentence {i}:**")
        for token in tokens:
            generated += token + " "
            placeholder.markdown(f"**Generating Left-to-Right:** {generated}")
            time.sleep(0.3)
        generated += " "

# --- Sidebar Model Selection and Input ---

st.sidebar.markdown("## Configuration")
model_type = st.sidebar.selectbox("Select Model Type", 
                                  ("Word2Vec", "FastText", "GloVe", "BERT", "GPT"))
input_text = st.sidebar.text_area("Enter text (sentence or paragraph):",
                                  "The quick brown fox jumps over the lazy dog. It runs fast.")

# --- Comparison Section ---

if st.sidebar.button("Compare Model Computations"):
    st.markdown("## Side-by-Side Comparison of Internal Computations")
    col_static, col_contextual = st.columns(2)
    sentences = [s.lower().split() for s in input_text.split(".") if s.strip()]
    if not sentences:
        st.error("Please enter some text!")
        st.stop()
    
    with col_static:
        st.header("Static Model: Word2Vec")
        st.markdown("**Simulated Process:** Token-by-token processing")
        if lottie_token_flow:
            st_lottie(lottie_token_flow, height=150, key="static_lottie")
        animate_progress("Processing Static Model", 1.5)
        simulate_word2vec(sentences)
        # Simulated t-SNE plot
        tokens = [token for s in sentences for token in s]
        dummy_vectors = [np.random.rand(50) for _ in tokens]
        tsne = TSNE(n_components=2, perplexity=5, random_state=42)
        tsne_results = tsne.fit_transform(np.array(dummy_vectors))
        fig_static = px.scatter(x=tsne_results[:, 0], y=tsne_results[:, 1], text=tokens,
                                title="Simulated Word2Vec Embedding Space",
                                labels={"x": "Dim 1", "y": "Dim 2"})
        st.plotly_chart(fig_static, use_container_width=True)
    
    with col_contextual:
        st.header("Contextual Model: BERT")
        st.markdown("**Simulated Process:** Bidirectional processing with subword tokenization")
        if lottie_attention:
            st_lottie(lottie_attention, height=150, key="contextual_lottie")
        animate_progress("Processing Contextual Model", 1.5)
        simulate_bert(sentences)
        # Simulated t-SNE plot
        processed_tokens = []
        for tokens in sentences:
            for token in tokens:
                if len(token) > 4:
                    mid = len(token) // 2
                    processed_tokens.extend([token[:mid], "##" + token[mid:]])
                else:
                    processed_tokens.append(token)
            processed_tokens.append("[SEP]")
        processed_tokens = processed_tokens[:-1]
        dummy_vectors = [np.random.rand(50) for _ in processed_tokens]
        tsne_context = TSNE(n_components=2, perplexity=5, random_state=42)
        tsne_results_context = tsne_context.fit_transform(np.array(dummy_vectors))
        fig_context = px.scatter(x=tsne_results_context[:, 0], y=tsne_results_context[:, 1], text=processed_tokens,
                                 title="Simulated BERT Embedding Space",
                                 labels={"x": "Dim 1", "y": "Dim 2"})
        st.plotly_chart(fig_context, use_container_width=True)
    
    st.success("Side-by-side simulation complete!")

# --- Single Model Processing ---

if input_text:
    st.markdown("## Single Model Processing")
    sentences = [s.lower().split() for s in input_text.split(".") if s.strip()]
    if not sentences:
        st.error("Please enter some text!")
    else:
        st.subheader(f"Processing with {model_type}")
        if model_type == "Word2Vec":
            st.markdown("**Simulated Process:** Token-by-token processing")
            if lottie_token_flow:
                st_lottie(lottie_token_flow, height=150, key="word2vec_lottie")
            animate_progress("Processing", 2.0)
            simulate_word2vec(sentences)
        elif model_type == "FastText":
            st.markdown("**Simulated Process:** Token-level with subword splitting")
            if lottie_token_flow:
                st_lottie(lottie_token_flow, height=150, key="fasttext_lottie")
            animate_progress("Processing", 2.0)
            simulate_fasttext(sentences)
        elif model_type == "GloVe":
            st.markdown("**Simulated Process:** Global context extraction")
            if lottie_token_flow:
                st_lottie(lottie_token_flow, height=150, key="glove_lottie")
            animate_progress("Processing", 2.0)
            simulate_glove(sentences)
        elif model_type == "BERT":
            st.markdown("**Simulated Process:** Bidirectional processing with subword tokenization")
            if lottie_attention:
                st_lottie(lottie_attention, height=150, key="bert_lottie")
            animate_progress("Processing", 2.0)
            simulate_bert(sentences)
        elif model_type == "GPT":
            st.markdown("**Simulated Process:** Left-to-right generation")
            if lottie_attention:
                st_lottie(lottie_attention, height=150, key="gpt_lottie")
            animate_progress("Processing", 2.0)
            simulate_gpt(sentences)
        
        # Simulated t-SNE plot for the selected model
        st.subheader(f"Simulated Embedding Visualization for {model_type}")
        if model_type in ("Word2Vec", "FastText", "GloVe"):
            tokens = [token for s in sentences for token in s]
        else:  # BERT or GPT
            tokens = []
            for s in sentences:
                for token in s:
                    if len(token) > 4 and model_type == "BERT":
                        mid = len(token) // 2
                        tokens.extend([token[:mid], "##" + token[mid:]])
                    else:
                        tokens.append(token)
                if model_type == "BERT":
                    tokens.append("[SEP]")
            if model_type == "BERT":
                tokens = tokens[:-1]
        dummy_vectors = [np.random.rand(50) for _ in tokens]
        tsne = TSNE(n_components=2, perplexity=5, random_state=42)
        tsne_results = tsne.fit_transform(np.array(dummy_vectors))
        fig = px.scatter(x=tsne_results[:, 0], y=tsne_results[:, 1], text=tokens,
                         title=f"Simulated {model_type} Embedding Space",
                         labels={"x": "Dimension 1", "y": "Dimension 2"})
        st.plotly_chart(fig, use_container_width=True)

        # Optional: Actual embeddings (kept from original code, simplified)
        if model_type in ("Word2Vec", "FastText"):
            sentences_flat = [token for s in sentences for token in s]
            model_static = Word2Vec([sentences_flat], vector_size=50, window=2, min_count=1, sg=1) if model_type == "Word2Vec" else FastText([sentences_flat], vector_size=50, window=2, min_count=1)
            word_vectors = [model_static.wv[token] for token in sentences_flat if token in model_static.wv]
            if word_vectors:
                tsne = TSNE(n_components=2, perplexity=5, random_state=42)
                tsne_results = tsne.fit_transform(np.array(word_vectors))
                fig = px.scatter(x=tsne_results[:, 0], y=tsne_results[:, 1], text=[token for token in sentences_flat if token in model_static.wv],
                                 title=f"Actual {model_type} Embedding Visualization",
                                 labels={"x": "Dimension 1", "y": "Dimension 2"})
                st.plotly_chart(fig, use_container_width=True)
        elif model_type in ("BERT", "GPT"):
            model_name = 'bert-base-uncased' if model_type == "BERT" else 'gpt2'
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            inputs = tokenizer(input_text, return_tensors="pt")
            outputs = model(**inputs)
            token_embeds = outputs.last_hidden_state[0].detach().numpy()
            token_labels = tokenizer.tokenize(input_text)
            num_vis = min(10, len(token_labels))
            tsne_context = TSNE(n_components=2, perplexity=5, random_state=42).fit_transform(token_embeds[:num_vis])
            fig = px.scatter(x=tsne_context[:, 0], y=tsne_context[:, 1], text=token_labels[:num_vis],
                             title=f"Actual {model_type} Contextual Embedding Visualization",
                             labels={"x": "Dimension 1", "y": "Dimension 2"})
            st.plotly_chart(fig, use_container_width=True)
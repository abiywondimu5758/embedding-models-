import streamlit as st
import time
import numpy as np
import plotly.express as px
from transformers import AutoTokenizer, AutoModel
from gensim.models import Word2Vec, FastText
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA  # New import for PCA
import torch
import os
import requests
from streamlit_lottie import st_lottie

# Monkey patch for torch if needed
if not hasattr(torch._classes, '__path__'):
    torch._classes.__path__ = []

# Set page configuration for wide layout
st.set_page_config(layout="wide")

# --- Page Title & Description ---
st.title("Animated Visualization: Internal Computations in Word Embedding Models")
st.markdown("""
This demo visualizes the internal computations of different word embedding models.

- **Static Models (Word2Vec, FastText, GloVe):** Process text with fixed embeddings.
- **Contextual Models (BERT, GPT):** Use dynamic, context-aware embeddings via transformers.

Use the sidebar to select a model and compare side by side!
Each simulation highlights improvements between models using animated flow diagrams and interactive visualizations.
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

# --- Model-Specific Simulation Functions with Enhanced Annotations ---

def simulate_word2vec(sentences, window_size, vector_size):
    # Improved simulation: display each token’s processing step with formatting for beauty and clarity
    placeholder = st.empty()
    simulation_steps = []
    for tokens in sentences:
        for j, token in enumerate(tokens):
            start = max(0, j - window_size)
            end = min(len(tokens), j + window_size + 1)
            # Exclude the current token from its context window
            context = tokens[start:j] + tokens[j+1:end]
            # Format step using HTML styles for a beautiful output
            step = (f"<span style='color:#4285F4; font-weight:bold;'>{token}</span> &rarr; [ "
                    f"{', '.join(context)} ] "
                    f"<span style='color:gray;'>(Vec d={vector_size})</span>")
            simulation_steps.append(step)
            # Display the evolving simulation with line breaks for readability
            placeholder.markdown("<br>".join(simulation_steps) + " <em>(processing...)</em>", unsafe_allow_html=True)
            time.sleep(0.5)
        simulation_steps.append("<br><em>-- End of Sentence --</em>")
    # Final display with a header and enhanced formatting
    placeholder.markdown("<h3>Final Simulation Results</h3>" + "<br>".join(simulation_steps), unsafe_allow_html=True)

def simulate_fasttext(sentences, window_size, vector_size, min_n, max_n):
    # Improved simulation: display FastText processing steps with beautiful formatting
    placeholder = st.empty()
    simulation_steps = []
    for tokens in sentences:
        for j, token in enumerate(tokens):
            start = max(0, j - window_size)
            end = min(len(tokens), j + window_size + 1)
            context = tokens[start:j] + tokens[j+1:end]
            context_str = ", ".join(context)
            if len(token) >= min_n:
                ngrams = []
                for n in range(min_n, max_n + 1):
                    for i in range(len(token) - n + 1):
                        ngrams.append(token[i:i+n])
                subword_str = ", ".join(ngrams)
                step = (f"<span style='color:#DB4437; font-weight:bold;'>{token}</span> &rarr; [ "
                        f"Subwords: {subword_str} ] | Context: [ {context_str} ] "
                        f"<span style='color:gray;'>(Vec d={vector_size})</span>")
            else:
                step = (f"<span style='color:#DB4437; font-weight:bold;'>{token}</span> (no subwords) | Context: [ {context_str} ] "
                        f"<span style='color:gray;'>(Vec d={vector_size})</span>")
            simulation_steps.append(step)
            placeholder.markdown("<br>".join(simulation_steps) + " <em>(processing...)</em>", unsafe_allow_html=True)
            time.sleep(0.6)
        simulation_steps.append("<br><em>-- End of Sentence --</em>")
    placeholder.markdown("<h3>Final FastText Simulation Results</h3>" + "<br>".join(simulation_steps), unsafe_allow_html=True)

def simulate_glove(sentences):
    """Simulate GloVe: Global context extraction and token extraction with an animated moving effect.
    
    GloVe builds a global context from the full text, then extracts tokens from it, unlike
    other models that process tokens sequentially or via subword splits.
    This simulation visually moves tokens from the global context to an extraction line.
    Additionally, a co-occurrence matrix is computed and displayed as an actual matrix using px.
    """
    placeholder_global = st.empty()
    placeholder_extracted = st.empty()
    
    tokens_all = []
    sentence_boundaries = []
    count = 0
    for tokens in sentences:
        tokens_all.extend(tokens)
        count += len(tokens)
        sentence_boundaries.append(count)
    
    global_context = " ".join([f"`{token}`" for token in tokens_all])
    placeholder_global.markdown(f"**Global Context:** {global_context}")
    
    st.info("GloVe builds a global context from all text and then extracts tokens based on overall co-occurrence.")
    time.sleep(2)
    
    remaining_tokens = tokens_all.copy()
    extracted_tokens = []
    for index, token in enumerate(tokens_all):
        if token in remaining_tokens:
            remaining_tokens.remove(token)
        updated_global = " ".join([f"`{t}`" for t in remaining_tokens])
        placeholder_global.markdown(f"**Global Context:** {updated_global}")
        extracted_tokens.append(f"`{token}`")
        if (index + 1) in sentence_boundaries:
            extracted_tokens.append("↘")
        extraction_line = " → ".join(extracted_tokens)
        placeholder_extracted.markdown(f"**Extracted Tokens:** {extraction_line}")
        time.sleep(0.5)
    
    st.success("GloVe processing complete!")
    
    # NEW: Compute the actual co-occurrence matrix
    unique_tokens = sorted(set([token for tokens in sentences for token in tokens]))
    co_occurrence = np.zeros((len(unique_tokens), len(unique_tokens)), dtype=int)
    for tokens in sentences:
        for i, token in enumerate(tokens):
            for j, token2 in enumerate(tokens):
                if i != j:
                    row = unique_tokens.index(token)
                    col = unique_tokens.index(token2)
                    co_occurrence[row, col] += 1
    
    # Display the co-occurrence matrix as an actual matrix using px.imshow
    fig_co = px.imshow(co_occurrence,
                       x=unique_tokens,
                       y=unique_tokens,
                       text_auto=True,
                       color_continuous_scale="Blues",
                       title="Actual Co-occurrence Matrix")
    st.plotly_chart(fig_co, use_container_width=True)

def simulate_bert(sentences):
    # Enhanced simulation: animated token-by-token processing similar to Word2Vec/FastText
    placeholder = st.empty()
    simulation_steps = []
    for tokens in sentences:
        for idx, token in enumerate(tokens):
            left_attn = "←" * (idx + 1)
            right_attn = "→" * (len(tokens) - idx)
            if len(token) > 4:
                mid = len(token) // 2
                part1 = token[:mid]
                part2 = "##" + token[mid:]
                step = f"<span style='color:#34A853; font-weight:bold;'>{token}</span> [{left_attn}|{right_attn}] → [ {part1} | {part2} ]"
            else:
                step = f"<span style='color:#34A853; font-weight:bold;'>{token}</span> [{left_attn}|{right_attn}]"
            simulation_steps.append(step)
            placeholder.markdown("<br>".join(simulation_steps) + " <em>(processing...)</em>", unsafe_allow_html=True)
            time.sleep(0.3)
        simulation_steps.append("<br><em>-- End of Sentence --</em>")
    placeholder.markdown("<h3>Final BERT Simulation Results</h3>" + "<br>".join(simulation_steps), unsafe_allow_html=True)
    st.info("BERT simulation complete!")

def simulate_gpt(sentences):
    """Simulate GPT: Left-to-right sequential generation annotated step-by-step.
    
    This enhanced simulation shows incremental token generation with simulated probability scores.
    """
    placeholder = st.empty()
    final_generation = ""
    for i, tokens in enumerate(sentences, start=1):
        st.markdown(f"**Generating Sentence {i}**")
        generated = ""
        for token in tokens:
            prob = np.round(np.random.rand(), 2)  # Simulate a probability for token generation
            generated += token + " "
            placeholder.markdown(f"**Progress:** {generated} _(Simulated Prob: {prob})_")
            time.sleep(0.3)
        final_generation += generated + " "
        placeholder.markdown(f"**Final Generated Text:** {final_generation}")
        time.sleep(0.5)

# --- Dimensionality Reduction Function ---

def reduce_dimensions(vectors, method="TSNE"):
    if method == "UMAP":
        reducer = umap.UMAP(n_components=2, random_state=42)
        return reducer.fit_transform(np.array(vectors))
    elif method == "PCA":  # New branch for PCA reduction
        pca = PCA(n_components=2)
        return pca.fit_transform(np.array(vectors))
    else:
        tsne = TSNE(n_components=2, perplexity=5, random_state=42)
        return tsne.fit_transform(np.array(vectors))

# --- Sidebar Model Selection & Input ---

st.sidebar.markdown("## Configuration")
model_type = st.sidebar.selectbox("Select Model Type", 
                                  ("Word2Vec", "FastText", "GloVe", "BERT", "GPT"))
# reduction_method = st.sidebar.selectbox("Reduction Method", ("TSNE", "UMAP", "PCA"))  # Updated options
# Set default values for parameters used by simulate_fasttext
reduction_method = "TSNE"
min_n = 3
max_n = 6
if model_type in ("Word2Vec", "FastText"):
    window_size = st.sidebar.slider("Window Size", min_value=1, max_value=10, value=2, step=1)
    vector_size = st.sidebar.slider("Vector Size", min_value=10, max_value=300, value=50, step=10)
    if model_type == "FastText":
        min_n = st.sidebar.slider("Min n", min_value=1, max_value=10, value=3, step=1)
        max_n = st.sidebar.slider("Max n", min_value=1, max_value=15, value=6, step=1)
else:
    window_size = 2
    vector_size = 50

if model_type == "GloVe":
    glove_vector_size = st.sidebar.selectbox("Glove Vector Size", options=[50, 100, 200, 300], index=1)
else:
    glove_vector_size = None
input_text = st.sidebar.text_area("Enter text (sentence or paragraph):",
                                  "The quick brown fox jumps over the lazy dog. It runs fast.")
# NEW: Visualization type selection
visualization_type = st.sidebar.selectbox("Visualization Type", ("Scatter", "Dimension Graph"))

# --- Side-by-Side Comparison Section with Explanatory Panels ---

if st.sidebar.button("Compare Model Computations"):
    st.markdown("## Side-by-Side Comparison of Internal Computations")
    sentences = [s.lower().split() for s in input_text.split(".") if s.strip()]
    if not sentences:
        st.error("Please enter some text!")
        st.stop()
        
    # Inject CSS for horizontal scrolling
    st.markdown(
    """
    <style>
    .horizontal-scroll {
        overflow-x: auto;
        white-space: nowrap;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Wrap the columns in a scrollable container
    st.markdown('<div class="horizontal-scroll">', unsafe_allow_html=True)
    cols = st.columns(5)
    # Word2Vec
    with cols[0]:
        st.header("Word2Vec")
        st.markdown("**Process:** Token-by-token processing with window size " + str(window_size) +
                    " and vector size " + str(vector_size))
        simulate_word2vec(sentences, window_size, vector_size)
        tokens = [token for s in sentences for token in s]
        dummy_vectors = [np.random.rand(vector_size) for _ in tokens]
        reduced_results = reduce_dimensions(dummy_vectors, method=reduction_method)
        fig = px.scatter(x=reduced_results[:, 0], y=reduced_results[:, 1], text=tokens,
                         title="Word2Vec Embedding Space",
                         labels={"x": "Dim 1", "y": "Dim 2"},
                         hover_data={"Token": tokens})
        st.plotly_chart(fig, use_container_width=True)
    # FastText
    with cols[1]:
        st.header("FastText")
        st.markdown("**Process:** Token-level processing with subword splitting")
        simulate_fasttext(sentences, window_size, vector_size, min_n, max_n)
        tokens = [token for s in sentences for token in s]
        dummy_vectors = [np.random.rand(vector_size) for _ in tokens]
        reduced_results = reduce_dimensions(dummy_vectors, method=reduction_method)
        fig = px.scatter(x=reduced_results[:, 0], y=reduced_results[:, 1], text=tokens,
                         title="FastText Embedding Space",
                         labels={"x": "Dim 1", "y": "Dim 2"},
                         hover_data={"Token": tokens})
        st.plotly_chart(fig, use_container_width=True)
    # GloVe
    with cols[2]:
        st.header("GloVe")
        st.markdown("**Process:** Global context extraction then token extraction")
        simulate_glove(sentences)
        tokens = [token for s in sentences for token in s]
        # Use glove_vector_size if defined, otherwise fallback to vector_size
        dim = glove_vector_size if glove_vector_size is not None else vector_size
        dummy_vectors = [np.random.rand(dim) for _ in tokens]
        reduced_results = reduce_dimensions(dummy_vectors, method=reduction_method)
        fig = px.scatter(x=reduced_results[:, 0], y=reduced_results[:, 1], text=tokens,
                         title="GloVe Embedding Space",
                         labels={"x": "Dim 1", "y": "Dim 2"},
                         hover_data={"Token": tokens})
        st.plotly_chart(fig, use_container_width=True)
    # BERT
    with cols[3]:
        st.header("BERT")
        st.markdown("**Process:** Bidirectional processing with subword tokenization")
        simulate_bert(sentences)
        processed_tokens = []
        for s in sentences:
            for token in s:
                if len(token) > 4:
                    mid = len(token) // 2
                    processed_tokens.extend([token[:mid], "##" + token[mid:]])
                else:
                    processed_tokens.append(token)
            processed_tokens.append("[SEP]")
        if processed_tokens and processed_tokens[-1] == "[SEP]":
            processed_tokens = processed_tokens[:-1]
        dummy_vectors = [np.random.rand(50) for _ in processed_tokens]
        reduced_results = reduce_dimensions(dummy_vectors, method=reduction_method)
        fig = px.scatter(x=reduced_results[:, 0], y=reduced_results[:, 1], text=processed_tokens,
                         title="BERT Embedding Space",
                         labels={"x": "Dim 1", "y": "Dim 2"},
                         hover_data={"Sub-token": processed_tokens})
        st.plotly_chart(fig, use_container_width=True)
    # GPT
    with cols[4]:
        st.header("GPT")
        st.markdown("**Process:** Sequential left-to-right generation")
        simulate_gpt(sentences)
        tokens = [token for s in sentences for token in s]
        dummy_vectors = [np.random.rand(vector_size) for _ in tokens]
        reduced_results = reduce_dimensions(dummy_vectors, method=reduction_method)
        fig = px.scatter(x=reduced_results[:, 0], y=reduced_results[:, 1], text=tokens,
                         title="GPT Generation Space",
                         labels={"x": "Dim 1", "y": "Dim 2"},
                         hover_data={"Token": tokens})
        st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --- Single Model Processing Section with Detailed Visualizations ---

if input_text:
    st.markdown("## Single Model Processing")
    sentences = [s.lower().split() for s in input_text.split(".") if s.strip()]
    if not sentences:
        st.error("Please enter some text!")
    else:
        st.subheader(f"Processing with {model_type}")
        if model_type == "Word2Vec":
            st.markdown("**Process:** Simple token-by-token processing with window size " + str(window_size) +
                        " and vector size " + str(vector_size))
            if lottie_token_flow:
                st_lottie(lottie_token_flow, height=150, key="word2vec_lottie")
            animate_progress("Processing", 2.0)
            simulate_word2vec(sentences, window_size, vector_size)
        elif model_type == "FastText":
            st.markdown("**Process:** Token-level processing with subword splitting")
            if lottie_token_flow:
                st_lottie(lottie_token_flow, height=150, key="fasttext_lottie")
            animate_progress("Processing", 2.0)
            simulate_fasttext(sentences, window_size, vector_size, min_n, max_n)
        elif model_type == "GloVe":
            st.markdown("**Process:** Global context extraction followed by token extraction (Actual GloVe Embedding)")
            if lottie_token_flow:
                st_lottie(lottie_token_flow, height=150, key="glove_lottie")
            animate_progress("Processing", 2.0)
            simulate_glove(sentences)
            # Optionally, run the simulation for visual effect
            
        elif model_type == "BERT":
            st.markdown("**Process:** Bidirectional processing with subword tokenization")
            if lottie_attention:
                st_lottie(lottie_attention, height=150, key="bert_lottie")
            animate_progress("Processing", 2.0)
            simulate_bert(sentences)
        elif model_type == "GPT":
            st.markdown("**Process:** Sequential left-to-right generation")
            if lottie_attention:
                st_lottie(lottie_attention, height=150, key="gpt_lottie")
            animate_progress("Processing", 2.0)
            simulate_gpt(sentences)
        
        # Simulated t-SNE Plot for the Selected Model with Additional Info
        st.subheader(f"Simulated Embedding Visualization for {model_type}")
        if visualization_type == "Scatter":
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
                if model_type == "BERT" and tokens and tokens[-1] == "[SEP]":
                    tokens = tokens[:-1]
            # Use glove_vector_size if GloVe, else use vector_size
            if model_type == "GloVe":
                dummy_vectors = [np.random.rand(glove_vector_size) for _ in tokens]
            else:
                dummy_vectors = [np.random.rand(vector_size) for _ in tokens]
            reduced_results = reduce_dimensions(dummy_vectors, method=reduction_method)
            fig = px.scatter(x=reduced_results[:, 0], y=reduced_results[:, 1], text=tokens,
                             title=f"Simulated {model_type} Embedding Space",
                             labels={"x": "Dimension 1", "y": "Dimension 2"},
                             hover_data={"Token": tokens})
            st.plotly_chart(fig, use_container_width=True)
        elif visualization_type == "Dimension Graph":
            tokens = [token for s in sentences for token in s]
            unique_tokens = list(set(tokens))
            selected_token = st.selectbox("Select Token", unique_tokens)
            if model_type == "GloVe":
                dummy_vector = np.random.rand(glove_vector_size)
            else:
                dummy_vector = np.random.rand(vector_size)
            fig = px.bar(x=list(range(len(dummy_vector))), y=dummy_vector,
                         labels={"x": "Dimension", "y": "Value"},
                         title=f"Word Vector Dimensions for '{selected_token}'")
            st.plotly_chart(fig, use_container_width=True)

        # Optional: Display Actual Embedding Visualizations for Deep Models
        if model_type in ("Word2Vec", "FastText"):
            sentences_flat = [token for s in sentences for token in s]
            model_static = Word2Vec([sentences_flat], vector_size=vector_size, window=window_size, min_count=1, sg=1) \
                            if model_type == "Word2Vec" else FastText([sentences_flat], vector_size=vector_size, window=window_size, min_count=1, min_n=2,max_n=5)
            word_vectors = [model_static.wv[token] for token in sentences_flat if token in model_static.wv]
            if word_vectors:
                reduced_results = reduce_dimensions(word_vectors, method=reduction_method)
                fig = px.scatter(x=reduced_results[:, 0], y=reduced_results[:, 1],
                                 text=[token for token in sentences_flat if token in model_static.wv],
                                 title=f"Actual {model_type} Embedding Visualization",
                                 labels={"x": "Dimension 1", "y": "Dimension 2"})
                st.plotly_chart(fig, use_container_width=True)
            # NEW: Print the vector of the first two words to show dimension changes
            
            keys = list(model_static.wv.key_to_index.keys())
            if len(keys) >= 2:
                st.write("First word:", keys[0], "embedding:", model_static.wv[keys[0]].tolist())
                st.write("Second word:", keys[1], "embedding:", model_static.wv[keys[1]].tolist())
        elif model_type == "GloVe":
            glove_file = f'/home/abiy/Documents/assignments/glove.6B/glove.6B.{glove_vector_size}d.txt'
            glove_word2vec_file = f'/home/abiy/Documents/assignments/glove.6B/glove.6B.{glove_vector_size}d.word2vec.txt'
            if not os.path.exists(glove_word2vec_file):
                glove2word2vec(glove_file, glove_word2vec_file)
            glove_model = KeyedVectors.load_word2vec_format(glove_word2vec_file, binary=False)
            # Retrieve actual embeddings for all tokens (ignoring tokens not in the GloVe vocabulary)
            tokens = [token for s in sentences for token in s]
            glove_vectors = [glove_model[token] for token in tokens if token in glove_model]
            if glove_vectors:
                reduced_results = reduce_dimensions(glove_vectors, method=reduction_method)
                fig = px.scatter(x=reduced_results[:, 0], y=reduced_results[:, 1], text=tokens,
                                 title="Actual GloVe Embedding Space",
                                 labels={"x": "Dim 1", "y": "Dim 2"},
                                 hover_data={"Token": tokens})
                st.plotly_chart(fig, use_container_width=True)
            st.write("GloVe Example Vector for first word:",
                     glove_model[list(glove_model.key_to_index.keys())[0]].tolist())
        elif model_type in ("BERT", "GPT"):
            model_name = 'bert-base-uncased' if model_type == "BERT" else 'gpt2'
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            inputs = tokenizer(input_text, return_tensors="pt")
            outputs = model(**inputs)
            token_embeds = outputs.last_hidden_state[0].detach().numpy()
            token_labels = tokenizer.tokenize(input_text)
            num_vis = min(10, len(token_labels))
            reduced_results = reduce_dimensions(token_embeds[:num_vis], method=reduction_method)
            fig = px.scatter(x=reduced_results[:, 0], y=reduced_results[:, 1],
                             text=token_labels[:num_vis],
                             title=f"Actual {model_type} Contextual Embedding Visualization",
                             labels={"x": "Dimension 1", "y": "Dimension 2"})
            st.plotly_chart(fig, use_container_width=True)

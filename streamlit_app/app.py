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

def simulate_word2vec(sentences, window_size):
    """Simulate Word2Vec with dynamic window size: shows each token with its context based on window size."""
    placeholder = st.empty()
    simulation_steps = []
    for tokens in sentences:
        for j, token in enumerate(tokens):
            start = max(0, j - window_size)
            end = min(len(tokens), j + window_size + 1)
            # Exclude the current token from its context window
            context = tokens[start:j] + tokens[j+1:end]
            simulation_steps.append(f"{token} -> {context}")
            placeholder.markdown(" | ".join(simulation_steps) + " _(processing...)_")
            time.sleep(0.5)
        simulation_steps.append("(End of Sentence)")
    placeholder.markdown(f"**Final Simulation:** {' | '.join(simulation_steps)}")

def simulate_fasttext(sentences):
    """Simulate FastText using a unified timeline with visual separators.
    
    FastText processes each token and, if the token is longer than 4 characters,
    it splits it into subword components. This function builds a continuous flow
    of processed tokens, separating sentences with a graphical marker.
    """
    placeholder = st.empty()
    flow_lines = []  # This list holds the processed output for each sentence
    
    for tokens in sentences:
        sentence_flow = []  # Hold processed tokens for the current sentence
        for token in tokens:
            if len(token) > 4:
                mid = len(token) // 2
                sub1, sub2 = token[:mid], token[mid:]
                # Represent the transformation in a concise manner:
                # e.g. "amazing" becomes: **amazing**→[`ama`,`##zing`]
                sentence_flow.append(f"**{token}**→[`{sub1}`,`##{sub2}`]")
            else:
                sentence_flow.append(f"**{token}**(no split)")
            # Update the placeholder with current processing state
            placeholder.markdown(" ".join(flow_lines + sentence_flow))
            time.sleep(0.6)
        
        # Append the processed sentence to the overall flow with a visual separator
        flow_lines.append(" ".join(sentence_flow))
        flow_lines.append("🔻🔻🔻")  # Graphical separator between sentences
    
    # Remove the last separator if present
    if flow_lines and flow_lines[-1] == "🔻🔻🔻":
        flow_lines = flow_lines[:-1]
    
    final_flow = " ".join(flow_lines)
    placeholder.markdown(final_flow)


def simulate_glove(sentences):
    """Simulate GloVe: Global context extraction and token extraction with an animated moving effect.
    
    GloVe builds a global context from the full text, then extracts tokens from it, unlike
    other models (Word2Vec, FastText, BERT, GPT) that process tokens sequentially or via subword splits.
    This simulation visually moves tokens from the global context to an extraction line.
    """
    # Create separate placeholders for global context and extracted tokens
    placeholder_global = st.empty()
    placeholder_extracted = st.empty()
    
    # Flatten sentences into a list and retain sentence boundaries (for optional separators)
    tokens_all = []
    sentence_boundaries = []  # indices where a sentence ends
    count = 0
    for tokens in sentences:
        tokens_all.extend(tokens)
        count += len(tokens)
        sentence_boundaries.append(count)
    
    # Display the full global context
    global_context = " ".join([f"`{token}`" for token in tokens_all])
    placeholder_global.markdown(f"**Global Context:** {global_context}")
    
    st.info("Unlike other models that process tokens one-by-one (Word2Vec/GPT) or use subword and bidirectional techniques (FastText/BERT),\nGloVe first builds a global context from all text, then extracts tokens based on overall co-occurrence.")
    time.sleep(2)
    
    # Prepare for animated extraction: duplicate the token list so we can remove tokens as they're extracted.
    remaining_tokens = tokens_all.copy()
    extracted_tokens = []  # tokens that have been "extracted"
    
    # Iterate over tokens to simulate extraction (moving one token at a time)
    for index, token in enumerate(tokens_all):
        # Remove the token from remaining_tokens (simulate extraction)
        if token in remaining_tokens:  # safety check
            remaining_tokens.remove(token)
        
        # Update the global context display with the remaining tokens
        updated_global = " ".join([f"`{t}`" for t in remaining_tokens])
        placeholder_global.markdown(f"**Global Context:** {updated_global}")
        
        # Add the extracted token (simulate movement by appending with an arrow)
        extracted_tokens.append(f"`{token}`")
        # If this token is the end of a sentence, add a graphical separator to emphasize boundary
        if (index + 1) in sentence_boundaries:
            extracted_tokens.append("↘")
        extraction_line = " → ".join(extracted_tokens)
        placeholder_extracted.markdown(f"**Extracted Tokens:** {extraction_line}")
        
        time.sleep(0.5)
    
    st.success("GloVe processing complete!")


def simulate_bert(sentences):
    """Simulate BERT: enhanced subword tokenization and bidirectional processing.
    
    BERT's approach splits longer tokens into subwords, then processes the entire
    sequence bidirectionally. Unlike sequential models (Word2Vec, GPT) or models that
    use only simple subword splitting (FastText), BERT incorporates both left and right
    context simultaneously for a more robust understanding.
    """
    placeholder = st.empty()
    annotated_flow = []  # List to hold the evolving processing flow.
    
    # Process each sentence, building a continuous flow.
    for tokens in sentences:
        sentence_flow = []
        for token in tokens:
            if len(token) > 4:
                # Split token into subwords for tokens longer than 4 characters.
                mid = len(token) // 2
                sub1, sub2 = token[:mid], "##" + token[mid:]
                sentence_flow.append(f"**{token}** → [ {sub1} | {sub2} ]")
            else:
                sentence_flow.append(f"**{token}**")
            # Update placeholder for an animated effect.
            placeholder.markdown("  |  ".join(annotated_flow + sentence_flow))
            time.sleep(0.3)
        # Append processed sentence flow and add a sentence separator.
        annotated_flow.extend(sentence_flow)
        annotated_flow.append("[SEP]")
    
    # Remove the last sentence separator, if present.
    if annotated_flow and annotated_flow[-1] == "[SEP]":
        annotated_flow.pop()
    
    # Final tokenization flow display.
    placeholder.markdown(f"**BERT Tokenization Flow:**\n{' → '.join(annotated_flow)}")
    time.sleep(1)
    
    # Build a visual representation of bidirectional context.
    left_context = " ".join(["←"] * len(annotated_flow))
    right_context = " ".join(["→"] * len(annotated_flow))
    
    # Display the bidirectional processing with context arrows.
    placeholder.markdown(
        f"**Bidirectional Context Visualization:**\n\n"
        f"**Left Context:** {left_context}\n"
        f"**Right Context:** {right_context}\n\n"
        f"**Final BERT Processing:** {' → '.join(annotated_flow)}"
    )
    time.sleep(0.8)
    
    # Informative note that compares BERT with other models.
    st.info(
        "BERT sets itself apart by employing bidirectional attention, meaning it processes "
        "the full input sequence (both left and right contexts) simultaneously. This leads to "
        "a richer understanding of word usage compared to one-directional models like GPT or the "
        "token-by-token approaches in Word2Vec and FastText. Its use of subword tokenization also "
        "effectively handles rare or misspelled words, making BERT more robust for diverse language tasks."
    )

def simulate_gpt(sentences):
    """Simulate GPT: Left-to-right sequential generation annotated step-by-step."""
    placeholder = st.empty()
    generated = ""
    for i, tokens in enumerate(sentences, 1):
        st.markdown(f"**Generating Sentence {i}:**")
        for token in tokens:
            generated += token + " "
            placeholder.markdown(f"**Progress:** {generated} _(left-to-right generation)_")
            time.sleep(0.3)
        generated += " "  # Separate sentences

# --- Sidebar Model Selection & Input ---

st.sidebar.markdown("## Configuration")
model_type = st.sidebar.selectbox("Select Model Type", 
                                  ("Word2Vec", "FastText", "GloVe", "BERT", "GPT"))
# Add configuration for Word2Vec window size
if model_type == "Word2Vec":
    window_size = st.sidebar.slider("Window Size", min_value=1, max_value=10, value=2, step=1)
input_text = st.sidebar.text_area("Enter text (sentence or paragraph):",
                                  "The quick brown fox jumps over the lazy dog. It runs fast.")

# --- Side-by-Side Comparison Section with Explanatory Panels ---

if st.sidebar.button("Compare Model Computations"):
    st.markdown("## Side-by-Side Comparison of Internal Computations")
    col_static, col_contextual = st.columns(2)
    sentences = [s.lower().split() for s in input_text.split(".") if s.strip()]
    if not sentences:
        st.error("Please enter some text!")
        st.stop()
    
    with col_static:
        st.header("Static Model: Word2Vec")
        st.markdown("**Process:** Token-by-token simple processing")
        if lottie_token_flow:
            st_lottie(lottie_token_flow, height=150, key="static_lottie")
        animate_progress("Static Model Processing", 1.5)
        simulate_word2vec(sentences, window_size)
        # Simulated t-SNE plot with annotations
        tokens = [token for s in sentences for token in s]
        dummy_vectors = [np.random.rand(50) for _ in tokens]
        tsne = TSNE(n_components=2, perplexity=5, random_state=42)
        tsne_results = tsne.fit_transform(np.array(dummy_vectors))
        fig_static = px.scatter(x=tsne_results[:, 0], y=tsne_results[:, 1], text=tokens,
                                title="Word2Vec Embedding Space",
                                labels={"x": "Dim 1", "y": "Dim 2"},
                                hover_data={"Token": tokens})
        st.plotly_chart(fig_static, use_container_width=True)
    
    with col_contextual:
        st.header("Contextual Model: BERT")
        st.markdown("**Process:** Bidirectional processing with enhanced subword tokenization")
        if lottie_attention:
            st_lottie(lottie_attention, height=150, key="contextual_lottie")
        animate_progress("Contextual Model Processing", 1.5)
        simulate_bert(sentences)
        # Prepare processed tokens for t-SNE plot
        processed_tokens = []
        for tokens in sentences:
            for token in tokens:
                if len(token) > 4:
                    mid = len(token) // 2
                    processed_tokens.extend([token[:mid], "##" + token[mid:]])
                else:
                    processed_tokens.append(token)
            processed_tokens.append("[SEP]")
        if processed_tokens and processed_tokens[-1] == "[SEP]":
            processed_tokens = processed_tokens[:-1]
        dummy_vectors = [np.random.rand(50) for _ in processed_tokens]
        tsne_context = TSNE(n_components=2, perplexity=5, random_state=42)
        tsne_results_context = tsne_context.fit_transform(np.array(dummy_vectors))
        fig_context = px.scatter(x=tsne_results_context[:, 0], y=tsne_results_context[:, 1], text=processed_tokens,
                                 title="BERT Embedding Space",
                                 labels={"x": "Dim 1", "y": "Dim 2"},
                                 hover_data={"Sub-token": processed_tokens})
        st.plotly_chart(fig_context, use_container_width=True)
    
    st.success("Side-by-side simulation complete!")

# --- Single Model Processing Section with Detailed Visualizations ---

if input_text:
    st.markdown("## Single Model Processing")
    sentences = [s.lower().split() for s in input_text.split(".") if s.strip()]
    if not sentences:
        st.error("Please enter some text!")
    else:
        st.subheader(f"Processing with {model_type}")
        if model_type == "Word2Vec":
            st.markdown("**Process:** Simple token-by-token processing with window size " + str(window_size))
            if lottie_token_flow:
                st_lottie(lottie_token_flow, height=150, key="word2vec_lottie")
            animate_progress("Processing", 2.0)
            simulate_word2vec(sentences, window_size)
        elif model_type == "FastText":
            st.markdown("**Process:** Token-level processing with subword splitting")
            if lottie_token_flow:
                st_lottie(lottie_token_flow, height=150, key="fasttext_lottie")
            animate_progress("Processing", 2.0)
            simulate_fasttext(sentences)
        elif model_type == "GloVe":
            st.markdown("**Process:** Global context extraction followed by token extraction")
            if lottie_token_flow:
                st_lottie(lottie_token_flow, height=150, key="glove_lottie")
            animate_progress("Processing", 2.0)
            simulate_glove(sentences)
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
        dummy_vectors = [np.random.rand(50) for _ in tokens]
        tsne = TSNE(n_components=2, perplexity=5, random_state=42)
        tsne_results = tsne.fit_transform(np.array(dummy_vectors))
        fig = px.scatter(x=tsne_results[:, 0], y=tsne_results[:, 1], text=tokens,
                         title=f"Simulated {model_type} Embedding Space",
                         labels={"x": "Dimension 1", "y": "Dimension 2"},
                         hover_data={"Token": tokens})
        st.plotly_chart(fig, use_container_width=True)

        # Optional: Display Actual Embedding Visualizations for Deep Models
        if model_type in ("Word2Vec", "FastText"):
            sentences_flat = [token for s in sentences for token in s]
            model_static = Word2Vec([sentences_flat], vector_size=50, window=window_size, min_count=1, sg=1) \
                            if model_type == "Word2Vec" else FastText([sentences_flat], vector_size=50, window=2, min_count=1)
            word_vectors = [model_static.wv[token] for token in sentences_flat if token in model_static.wv]
            if word_vectors:
                tsne = TSNE(n_components=2, perplexity=5, random_state=42)
                tsne_results = tsne.fit_transform(np.array(word_vectors))
                fig = px.scatter(x=tsne_results[:, 0], y=tsne_results[:, 1],
                                 text=[token for token in sentences_flat if token in model_static.wv],
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
            fig = px.scatter(x=tsne_context[:, 0], y=tsne_context[:, 1],
                             text=token_labels[:num_vis],
                             title=f"Actual {model_type} Contextual Embedding Visualization",
                             labels={"x": "Dimension 1", "y": "Dimension 2"})
            st.plotly_chart(fig, use_container_width=True)

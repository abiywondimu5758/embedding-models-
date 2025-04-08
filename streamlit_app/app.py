# streamlit_app/app.py
import streamlit as st
import asyncio
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())
from transformers import AutoTokenizer, AutoModel
from gensim.models import Word2Vec, FastText
import plotly.express as px
from sklearn.manifold import TSNE
import numpy as np
import torch
if not hasattr(torch._classes, '__path__'):
    torch._classes.__path__ = []  # monkey-patch to avoid __path__ access error
import os
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

st.title("Interactive Word Embeddings Demo")
st.markdown("This interactive demo shows how different embedding models process text, including tokenization, subword splitting, and vector visualization using Plotly.")

# Sidebar model selector
model_type = st.sidebar.selectbox("Select Model Type", 
                                  ("Word2Vec", "FastText", "GloVe", "BERT", "GPT"))

# Input sentence
sentence = st.text_input("Enter a sentence:", "The quick brown fox jumps over the lazy dog.")

if sentence:
    st.markdown("### Raw Tokens")
    # For static models: a simple whitespace tokenization
    tokens = sentence.lower().split()
    st.write(tokens)

    # === For static models: Word2Vec, FastText, and GloVe ===
    if model_type in ("Word2Vec", "FastText", "GloVe"):
        sentences = [tokens]  # Toy dataset using the input sentence
        if model_type == "Word2Vec":
            model_static = Word2Vec(sentences, vector_size=50, window=2, min_count=1, sg=1)
        elif model_type == "FastText":
            model_static = FastText(sentences, vector_size=50, window=2, min_count=1)
        elif model_type == "GloVe":
            glove_file = '/home/abiy/Documents/assignments/Embedding/glove.6B.100d.txt'
            word2vec_glove_file = '/home/abiy/Documents/assignments/Embedding/glove.6B.100d.word2vec.txt'
            if not os.path.exists(word2vec_glove_file):
                glove2word2vec(glove_file, word2vec_glove_file)
            model_static = KeyedVectors.load_word2vec_format(word2vec_glove_file, binary=False)

        # Gather embeddings for each token
        word_vectors = []
        token_list = []
        for word in tokens:
            if word in model_static.wv:
                vec = model_static.wv[word]
                word_vectors.append(vec)
                token_list.append(word)
            else:
                st.write(f"'{word}' not found in the model vocabulary.")
        st.markdown("#### Embeddings (Static Model)")
        for word, vec in zip(token_list, word_vectors):
            st.write(f"`{word}`: {vec.tolist()}")

        # Interactive t-SNE visualization with Plotly
        if word_vectors:
            tsne = TSNE(n_components=2, perplexity=5, random_state=42)
            tsne_results = tsne.fit_transform(np.array(word_vectors))
            fig = px.scatter(x=tsne_results[:,0], y=tsne_results[:,1], text=token_list,
                             title=f"t-SNE Visualization of {model_type} Embeddings",
                             labels={"x": "Dimension 1", "y": "Dimension 2"})
            fig.update_traces(textposition='top center')
            st.plotly_chart(fig, use_container_width=True)

    # === For contextual models: BERT and GPT ===
    elif model_type in ("BERT", "GPT"):
        if model_type == "BERT":
            model_name = 'bert-base-uncased'
        else:
            model_name = 'gpt2'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)

        st.markdown("#### Tokenization (with subword splitting)")
        tokens_sub = tokenizer.tokenize(sentence)
        st.write(tokens_sub)

        inputs = tokenizer(sentence, return_tensors="pt")
        outputs = model(**inputs)
        # Contextual embeddings shape: (batch_size, sequence_length, hidden_size)
        st.write("Contextual embedding shape:", outputs.last_hidden_state.shape)

        # For visualization, take a subset (first 10 tokens)
        num_tokens_vis = min(10, outputs.last_hidden_state.shape[1])
        token_embeds = outputs.last_hidden_state[0][:num_tokens_vis].detach().numpy()
        token_labels = tokens_sub[:num_tokens_vis]
        tsne_context = TSNE(n_components=2, perplexity=5, random_state=42).fit_transform(token_embeds)
        fig2 = px.scatter(x=tsne_context[:,0], y=tsne_context[:,1], text=token_labels,
                          title=f"t-SNE Visualization of {model_type} Contextual Embeddings",
                          labels={"x": "Dimension 1", "y": "Dimension 2"})
        fig2.update_traces(textposition='top center')
        st.plotly_chart(fig2, use_container_width=True)
        
        st.markdown("**Attention Map:** For detailed attention visualization, please refer to the BertViz tool. (Include your BertViz demo video in your presentation slides.)")

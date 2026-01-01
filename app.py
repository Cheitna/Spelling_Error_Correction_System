# # app.py
# import streamlit as st
# import pickle
# from nltk.metrics import edit_distance
# import nltk

# # -----------------------------
# # Download required NLTK data for POS tagging
# # -----------------------------
# @st.cache_resource
# def setup_nltk():
#     nltk.download("punkt")
#     nltk.download("averaged_perceptron_tagger")
#     nltk.download("wordnet")

# setup_nltk()

# # -----------------------------
# # Import your modules after NLTK setup
# # -----------------------------
# from corrections import detect_errors, display_tokens, FUNCTION_WORDS

# # -----------------------------
# # Page configuration
# # -----------------------------
# st.set_page_config(
#     page_title="Spelling & Grammar Correction System",
#     page_icon="✍️",
#     layout="centered"
# )

# # -----------------------------
# # Load vocabulary
# # -----------------------------
# @st.cache_resource
# def load_vocab():
#     with open("word_freq.pkl", "rb") as f:
#         word_freq = pickle.load(f)
#     vocab = set(word_freq.keys())
#     return word_freq, vocab

# word_freq, vocab = load_vocab()

# # -----------------------------
# # UI Header
# # -----------------------------
# st.title("✍️ Spelling & Grammar Correction System")
# st.markdown("""
# This application detects:
# - **Spelling errors** (red)
# - **Grammar-aware corrections** (green, e.g., "is rise → is rising")  
# Click on a highlighted word to see suggested corrections (sorted by minimum edit distance).
# """)

# # -----------------------------
# # User Input
# # -----------------------------
# st.subheader("📝 Enter Text")
# user_input = st.text_area(
#     "Type or paste your text (max 500 characters):",
#     height=120,
#     max_chars=500,
#     placeholder="Example: AI is helping in the medical domain"
# )

# # -----------------------------
# # Spell & Grammar Check Button
# # -----------------------------
# if st.button("🔍 Check Text", use_container_width=True):
#     if not user_input.strip():
#         st.warning("⚠️ Please enter some text before checking.")
#     else:
#         # Grammar-aware display tokens + indices
#         display_version, grammar_indices, grammar_map = display_tokens(user_input)

#         # Detect spelling errors
#         errors = detect_errors(user_input)
#         spelling_words = {err['word'].lower() for err in errors}  # red highlights

#         # Map word → suggestions sorted by edit distance
#         suggestion_map = {}
#         for err in errors:
#             suggestions = sorted(
#                 err.get('suggestions', []),
#                 key=lambda w: edit_distance(err['word'], w)
#             )
#             suggestion_map[err['word'].lower()] = suggestions

#         # Build highlighted text
#         highlighted_text = []
#         for i, word in enumerate(display_version):
#             lw = word.lower()
#             if lw in spelling_words and lw not in FUNCTION_WORDS:
#                 highlighted_text.append(f"[**:red[{word}]**](#)")
#             elif i in grammar_indices:
#                 highlighted_text.append(f"[**:green[{word}]**](#)")
#             else:
#                 highlighted_text.append(word)

#         st.subheader("🖍 Highlighted Text")
#         st.markdown(" ".join(highlighted_text))

#         # -----------------------------
#         # Error Details & Suggestions
#         # -----------------------------
#         st.subheader("📌 Error Details & Suggestions")
#         if not errors:
#             st.success("✅ No spelling errors detected!")
#         else:
#             for i, word in enumerate(display_version):
#                 lw = word.lower()
#                 if lw in spelling_words or i in grammar_indices:
#                     err_type = next((err['type'] for err in errors if err['word'].lower() == lw), "grammar")
#                     suggestions = suggestion_map.get(lw, [])
#                     with st.expander(f"❌ `{word}` — {err_type} error"):
#                         if suggestions:
#                             st.markdown("**Suggested corrections (sorted by edit distance):**")
#                             for s in suggestions:
#                                 dist = edit_distance(lw, s)
#                                 st.markdown(f"- **{s}** (Edit distance: {dist})")
#                         else:
#                             st.info("No suggestions available for grammar corrections.")

# # -----------------------------
# # Search / Explore Words
# # -----------------------------
# st.subheader("🔎 Search / Explore Words")
# search_word = st.text_input("Search a word in the corpus:", placeholder="Type a word")

# if search_word:
#     lw = search_word.lower()
#     if lw in vocab:
#         freq = word_freq.get(lw, 0)
#         st.success(f"✅ '{search_word}' exists (frequency: {freq})")
#     else:
#         st.error(f"❌ '{search_word}' not found in corpus.")

# # Scrollable vocabulary list
# with st.expander("📜 Vocabulary List (scrollable)"):
#     st.text_area("Vocabulary", value="\n".join(sorted(vocab)), height=200)

# # -----------------------------
# # Footer
# # -----------------------------
# st.caption("📘 MSc Artificial Intelligence | Spelling & Grammar Correction System")

# app.py

import streamlit as st

# -----------------------------
# Ensure NLTK resources exist BEFORE importing modules that use pos_tag
# -----------------------------
import nltk
for pkg, path in [
    ("punkt", "tokenizers/punkt"),
    ("averaged_perceptron_tagger", "taggers/averaged_perceptron_tagger"),
    ("wordnet", "corpora/wordnet"),
    ("omw-1.4", "corpora/omw-1.4")
]:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(pkg)

# -----------------------------
# Now safe to import modules
# -----------------------------
from nltk.metrics import edit_distance
import pickle
from corrections import detect_errors, display_tokens, FUNCTION_WORDS


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Spelling & Grammar Correction System",
    page_icon="✍️",
    layout="centered"
)

# -----------------------------
# Load vocabulary
# -----------------------------
@st.cache_resource
def load_vocab():
    with open("word_freq.pkl", "rb") as f:
        word_freq = pickle.load(f)
    vocab = set(word_freq.keys())
    return word_freq, vocab

word_freq, vocab = load_vocab()

# -----------------------------
# UI Header
# -----------------------------
st.title("✍️ Spelling & Grammar Correction System")
st.markdown("""
This application detects:
- **Spelling errors** (red)
- **Grammar-aware corrections** (green, e.g., "is rise → is rising")  
Click on a highlighted word to see suggested corrections (sorted by minimum edit distance).
""")

# -----------------------------
# User Input
# -----------------------------
st.subheader("📝 Enter Text")
user_input = st.text_area(
    "Type or paste your text (max 500 characters):",
    height=120,
    max_chars=500,
    placeholder="Example: AI is helping in the medical domain"
)

# -----------------------------
# Spell & Grammar Check Button
# -----------------------------
if st.button("🔍 Check Text", use_container_width=True):

    # -----------------------------
    # Ensure NLTK resources are available at runtime
    # -----------------------------
    import nltk
    for pkg, path in [
        ("punkt", "tokenizers/punkt"),
        ("averaged_perceptron_tagger", "taggers/averaged_perceptron_tagger"),
        ("wordnet", "corpora/wordnet"),
        ("omw-1.4", "corpora/omw-1.4")
    ]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg)

    # -----------------------------
    # Validate input
    # -----------------------------
    if not user_input.strip():
        st.warning("⚠️ Please enter some text before checking.")
    else:
        # Grammar-aware display tokens + indices
        display_version, grammar_indices, grammar_map = display_tokens(user_input)

        # Detect spelling errors
        errors = detect_errors(user_input)
        spelling_words = {err['word'].lower() for err in errors}  # red highlights

        # Map word → suggestions sorted by edit distance
        suggestion_map = {}
        for err in errors:
            suggestions = sorted(
                err.get('suggestions', []),
                key=lambda w: edit_distance(err['word'], w)
            )
            suggestion_map[err['word'].lower()] = suggestions

        # Build highlighted text
        highlighted_text = []
        for i, word in enumerate(display_version):
            lw = word.lower()
            if lw in spelling_words and lw not in FUNCTION_WORDS:
                highlighted_text.append(f"[**:red[{word}]**](#)")
            elif i in grammar_indices:
                highlighted_text.append(f"[**:green[{word}]**](#)")
            else:
                highlighted_text.append(word)

        st.subheader("🖍 Highlighted Text")
        st.markdown(" ".join(highlighted_text))

        # -----------------------------
        # Error Details & Suggestions
        # -----------------------------
        st.subheader("📌 Error Details & Suggestions")
        if not errors:
            st.success("✅ No spelling errors detected!")
        else:
            for i, word in enumerate(display_version):
                lw = word.lower()
                if lw in spelling_words or i in grammar_indices:
                    err_type = next((err['type'] for err in errors if err['word'].lower() == lw), "grammar")
                    suggestions = suggestion_map.get(lw, [])
                    with st.expander(f"❌ `{word}` — {err_type} error"):
                        if suggestions:
                            st.markdown("**Suggested corrections (sorted by edit distance):**")
                            for s in suggestions:
                                dist = edit_distance(lw, s)
                                st.markdown(f"- **{s}** (Edit distance: {dist})")
                        else:
                            st.info("No suggestions available for grammar corrections.")

# -----------------------------
# Search / Explore Words (after main functionality)
# -----------------------------
st.subheader("🔎 Search / Explore Words")
search_word = st.text_input("Search a word in the corpus:", placeholder="Type a word")

if search_word:
    lw = search_word.lower()
    if lw in vocab:
        freq = word_freq.get(lw, 0)
        st.success(f"✅ '{search_word}' exists (frequency: {freq})")
    else:
        st.error(f"❌ '{search_word}' not found in corpus.")

# Scrollable vocabulary list
with st.expander("📜 Vocabulary List (scrollable)"):
    st.text_area("Vocabulary", value="\n".join(sorted(vocab)), height=200)

# -----------------------------
# Footer
# -----------------------------
st.caption("📘 MSc Artificial Intelligence | Spelling & Grammar Correction System")

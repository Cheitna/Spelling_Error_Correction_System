# # user_preprocess.py
# """
# Grammar-aware preprocessing for user input
# - Lowercase
# - Tokenize
# - Lemmatize using WordNet
# - Apply grammar rules:
#     1. BE + VB → VBG (present participle)
#     2. HAS/HAVE/HAD + VB → VBN (past participle)
# - Returns:
#     - lemma tokens (for detection)
#     - display tokens (with grammar corrections applied)
#     - indices of grammar-corrected tokens
#     - grammar map {index: (original, corrected)}
# """

# import re
# import nltk
# from nltk.stem import WordNetLemmatizer
# from nltk import pos_tag
# from POS import to_present_participle, to_past_participle, BE_VERBS, HAS_VERBS

# # -----------------------------
# # NLTK downloads (safe if already present)
# # -----------------------------
# nltk.download("punkt")
# nltk.download("averaged_perceptron_tagger")
# nltk.download("wordnet")
# nltk.download("omw-1.4")

# lemmatizer = WordNetLemmatizer()

# # Function words: never flagged as errors
# FUNCTION_WORDS = {
#     "this","that","which","who","whom","whose",
#     "it","they","we","he","she",
#     "a","an","the",
#     "and","or","but",
#     "of","in","on","for","to","with","by","at",
#     "not"
# }

# # -----------------------------
# # Preprocessing for detection
# # -----------------------------
# def preprocess_user_input(text):
#     """
#     Lemmatize tokens for spelling detection.
#     """
#     text = text.lower()
#     tokens = re.findall(r'\b[a-z]+\b', text)
#     tagged_tokens = pos_tag(tokens)

#     processed = []
#     for word, tag in tagged_tokens:
#         if word in BE_VERBS | HAS_VERBS:
#             lemma = word
#         elif tag.startswith("V"):
#             lemma = lemmatizer.lemmatize(word, pos="v")
#         else:
#             lemma = lemmatizer.lemmatize(word)
#         processed.append(lemma)

#     return processed

# # -----------------------------
# # Grammar correction for display
# # -----------------------------
# def apply_display_grammar(tokens):
#     """
#     Convert lemma tokens into display-friendly form.
#     Handles:
#     - BE + VB → VBG
#     - HAS/HAVE/HAD + VB → VBN
#     Returns:
#         display_tokens: tokens with grammar applied
#         grammar_indices: indices of corrected tokens (for green highlight)
#         grammar_map: {index: (original_token, corrected_token)}
#     """
#     display_tokens = []
#     grammar_indices = []
#     grammar_map = {}
#     i = 0

#     while i < len(tokens):
#         tok = tokens[i]
#         prev = display_tokens[-1] if i > 0 else ""

#         # -----------------------
#         # BE + VB → VBG
#         # -----------------------
#         if prev in BE_VERBS:
#             corrected = to_present_participle(tok)
#             display_tokens.append(corrected)
#             grammar_indices.append(i)
#             grammar_map[i] = (tok, corrected)
#             i += 1
#             continue

#         # -----------------------
#         # HAS/HAVE/HAD + (optional NOT) + VB → VBN
#         # -----------------------
#         if prev in HAS_VERBS:
#             if tok == "not" and (i + 1 < len(tokens)):
#                 next_tok = tokens[i + 1]
#                 corrected = to_past_participle(next_tok)
#                 display_tokens.append(tok)          # keep 'not'
#                 display_tokens.append(corrected)    # corrected verb
#                 grammar_indices.append(i + 1)
#                 grammar_map[i + 1] = (next_tok, corrected)
#                 i += 2
#                 continue
#             else:
#                 corrected = to_past_participle(tok)
#                 display_tokens.append(corrected)
#                 grammar_indices.append(i)
#                 grammar_map[i] = (tok, corrected)
#                 i += 1
#                 continue

#         # -----------------------
#         # Default: no change
#         # -----------------------
#         display_tokens.append(tok)
#         i += 1

#     return display_tokens, grammar_indices, grammar_map

# # -----------------------------
# # Example test
# # -----------------------------
# if __name__ == "__main__":
#     sentences = [
#         "Bitcoin is rise this year",
#         "62% of Bitcoin has not move in a year",
#         "He was come late",
#         "They have move to another city"
#     ]

#     for sent in sentences:
#         lemma_tokens = preprocess_user_input(sent)
#         display_tokens_list, grammar_idx, grammar_map = apply_display_grammar(lemma_tokens)
#         print("Input:", sent)
#         print("Lemma tokens:", lemma_tokens)
#         print("Display tokens:", display_tokens_list)
#         print("Grammar corrected indices:", grammar_idx)
#         print("Grammar map:", grammar_map)
#         print("-" * 50)

# user_preprocess.py
"""
Grammar-aware preprocessing for user input
- Lowercase
- Tokenize
- Lemmatize using WordNet
- Apply grammar rules:
    1. BE + VB → VBG (present participle)
    2. HAS/HAVE/HAD + VB → VBN (past participle)
- Returns:
    - lemma tokens (for detection)
    - display tokens (with grammar corrections applied)
    - indices of grammar-corrected tokens
    - grammar map {index: (original, corrected)}
"""

import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from POS import to_present_participle, to_past_participle, BE_VERBS, HAS_VERBS

# -----------------------------
# Safe NLTK downloads
# -----------------------------
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

lemmatizer = WordNetLemmatizer()

# Function words: never flagged as errors
FUNCTION_WORDS = {
    "this","that","which","who","whom","whose",
    "it","they","we","he","she",
    "a","an","the",
    "and","or","but",
    "of","in","on","for","to","with","by","at",
    "not"
}

# -----------------------------
# Preprocessing for detection
# -----------------------------
def preprocess_user_input(text):
    """
    Lemmatize tokens for spelling detection.
    """
    text = text.lower()
    tokens = re.findall(r'\b[a-z]+\b', text)
    tagged_tokens = pos_tag(tokens)

    processed = []
    for word, tag in tagged_tokens:
        if word in BE_VERBS | HAS_VERBS:
            lemma = word
        elif tag.startswith("V"):
            lemma = lemmatizer.lemmatize(word, pos="v")
        else:
            lemma = lemmatizer.lemmatize(word)
        processed.append(lemma)

    return processed

# -----------------------------
# Grammar correction for display
# -----------------------------
def apply_display_grammar(tokens):
    """
    Convert lemma tokens into display-friendly form.
    Handles:
    - BE + VB → VBG
    - HAS/HAVE/HAD + VB → VBN
    Returns:
        display_tokens: tokens with grammar applied
        grammar_indices: indices of corrected tokens (for green highlight)
        grammar_map: {index: (original_token, corrected_token)}
    """
    display_tokens = []
    grammar_indices = []
    grammar_map = {}
    i = 0

    while i < len(tokens):
        tok = tokens[i]
        prev = display_tokens[-1] if i > 0 else ""

        # -----------------------
        # BE + VB → VBG
        # -----------------------
        if prev in BE_VERBS:
            corrected = to_present_participle(tok)
            display_tokens.append(corrected)
            grammar_indices.append(i)
            grammar_map[i] = (tok, corrected)
            i += 1
            continue

        # -----------------------
        # HAS/HAVE/HAD + (optional NOT) + VB → VBN
        # -----------------------
        if prev in HAS_VERBS:
            if tok == "not" and (i + 1 < len(tokens)):
                next_tok = tokens[i + 1]
                corrected = to_past_participle(next_tok)
                display_tokens.append(tok)          # keep 'not'
                display_tokens.append(corrected)    # corrected verb
                grammar_indices.append(i + 1)
                grammar_map[i + 1] = (next_tok, corrected)
                i += 2
                continue
            else:
                corrected = to_past_participle(tok)
                display_tokens.append(corrected)
                grammar_indices.append(i)
                grammar_map[i] = (tok, corrected)
                i += 1
                continue

        # -----------------------
        # Default: no change
        # -----------------------
        display_tokens.append(tok)
        i += 1

    return display_tokens, grammar_indices, grammar_map

# -----------------------------
# Module test (only runs if executed directly)
# -----------------------------
if __name__ == "__main__":
    sentences = [
        "Bitcoin is rise this year",
        "62% of Bitcoin has not move in a year",
        "He was come late",
        "They have move to another city"
    ]

    for sent in sentences:
        lemma_tokens = preprocess_user_input(sent)
        display_tokens_list, grammar_idx, grammar_map = apply_display_grammar(lemma_tokens)
        print("Input:", sent)
        print("Lemma tokens:", lemma_tokens)
        print("Display tokens:", display_tokens_list)
        print("Grammar corrected indices:", grammar_idx)
        print("Grammar map:", grammar_map)
        print("-" * 50)

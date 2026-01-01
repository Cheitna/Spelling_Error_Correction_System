# # POS.py
# """
# POS-based grammar correction module
# - Applies rule-based grammar:
#     1. BE + VB → VBG (present participle)
#     2. HAS/HAVE/HAD + VB → VBN (past participle)
# - Converts base verbs to present participle (-ing) and past participle (-ed) forms
# """

# import nltk
# from nltk import pos_tag, word_tokenize
# from nltk.stem import WordNetLemmatizer

# # -----------------------------
# # NLTK downloads (safe if already present)
# # -----------------------------
# nltk.download("punkt")
# nltk.download("averaged_perceptron_tagger")
# nltk.download("wordnet")

# # -----------------------------
# # Initialize lemmatizer
# # -----------------------------
# lemmatizer = WordNetLemmatizer()

# # -----------------------------
# # Auxiliary verbs
# # -----------------------------
# BE_VERBS = {"am", "is", "are", "was", "were"}
# HAS_VERBS = {"has", "have", "had"}

# # -----------------------------
# # Irregular verbs dictionaries
# # -----------------------------
# IRREGULAR_PRESENT = {
#     "be": "being",
#     "have": "having",
#     "do": "doing",
#     "go": "going",
#     "rise": "rising",
#     "come": "coming",
#     "happen": "happening",
#     "move": "moving",
#     "determine": "determining",
#     "use": "using"
# }

# IRREGULAR_PAST = {
#     "be": "been",
#     "have": "had",
#     "do": "done",
#     "go": "gone",
#     "rise": "risen",
#     "come": "come",
#     "move": "moved",
#     "determine": "determined",
#     "use": "used",
#     "happen": "happened"
# }

# # -----------------------------
# # Helper functions
# # -----------------------------
# def to_present_participle(verb):
#     """Convert a base verb to present participle (-ing) form"""
#     verb = verb.lower()
#     if verb in IRREGULAR_PRESENT:
#         return IRREGULAR_PRESENT[verb]
#     if verb.endswith("e"):
#         return verb[:-1] + "ing"
#     return verb + "ing"

# def to_past_participle(verb):
#     """Convert a base verb to past participle (-ed) form"""
#     verb = verb.lower()
#     if verb in IRREGULAR_PAST:
#         return IRREGULAR_PAST[verb]
#     if verb.endswith("e"):
#         return verb + "d"
#     return verb + "ed"

# # -----------------------------
# # Grammar correction function
# # -----------------------------
# def apply_rule_based_grammar(tokens):
#     """
#     Apply POS-tag-based grammar normalization:
#     - BE + VB → BE + VBG
#     - HAS/HAVE/HAD + VB → HAS/HAVE/HAD + VBN
#     """
#     corrected_tokens = []
#     tagged_tokens = pos_tag(tokens)

#     for i, (word, tag) in enumerate(tagged_tokens):
#         new_word = word  # default: unchanged

#         if i > 0:
#             prev_word = corrected_tokens[-1].lower()

#             # BE + VB → VBG
#             if prev_word in BE_VERBS:
#                 lemma = lemmatizer.lemmatize(word, pos="v")
#                 new_word = to_present_participle(lemma)

#             # HAS/HAVE/HAD + VB → VBN
#             elif prev_word in HAS_VERBS:
#                 lemma = lemmatizer.lemmatize(word, pos="v")
#                 new_word = to_past_participle(lemma)

#         corrected_tokens.append(new_word)

#     return corrected_tokens

# # -----------------------------
# # Test
# # -----------------------------
# if __name__ == "__main__":
#     test_sentences = [
#         "Bitcoin is rise this year",
#         "She has go to the market",
#         "He was come late",
#         "They have move to another city"
#     ]

#     for sent in test_sentences:
#         tokens = word_tokenize(sent)
#         corrected = apply_rule_based_grammar(tokens)
#         print("Input:    ", sent)
#         print("Corrected:", " ".join(corrected))
#         print("-" * 50)
# POS.py
"""
POS-based grammar correction module
- Applies rule-based grammar:
    1. BE + VB → VBG (present participle)
    2. HAS/HAVE/HAD + VB → VBN (past participle)
- Converts base verbs to present participle (-ing) and past participle (-ed) forms
"""

import nltk
from nltk import pos_tag, word_tokenize
from nltk.stem import WordNetLemmatizer

# -----------------------------
# Safe NLTK downloads
# -----------------------------
for pkg, path in [
    ("punkt", "tokenizers/punkt"),
    ("averaged_perceptron_tagger", "taggers/averaged_perceptron_tagger"),
    ("wordnet", "corpora/wordnet")
]:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(pkg)

# -----------------------------
# Initialize lemmatizer
# -----------------------------
lemmatizer = WordNetLemmatizer()

# -----------------------------
# Auxiliary verbs
# -----------------------------
BE_VERBS = {"am", "is", "are", "was", "were"}
HAS_VERBS = {"has", "have", "had"}

# -----------------------------
# Irregular verbs dictionaries
# -----------------------------
IRREGULAR_PRESENT = {
    "be": "being",
    "have": "having",
    "do": "doing",
    "go": "going",
    "rise": "rising",
    "come": "coming",
    "happen": "happening",
    "move": "moving",
    "determine": "determining",
    "use": "using"
}

IRREGULAR_PAST = {
    "be": "been",
    "have": "had",
    "do": "done",
    "go": "gone",
    "rise": "risen",
    "come": "come",
    "move": "moved",
    "determine": "determined",
    "use": "used",
    "happen": "happened"
}

# -----------------------------
# Helper functions
# -----------------------------
def to_present_participle(verb):
    """Convert a base verb to present participle (-ing) form"""
    verb = verb.lower()
    if verb in IRREGULAR_PRESENT:
        return IRREGULAR_PRESENT[verb]
    if verb.endswith("e"):
        return verb[:-1] + "ing"
    return verb + "ing"

def to_past_participle(verb):
    """Convert a base verb to past participle (-ed) form"""
    verb = verb.lower()
    if verb in IRREGULAR_PAST:
        return IRREGULAR_PAST[verb]
    if verb.endswith("e"):
        return verb + "d"
    return verb + "ed"

# -----------------------------
# Grammar correction function
# -----------------------------
def apply_rule_based_grammar(tokens):
    """
    Apply POS-tag-based grammar normalization:
    - BE + VB → BE + VBG
    - HAS/HAVE/HAD + VB → HAS/HAVE/HAD + VBN
    """
    corrected_tokens = []
    tagged_tokens = pos_tag(tokens)

    for i, (word, tag) in enumerate(tagged_tokens):
        new_word = word  # default: unchanged

        if i > 0:
            prev_word = corrected_tokens[-1].lower()

            # BE + VB → VBG
            if prev_word in BE_VERBS:
                lemma = lemmatizer.lemmatize(word, pos="v")
                new_word = to_present_participle(lemma)

            # HAS/HAVE/HAD + VB → VBN
            elif prev_word in HAS_VERBS:
                lemma = lemmatizer.lemmatize(word, pos="v")
                new_word = to_past_participle(lemma)

        corrected_tokens.append(new_word)

    return corrected_tokens

# -----------------------------
# Module test (only runs if executed directly)
# -----------------------------
if __name__ == "__main__":
    test_sentences = [
        "Bitcoin is rise this year",
        "She has go to the market",
        "He was come late",
        "They have move to another city"
    ]

    for sent in test_sentences:
        tokens = word_tokenize(sent)
        corrected = apply_rule_based_grammar(tokens)
        print("Input:    ", sent)
        print("Corrected:", " ".join(corrected))
        print("-" * 50)

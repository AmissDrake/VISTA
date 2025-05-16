import re
import os
from collections import defaultdict
from language_tool_python import LanguageTool
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import nltk
# Initialize grammar checker
tool = LanguageTool('en-US')

# Initialize the grammar correction model (prithivida)


# Initialize punctuation restoration model
model_name = "kredor/punctuate-all"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)
punctuation_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Initialize masked language model for context-based word fixes
from nltk import pos_tag, word_tokenize

# Download NLTK resources if not already downloaded
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

# Initialize grammar correction model and language tool
grammar_correction = pipeline("text2text-generation", model="prithivida/grammar_error_correcter_v1")
tool = LanguageTool('en-US')

def is_noun(word):
    """
    Checks if a word is tagged as a noun (NN or NNP).
    """
    tag = pos_tag([word])[0][1]
    return tag in ("NN", "NNP")

def meaning_based_correction(sentence):
    """
    Corrects sentences by:
    1. Replacing 'they' with 'the' if followed by a singular noun.
    2. Performing grammar correction.
    3. Polishing with LanguageTool.
    """
    words = word_tokenize(sentence)
    corrected_words = []

    for i, word in enumerate(words):
        # Check if 'they' is followed by a singular noun
        if word.lower() == "they" and (i + 1) < len(words) and is_noun(words[i + 1]):
            corrected_words.append("The")
        else:
            corrected_words.append(word)

    # Reconstruct the sentence
    corrected_sentence = ' '.join(corrected_words)

    # Step 2: Grammar correction using prithivida's model
    grammar_corrected = grammar_correction(corrected_sentence)[0]['generated_text']

    # Step 3: LanguageTool for additional grammar fixes
    final_sentence = tool.correct(grammar_corrected)

    return final_sentence


def restore_punctuation(text):
    """Restores punctuation using the punctuation model."""
    results = punctuation_pipeline(text)
    punctuated_text = ""
    last_end = 0
    for entity in results:
        start, end = entity['start'], entity['end']
        word = text[start:end]
        punct = entity['entity_group']
        if punct not in ("O", "0"):
            word += punct
        punctuated_text += text[last_end:start] + word
        last_end = end
    punctuated_text += text[last_end:]
    return punctuated_text


def load_word_freq(filename):
    """Load word frequencies from a file into a defaultdict"""
    word_freq = defaultdict(int)
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                word = parts[0].lower()
                try:
                    freq = int(parts[1])
                except ValueError:
                    freq = 1  # Default frequency for unparseable entries
                word_freq[word] += freq
    return word_freq


class TextCorrector:
    def __init__(self, corpus_path=r"C:\Users\sumed\Downloads\frequency_dictionary_en_82_765.txt", min_freq=100):
        if not os.path.exists(corpus_path):
            raise FileNotFoundError(
                f"Dictionary file '{corpus_path}' not found.\n"
                "Download from: https://github.com/wolfgarbe/SymSpell/raw/master/SymSpell/frequency_dictionary_en_82_765.txt"
            )
        self.word_freq = load_word_freq(corpus_path)
        self.min_freq = min_freq
        self.total_words = sum(self.word_freq.values())

    def edits1(self, word):
        """Generate all edits 1 edit away from word"""
        letters = 'abcdefghijklmnopqrstuvwxyz'
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        return set(
            [L + R[1:] for L, R in splits if R] +  # Deletes
            [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1] +  # Transposes
            [L + c + R[1:] for L, R in splits if R for c in letters] +  # Replaces
            [L + c + R for L, R in splits for c in letters]  # Inserts
        )

    def correct_word(self, word):
        """Get best correction for a single word"""
        original = word.lower()
        if self.word_freq[original] >= self.min_freq:
            return original
        candidates = self.edits1(original) | {original}
        valid = [w for w in candidates if self.word_freq[w] >= self.min_freq]
        return max(valid, key=lambda w: self.word_freq[w]) if valid else original

    def correct_sentence(self, sentence):
        """Correct full sentence with space handling"""
        tokens = re.findall(r'\w+|[^\w\s]', sentence)
        corrected = []
        for token in tokens:
            if token.isalpha():
                corrected_word = self.correct_word(token)
                corrected.extend(corrected_word.split())
            else:
                corrected.append(token)
        return ' '.join(corrected)


if __name__ == "__main__":
    try:
        corrector = TextCorrector()
        test_cases = [
            "i red a bok",
            "ired a book",
            "teh qick brown fx",
            "are yu workng hard",
            "I saw elephant in zoo. She bought umbrella. He is best player.",
            "He will did the work before he goes to home.",
            "They cat is on the roof.",
            "They think this book is interesting.",
            "They person is very kind."
        ]

        for text in test_cases:
            print("Input:       ", text)
            spell_checked = corrector.correct_sentence(text)
            print("Corrected:   ", spell_checked)
            context_corrected = meaning_based_correction(spell_checked)
            print("Context:     ", context_corrected)
            punctuated = restore_punctuation(context_corrected)
            print("Punctuated:  ", punctuated)
            print("Grammar:     ", tool.correct(punctuated))    
            print("-" * 50)

    except FileNotFoundError as e:
        print(f"Error: {e}")
 
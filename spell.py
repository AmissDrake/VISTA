import re
import os
from collections import defaultdict
from language_tool_python import LanguageTool
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from transformers import pipeline
# Initialize grammar checker
tool = LanguageTool('en-US')

grammar_correction = pipeline("text2text-generation", model="prithivida/grammar_error_correcter_v1")


fill_mask = pipeline("fill-mask", model="bert-base-uncased")


def meaning_based_correction(sentence):
    """Corrects sentences using masked language modeling and grammar correction."""
    
    # Step 1: Mask common mistakes
    masked_sentence = sentence
    if "[MASK]" not in masked_sentence:
        corrected_sentence = masked_sentence
    else:
        # Get suggestions for masked words
        suggestions = fill_mask(masked_sentence)

        # Pick the best suggestions for each mask
        corrected_sentence = masked_sentence
        for suggestion in suggestions:
            corrected_sentence = corrected_sentence.replace("[MASK]", suggestion['token_str'], 1)
    
    # Step 2: Grammar correction using prithivida's model
    grammar_corrected = grammar_correction(corrected_sentence)[0]['generated_text']

    # Step 3: LanguageTool for additional grammar fixes
    final_sentence = tool.correct(grammar_corrected)
    return final_sentence
def context_based_correction(sentence):
    """Corrects sentences using regex for determiners and prithivida model."""
    
    # Step 1: Basic determiner and pronoun fixes
    sentence = re.sub(r"\b[Tt]hey (\w+)\b", r"The \1", sentence)                    
    sentence = re.sub(r"\b[Aa]n ([^aeiouAEIOU])", r"A \1", sentence)
    sentence = re.sub(r"\b[Aa] ([aeiouAEIOU])", r"An \1", sentence)

    # Step 2: Grammar correction using prithivida's model
    grammar_corrected = grammar_correction(sentence)[0]['generated_text']

    # Step 3: LanguageTool for additional grammar fixes
    end_sentence = tool.correct(grammar_corrected)
    return end_sentence

# Initialize punctuation restoration model
model_name = "kredor/punctuate-all"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)
punctuation_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

def restore_punctuation(text):
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
    with open(r"C:\Users\sumed\Downloads\frequency_dictionary_en_82_765.txt", 'r', encoding='utf-8') as f:
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

    def split_word(self, word):
        """Attempt to split merged words"""
        for i in range(1, len(word)):
            left, right = word[:i], word[i:]
            if (self.word_freq[left] >= self.min_freq and 
                self.word_freq[right] >= self.min_freq):
                return f"{left} {right}"
        return None

    def correct_word(self, word):
        """Get best correction for a single word"""
        original = word.lower()
        
        # Return valid words as-is
        if self.word_freq[original] >= self.min_freq:
            return original
            
        # Try space splitting first
        split = self.split_word(original)
        if split:
            return split
            
        # Generate and filter candidates
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
            "ired abook",               
            "teh qick brown fx",        
            "are yu workng hard",       
            "He will did the work before he goes to home.",
            "he said hello how are you i am fine she replied im good",
            "They cat is on the roof.",
            "they like this book.",
            "They person is very kind."
        ]

        for text in test_cases:
            print("\nInput:       ", text)

            # Step 1: Spell correction
            spell_checked = corrector.correct_sentence(text)
            print("Corrected:   ", spell_checked)

            # Step 2: Meaning-based correction
            results = meaning_based_correction(spell_checked)
            print("Contextual:  ", results) 
            context = context_based_correction(results)
            print("Context:     ", context)
            punctuated = restore_punctuation(context)
            print("Punctuated:  ", punctuated)
            # Step 3: Grammar correction
            grammar_corrected = tool.correct(punctuated)
            print("Grammar:     ", grammar_corrected)
            print("-" * 50)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Download the frequency file from:")
        print("https://github.com/wolfgarbe/SymSpell/raw/master/SymSpell/frequency_dictionary_en_82_765.txt")

import re
import os
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline



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
    def __init__(self, corpus_path=r"freq.txt", min_freq=100):
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
     original = word.lower()
     if self.word_freq[original] >= self.min_freq:
        return original
     candidates = self.edits1(original) | {original}
     valid = [w for w in candidates if self.word_freq[w] >= self.min_freq]
     if not valid:
        return original
     best = max(valid, key=lambda w: self.word_freq[w])
    # Only replace if significantly more frequent
     if self.word_freq[best] > self.word_freq[original] * 1.2:
        return best
     return original

if __name__ == "__main__":
    try:
        corrector = TextCorrector()
        test_cases = [
            "lik",
        ]

        for text in test_cases:
            print("Input:       ", text)
            spell_checked = corrector.correct_word(text)
            print(spell_checked)
    except FileNotFoundError as e:
        print(f"Error: {e}")
 
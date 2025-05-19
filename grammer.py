import nltk
import re
from transformers import pipeline
from language_tool_python import LanguageTool

# Download required NLTK data
nltk.download('punkt')

# Initialize grammar correction models
grammar_model = pipeline("text2text-generation", model="prithivida/grammar_error_correcter_v1")
tool = LanguageTool('en-US')

def remove_echo(input_text, output_text):
    """
    Removes repetition of the original text in the model output.
    """
    input_text = input_text.strip().lower()
    output_text = output_text.strip()
    if output_text.lower().startswith(input_text):
        return output_text[len(input_text):].strip()
    return output_text

def grammar_correct(text):
    """
    Performs grammar correction using both transformer and LanguageTool.
    """
    # Step 1: Grammar correction using transformer
    model_output = grammar_model(text)[0]['generated_text']
    model_output = remove_echo(text, model_output)

    # Step 2: Final polishing using LanguageTool
    final_output = tool.correct(model_output)
    return final_output

if __name__ == "__main__":
    test_sentences = [
        "they car is fast",
        "he eat pizza",
    ]

    for sentence in test_sentences:
        corrected_sentence = grammar_correct(sentence)
        print(f"Original: {sentence}")
        print(f"Corrected: {corrected_sentence}")
        print("-" * 40)         

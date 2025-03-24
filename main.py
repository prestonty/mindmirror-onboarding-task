"""
STUDENT NAME: Preston Tom-Ying
STUDENT ID: 21065097
"""

from transformers import pipeline
import json

# Load your chosen models here
def load_emotion_model():
    # Research and return a pre-trained emotion detection pipeline
    classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)
    return  classifier

def load_summarization_model():
    # Research and return a pre-trained summarization pipeline
    pass

# Process journal entries and return emotion predictions
def detect_emotions(text_entries, emotion_model):
    # Implement logic to process entries and extract emotions
    emotions = emotion_model(text_entries)
    
    return emotions

# Generate summaries for journal entries
def summarize_entries(entries, summarizer):
    # Implement logic to summarize each entry
    pass

# Use later
# def read_entries(file_name):
#     with open(file_name + '.json') as f:
#         data = json.load(f)
#     return data

if __name__ == '__main__':
    # Load models
    emotion_model = load_emotion_model()
    summarizer = load_summarization_model()

    # Example input
    journal_entries = [
        "Felt anxious about my exam, but happy after completing it.",
        "It rained all day and I stayed inside feeling calm.",
    ]

    # Apply pipelines
    emotions = detect_emotions(journal_entries, emotion_model)
    summaries = summarize_entries(journal_entries, summarizer)

    # Output results
    print("Emotion Predictions:", emotions)
    print("Summaries:", summaries)
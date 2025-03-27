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
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return summarizer

# Process journal entries and return emotion predictions
def detect_emotions(text_entries, emotion_model):
    # Implement logic to process entries and extract emotions
    emotions = emotion_model(text_entries)
    
    return emotions

# Generate summaries for journal entries
def summarize_entries(entries, summarizer):
    return summarizer(entries, max_length=70, min_length=30, do_sample=False)

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
        "Today was absolutely incredible! I finally got to visit Canada’s Wonderland, and it was everything I hoped for—and more. The moment we walked through the entrance, I was hit with the sound of excited screams, the smell of funnel cakes, and the sight of towering roller coasters in the distance. We started the day with Leviathan, one of the tallest and fastest coasters in the park. The climb to the top felt like forever, and my heart was pounding the whole way up. The drop? Unreal. I could barely catch my breath as we sped down at what felt like the speed of light. It was terrifying but also insanely fun. Next up was Behemoth, which had even more airtime than I expected. The way my stomach dropped on each hill was both thrilling and slightly nauseating—but totally worth it. We also tried Yukon Striker, the park’s massive dive coaster. Hanging over the edge before the drop was pure suspense, but the rush that followed was insane. After all the adrenaline, we took a break and grabbed some food—classic poutine and a massive turkey leg. I swear the poutine tasted even better after those coasters. We wandered around a bit, played a few carnival games (and lost, of course), then decided to cool off at Splash Works. The lazy river was a perfect break before we tackled more rides.",
        "Tonight, I went to a haunted house, and it was terrifying in the best way. The moment we stepped inside, eerie music played, and fog filled the air. Shadows moved, and whispers came from nowhere. Suddenly, a figure jumped out—my heart nearly stopped! We navigated pitch-black hallways, dodging creepy dolls, eerie clowns, and chainsaw-wielding maniacs. At one point, I swore something grabbed my arm. The final room was the worst—flashing lights, deafening screams, and no clear way out. When we finally escaped, I was breathless but laughing. Definitely the scariest (and most fun) night of the year!",
    ]

    # Apply pipelines
    emotions = detect_emotions(journal_entries, emotion_model)
    summaries = summarize_entries(journal_entries, summarizer)

    # Output results
    print("Emotion Predictions:", emotions)
    print("Summaries:", summaries)
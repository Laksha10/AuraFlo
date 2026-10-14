# sentiment.py
# VADER sentiment + a pain-word safety check -> one label the routine can act on.

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import config

_analyzer = SentimentIntensityAnalyzer()


def analyze_sentiment(text):
    """
    Returns (label, compound_score).
    label is one of: "no_answer", "pain", "negative", "neutral", "positive"
    """
    if text.strip() == "":
        return "no_answer", 0.0

    compound = _analyzer.polarity_scores(text)["compound"]

    # Safety first: pain words win over whatever VADER says.
    lowered = text.lower()
    for word in config.PAIN_WORDS:
        if word in lowered:
            return "pain", compound

    if compound <= config.NEGATIVE_THRESHOLD:
        return "negative", compound
    if compound >= config.POSITIVE_THRESHOLD:
        return "positive", compound
    return "neutral", compound
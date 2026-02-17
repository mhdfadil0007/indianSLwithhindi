import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

nltk.download('averaged_perceptron_tagger_eng')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('stopwords')

AUXILIARY_VERBS = {"am", "is", "are", "was", "were", "be", "been", "being"}

lemmatizer = WordNetLemmatizer()


def get_wordnet_pos(tag):
    if tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('R'):
        return wordnet.ADV
    return None


def safe_lemmatize(word, pos_tag):
    if word.lower() in AUXILIARY_VERBS:
        return word.lower()

    wn_pos = get_wordnet_pos(pos_tag)
    if wn_pos:
        return lemmatizer.lemmatize(word.lower(), wn_pos)

    return word.lower()

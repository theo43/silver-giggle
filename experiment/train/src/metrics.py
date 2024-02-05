from nltk.translate.bleu_score import sentence_bleu
from nltk import word_tokenize
import nltk
from rouge import Rouge


nltk.download()


def calculate_bleu_score(candidate, reference):
    """
    candidate, reference: generated and ground-truth sentences
    """
    reference = word_tokenize(reference)
    candidate = word_tokenize(candidate)
    score = sentence_bleu(reference, candidate)
    return score


def calculate_rouge_score(candidate, reference):
    """
    candidate, reference: generated and ground-truth sentences
    """
    rouge = Rouge()
    scores = rouge.get_scores([candidate], reference)
    return scores

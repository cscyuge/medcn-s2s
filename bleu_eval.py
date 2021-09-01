from nltk.translate.bleu_score import sentence_bleu
import nltk

def get_sentence_bleu(candidate, reference):
    score = sentence_bleu(reference, candidate)
    return score


def count_score(candidate, reference, config):
    avg_score = 0
    for k in range(len(candidate)):
        reference_ = reference[k]
        for m in range(len(reference_)):
            reference_[m] = config.tokenizer.tokenize(reference_[m])
        candidate[k] = config.tokenizer.tokenize(candidate[k])
        try:
            avg_score += get_sentence_bleu(candidate[k], reference_)/len(candidate)
        except:
            print(candidate[k])
            print(reference[k])
    return avg_score

def main():
    pass


if __name__ == '__main__':
    main()
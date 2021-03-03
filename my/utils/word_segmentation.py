import os
import sys
import jieba


def word_segmentation_jieba(input_p, out_p):
    with open(input_p, 'r', encoding='utf-8') as rf, open(out_p, 'w+', encoding='utf-8') as wf:
        for line in rf:
            text = line.strip()
            words = jieba.cut(text, HMM=False) # turn off new word discovery (HMM-based)
            wf.write(' '.join(words) + '\n')

def word_segmentation(input_p, out_p):
    with open(input_p, 'r', encoding='utf-8') as rf, open(out_p, 'w+', encoding='utf-8') as wf:
        for line in rf:
            text = line.strip()
            words = []
            for word in text:
                words.append(word)
            wf.write(' '.join(words) + '\n')

if __name__ == '__main__':
    input_p = sys.argv[1]
    out_p = sys.argv[2]
    word_segmentation(input_p, out_p)

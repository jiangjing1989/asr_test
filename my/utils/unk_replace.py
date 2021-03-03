import os
import sys


def main(word_p, lexicon_p, out_p):
    lexicon_l = []
    with open(lexicon_p, 'r', encoding='utf-8') as f:
        for line in f:
            index, lexicon = line.strip().split(' ')
            lexicon_l.append(lexicon)
    with open(word_p, 'r', encoding='utf-8') as rf, open(out_p, 'w+', encoding='utf-8') as wf:
        for line in rf:
            word_l = line.strip().split(' ')
            new_l = []
            for word in word_l:
                if word not in lexicon_l:
                    new_l.append('unk')
                else:
                    new_l.append(word)
            wf.write(' '.join(new_l) + '\n')


if __name__ == '__main__':
    word_p = sys.argv[1]
    lexicon_p = sys.argv[2]
    out_p = sys.argv[3]
    main(word_p, lexicon_p, out_p)


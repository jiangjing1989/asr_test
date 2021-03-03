import os
import sys
from collections import defaultdict

def main(seg_p, out_p, num=400):
    word2num = defaultdict(lambda:0)
    with open(seg_p, 'r', encoding='utf-8') as f:
        for line in f:
            words = line.strip().split(' ')
            for word in words:
                word2num[word] += 1
    word_num = sorted(word2num.items(), key = lambda kv:(kv[1], kv[0]), reverse = True)
    with open(out_p, 'w+', encoding='utf-8') as f:
        index = 0
        for word, _ in word_num[:num]:
            f.write('{} {}\n'.format(index, word))
            index += 1
        f.write('{} unk\n'.format(index))


if __name__ == '__main__':
    seg_p = sys.argv[1]
    out_p = sys.argv[2]
    main(seg_p, out_p)

            

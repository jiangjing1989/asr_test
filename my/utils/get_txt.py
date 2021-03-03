import os
import sys
import glob

def main(input_dir, out_p):
    with open(out_p, 'w+', encoding='utf-8') as wf:
        for txt in glob.iglob(input_dir + '/*/*txt'):
            print(txt)
            with open(txt, 'r', encoding='utf-8') as rf:
                lines = rf.readlines()
            context = lines[1].split('\t')[1]
            wf.write(context + '\n')

if __name__ == '__main__':
    input_dir = sys.argv[1]
    out_p = sys.argv[2]
    main(input_dir, out_p)
        

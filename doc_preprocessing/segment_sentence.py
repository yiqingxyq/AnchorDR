import re
import sys
import random
import spacy
import multiprocessing
from collections import Counter
from multiprocessing import Pool

nlp = None

def init():
    global nlp
    nlp = spacy.load('en_core_web_lg', disable=['tagger', 'lemmatizer', 'ner', 'textcat'])
    #nlp = English()
    #sentencizer = nlp.create_pipe("sentencizer")
    #nlp.add_pipe(sentencizer)
    #nlp.add_pipe(LanguageDetector())
    random.seed(43)


def segment(line):
    if line[:4] == 'Url:':
        return line
    else:
        global nlp
        line = line.strip()
        need_segment = False
        l = len(line.split())
        if l > 256:
            need_segment = True
        elif l <= 16:
            need_segment = False
        else:
            # randomly keep some long documents
            need_segment = random.random() < 0.98
        if line:
            if need_segment:
                doc = nlp(line)
                # don't break short sentences
                sents = []
                for sent in doc.sents:
                    sent = str(sent.text).strip()
                    sents.append(sent)
                return '\n'.join(sents) + '\n'
            else:
                return line + '\n'
        return '\n'

def main():
    with Pool(processes=multiprocessing.cpu_count(), initializer=init) as pool:
        for text in pool.imap(segment, sys.stdin, chunksize=1024):
            sys.stdout.write(text)

if __name__ == '__main__':
    main()
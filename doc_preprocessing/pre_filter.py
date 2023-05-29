import sys
import nltk
import re
from collections import Counter
#nltk.download('words')
from nltk.corpus import words
# import enchant
import multiprocessing
from multiprocessing import Pool
import unidecode

def acsii_only(s):
    return unidecode.unidecode(s)


quote_sets = None
end_sents = None
filter_prefix = None
email_re = None
url_re1 = None
url_re2 = None

def is_valid_sent(line):
    global quote_sets
    global end_sents
    global filter_prefix
    global email_re
    global url_re1
    global url_re2
    line = line.strip()
    l = len(line)
    letters = sum(c.isalpha() for c in line)
    count_quote = sum(c in quote_sets for c in line)
    count_end = sum(c in end_sents for c in line)
    count = Counter(line)
    if l > 1000000 or l < 4:
        return False
    if count['\\'] / l > 0.05:  # filter latex math equations
        return False
    if count['|'] / l > 0.05 or line[0] == '|':  # filter remaining tables
        return False
    if (letters + count_quote + count_end) / l < 0.7 or letters < 2: # too few letters
        return False
    has_url = False
    lower_line = line.lower()
    if l < 100:
        has_url = has_url or len(email_re.findall(lower_line)) > 0
        has_url = has_url or len(url_re1.findall(lower_line)) > 0
        has_url = has_url or len(url_re2.findall(lower_line)) > 0
    if has_url:
        return False
    words = line.split()
    n = 0
    nword = len(words)
    if nword <= 10 or l < 50 or letters < 50:
        if words[0].lower() in filter_prefix:
            return False
        if words[0].startswith('<') or words[0].startswith('!'):
            return False
        if 'all rights reserved' in lower_line:
            return False
        is_quote = words[0][0] in quote_sets or words[-1][-1] in quote_sets
        #is_title = words[0][0].isupper() and words[-1][0].isupper() and len(words) > 3
        is_sent = words[-1][-1] in end_sents
        if is_sent or is_quote:
            return True
        else:
            return False
    return True

def init():
    global quote_sets
    global end_sents
    global filter_prefix
    global email_re
    global url_re1
    global url_re2
    quote_sets = set(['“', "'", '"', '‘', '”', '’'])
    end_sents = set(['.', ',', '?', '!', '-', '…', ';', ':'])
    filter_prefix = set(['chapter', 'part', 'copyright', 'copyrighted'])
    email_p = r'([a-z0-9_\-\.]+)@((\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.)|(([a-z0-9\-]+\.)+))([a-z]{2,4}|[0-9]{1,3})(\]?)'
    url1 = r'(?:(?:https?|ftp):\/\/|www\.)[-a-z0-9+&@#\/%?=~_|!:,.;]*[-a-z0-9+&@#\/%=~_|]'
    # simple fooo-bar.com cases without the prefix
    url2 = r'\b[^$s]{3}[^$s]*(\.\w)*\.(?:com|net|org|edu|gov|cn)(\/\w*)*\/?'
    email_re = re.compile(email_p)
    url_re1 = re.compile(url1)
    url_re2 = re.compile(url2)

def process(line):
    if line[:4] == 'Url:':
        return line
    else:
        line = line.strip()
        if not line:
            return "\n"
        try:
            if is_valid_sent(line):
                return line + '\n'
        except:
            pass
        return None

def main():
    with Pool(processes=multiprocessing.cpu_count(), initializer=init) as pool:
        for text in pool.imap(process, sys.stdin, chunksize=512):
            if text:
                sys.stdout.write(text)

if __name__ == '__main__':
    main()
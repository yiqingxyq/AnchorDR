import sys
import re
import random
import unicodedata

min_char_per_doc = 50

min_word_per_sent = 2
# alpha is for all kinds of lanuage
min_alpha_frac = 0.6 # 0.7 for 16g
#min_alpha_frac = 0.7
# some very long words are noise
max_word_len = 40

random_p = 0.01

ascii_letters = set(list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'))

def main():
    f_fliter = open(sys.argv[1], 'w', encoding='utf-8')
    is_new_doc = True
    cnt_char_cur_doc = 0
    cnt_ascii_letter_cur_doc = 0
    buf = []
    dup_set = set()
    word_buf = []
    cnt_filter = 0
    cnt_total = 0

    for line in sys.stdin:
        if line[:4] == 'Url:':
            sys.stdout.write(line)
        else:
            line = line.strip()
            has_sep = False
            if not line:
                has_sep = True
            else:
                words = line.split()
                cur_cnt_word = len(words)
                # ignore spaces
                reduced_line = ''.join(words)
                wc = len(reduced_line)
                cur_cnt_char = sum(c.isalpha() for c in reduced_line)

                alpha_frac = float(cur_cnt_char) / wc if wc else 0
                expand_word = line.replace('-', ' - ').replace('/', ' / ').replace('_', ' _ ').replace('"', ' " ').split()
                cur_max_word_len = max(len(w) for w in expand_word) if expand_word else 0
                # soft boundary
                rand_p = random.uniform(-random_p, random_p)
                alpha_frac += rand_p
                if cur_cnt_word >= min_word_per_sent and alpha_frac > min_alpha_frac and cur_max_word_len < max_word_len:
                    buf.append(line)
                    cnt_char_cur_doc += cur_cnt_char
                    word_buf += words
                else:
                    cnt_filter += 1
                # has_sep = True
                    f_fliter.write(line + '\n')
            if has_sep:
                # end of current document
                dup_str = ' '.join(word_buf)
                if dup_str and dup_str not in dup_set and len(buf) > 0 and cnt_char_cur_doc > min_char_per_doc:
                    sys.stdout.write('\n'.join(buf) + '\n')
                else:
                    cnt_filter += len(buf)
                    f_fliter.write('\n'.join(buf) + '\n')
                sys.stdout.write('\n')
                dup_set.add(dup_str)
                buf = []
                cnt_char_cur_doc = 0
                cnt_ascii_letter_cur_doc = 0
                word_buf = []
            cnt_total += 1
            if cnt_total % 1000000 == 0:
                print('filtered {} lines, total lines {}, fraction {}'.format(cnt_filter, cnt_total, cnt_filter / cnt_total), file=sys.stderr)
    if buf:
        dup_str = ' '.join(word_buf)
        #dup_str = re.sub('[\d]+', ' ', dup_str)
        if dup_str not in dup_set and cnt_char_cur_doc > min_char_per_doc:
            sys.stdout.write('\n'.join(buf) + '\n')
        else:
            f_fliter.write('\n'.join(buf) + '\n')
            cnt_filter += len(buf)
        sys.stdout.write('\n')
    
    print('filtered {} lines, total lines {}, fraction {}'.format(cnt_filter, cnt_total, cnt_filter / cnt_total), file=sys.stderr)
    f_fliter.close()

if __name__ == '__main__':
    main()
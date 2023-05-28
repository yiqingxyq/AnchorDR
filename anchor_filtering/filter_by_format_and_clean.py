import sys
import json
import re 
import pickle
from tqdm import tqdm

placeholder_tags = {'math': 'formula', 'code': 'codice'}

# Match HTML placeholder tags
placeholder_tag_patterns = [
    (re.compile(r'<\s*%s(\s*| [^>]+?)>.*?<\s*/\s*%s\s*>' % (tag, tag), re.DOTALL | re.IGNORECASE),
     repl) for tag, repl in placeholder_tags.items()
]

# Matches space
spaces = re.compile(r' {2,}')

# Matches dots
dots = re.compile(r'\.{4,}')


def clean(text):
    """
    Removes irrelevant parts from :param: text.
    """
    # Expand placeholders
    for pattern, placeholder in placeholder_tag_patterns:
        index = 1
        for match in pattern.finditer(text):
            text = text.replace(match.group(), '%s_%d' % (placeholder, index))
            index += 1

    text = text.replace('<<', '«').replace('>>', '»')

    #############################################

    # Cleanup scripts in WebExtractor.py 
    # text = '\n' + text + '\n'
    text = text.replace('\n', ' ')
    text = text.replace('\t', ' ')
    text = spaces.sub(' ', text)
    text = dots.sub('...', text)
    text = re.sub(' (,:\.\)\]»)', r'\1', text)
    text = re.sub('(\[\(«) ', r'\1', text)
    text = text.replace(',,', ',').replace(',.', '.')

    text = re.sub(r'(^[ \t]+)|([ \t]+$)', '', text, flags=re.MULTILINE)
    # text = re.sub(r'\\', ' ', text)
    text = re.sub(r' \*([^\s])', r' \1', text)
    text = re.sub(r'(\w)\n([a-z])', r'\1 \2', text)
    text = re.sub(r'^\|.*$', '', text, flags=re.MULTILINE)


    # Cleanup scripts in preprocess.sh 
    text = text.replace('\x00','')
    text = re.sub(r'^>+', ' ', text)
    text = re.sub(r'^-+', ' ', text)
    text = re.sub(r'^\++', ' ', text)
    text = re.sub(r"[^a-zA-Z0-9 ]", "", text)
    text = ' '.join(text.split())

    text0 = text.replace(' ','')
    # numerical, special symbols and single letter 
    if text0.isnumeric() or len(text0) <= 1:
        return ''

    return text

def get_domain(link):
    if 'https://' in link:
        domain = link[8:].split('/')[0]
    elif 'http://' in link:
        domain = link[7:].split('/')[0]
    else:
        domain = ''

    return domain 

def main():
    out_file = sys.argv[1]
    stat_dir = sys.argv[2]

    anchor2freq = {}
    doc_num = 0
    doc_with_anchor_num = 0
    anchor_num = 0
    in_doc_anchor_num = 0
    selected_anchor_num = 0
    with open(out_file, 'w+') as fout:
        for line in sys.stdin:
            try:
                d = json.loads(line)
            except:
                continue
            url = d['url']
            anchor_num += len(d['anchors'])

            doc_num += 1

            tgt_domain = get_domain(url)
            if not tgt_domain:
                continue
            
            anchor_texts = set()
            total_anchor_texts = set()
            for anchor in d['anchors']:
                total_anchor_texts.add(anchor[2])
                if anchor[3] != '':
                    if anchor[3] != '0':
                        # filter out header and footer
                        continue
                
                src_domain = get_domain(anchor[0])
                if not src_domain or src_domain == tgt_domain:
                    # filter out in-domain anchors
                    continue

                text = anchor[2]
                text = clean(text)
                if text:
                    anchor_texts.add(text)
                    if text in anchor2freq:
                        anchor2freq[text] += 1
                    else:
                        anchor2freq[text] = 0
            
            in_doc_anchor_num += len(total_anchor_texts)
            if len(anchor_texts) > 0:
                doc_with_anchor_num += 1
                selected_anchor_num += len(anchor_texts)
                tmp_dict = {'url':url, 'anchor_text':list(anchor_texts)}
                # fout.write(tmp_dict)
                fout.write('{}\n'.format(json.dumps(tmp_dict)))
    
    print('finished processing', str(doc_with_anchor_num), '/', str(doc_num), 'docs')
    print('finished processing', str(selected_anchor_num), '/', str(in_doc_anchor_num), 'anchors')

    with open(stat_dir + '/anchor_rule_filtered_step1_data.txt','w') as fout:
        fout.write('doc_with_anchor: ' + str(doc_with_anchor_num) + '\n')
        fout.write('total doc num: ' + str(doc_num) + '\n')
        fout.write('selected anchor num: ' + str(selected_anchor_num) + '\n')
        fout.write('total anchor num (inside train/valid set): ' + str(in_doc_anchor_num) + '\n')

    with open(stat_dir + '/anchor2freq.pkl','wb') as fout:
        pickle.dump(anchor2freq, fout)

if __name__ == '__main__':
    main()
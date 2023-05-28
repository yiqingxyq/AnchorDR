import sys
import json
import re 
import pickle
from tqdm import tqdm

filter_out_keyword_set = {
    'website',
    'official website',
    'original',
    'view website',
    'visit website',
    'visit our website',
    'visit website',
    'visit site',
    'home',
    'home page',
    'homepage',
    'about',
    'about us',
    'here',
    'this',
    'this article',
    'this page',
    'click here',
    'link',
    'source link',
    'offsite link',
    'this link',
    'more',
    'more info',
    'more information',
    'view more',
    'learn more',
    'read more',
    'see more',
    'find out more',
    'english',
    'en',
    'download',
    'save',
    'login',
    'sign in',
    'sign up',
    'register',
    'reply',
}
filter_out_keyword_set = {x.replace(' ','') for x in filter_out_keyword_set}

def main():
    print(filter_out_keyword_set)
    out_file = sys.argv[1]
    out_dir = sys.argv[2]

    doc_num = 0
    doc_with_anchor_num = 0
    anchor_num = 0
    selected_anchor_num = 0
    url2anchor = {}
    # with open(out_file, 'w+') as fout:
    for line in sys.stdin:
        try:
            d = json.loads(line)
        except:
            continue
        url = d['url']
        doc_num += 1
        anchor_set = set(d['anchor_text'])
        anchor_num += len(anchor_set)

        anchor_texts = []
        for anchor in anchor_set:
            if len(anchor.split()) > 64:
                continue
            if anchor.replace(' ','').lower() in filter_out_keyword_set:
                continue 
            anchor_texts.append(anchor)

        if anchor_texts:
            doc_with_anchor_num += 1
            selected_anchor_num += len(anchor_texts)
            url2anchor[url] = anchor_texts
            # tmp_dict = {'url':url, 'anchor_text':anchor_texts}
            # fout.write('{}\n'.format(json.dumps(tmp_dict)))

    print('finished processing', str(doc_with_anchor_num), '/', str(doc_num), 'docs')
    print('finished processing', str(selected_anchor_num), '/', str(anchor_num), 'anchors')

    with open(out_dir + '/anchor_rule_filtered_step2_data.txt','w') as fout:
        fout.write('doc_with_anchor: ' + str(doc_with_anchor_num) + '\n')
        fout.write('total doc num: ' + str(doc_num) + '\n')
        fout.write('selected anchor num: ' + str(selected_anchor_num) + '\n')
        fout.write('total anchor num (inside train/valid set): ' + str(anchor_num) + '\n')

    with open(out_file, 'wb') as fout:
        pickle.dump(url2anchor, fout)

if __name__ == '__main__':
    main()
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert MS MARCO qrels to TREC qrels.')
    parser.add_argument('--input', required=True, default='', help='Input MS MARCO qrels file.')
    parser.add_argument('--output', required=True, default='', help='Output TREC qrels file.')

    args = parser.parse_args()

    with open(args.output, 'w') as fout:
        for line in open(args.input):
            fout.write(line.replace('\t', ' '))

    print('Done!')
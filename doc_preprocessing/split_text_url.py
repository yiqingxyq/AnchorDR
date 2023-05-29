import sys

def main():
    prefix = sys.argv[1]

    f_o_text = open(f'{prefix}_clean', 'w', encoding='utf-8')
    f_o_url = open(f'{prefix}_url', 'w', encoding='utf-8')

    for line in sys.stdin:
        if line[:4] == 'Url:':
            f_o_url.write(line)
        else:
            f_o_text.write(line)

    f_o_text.close()
    f_o_url.close()

if __name__ == '__main__':
    main()
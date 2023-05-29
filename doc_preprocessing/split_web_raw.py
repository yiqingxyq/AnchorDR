from distutils.command.build_scripts import first_line_re
import sys


def main():
    output_prefix = sys.argv[1]
    val_freq = int(sys.argv[2])
    f_ov = open(f'{output_prefix}.valid.txt', 'w', encoding='utf-8')
    f_ot = open(f'{output_prefix}.train.txt', 'w', encoding='utf-8')
    cur_f = None
    doc_cnt = 0
    cur_url = ''
    for line in sys.stdin:
        if line[:4] == 'Url:': 
            if cur_url == '':
                doc_cnt += 1
            cur_url = line
        else:
            line = line.strip()
            # if not line:
            #     doc_cnt += 1
                # # do not print \n unless we have a Url
                # if cur_f:
                #     cur_f.write('\n')
            if line:
                if doc_cnt % val_freq == val_freq - 1:
                    if cur_f != f_ov:
                        f_ot.write('\n')
                    if cur_url != '':
                        f_ov.write('\n' + cur_url)
                        cur_url = ''
                    f_ov.write(line + '\n')
                    cur_f = f_ov
                else:
                    if cur_f != f_ot:
                        f_ov.write('\n')
                    if cur_url != '':
                        f_ot.write('\n' + cur_url)
                        cur_url = ''
                    f_ot.write(line + '\n')
                    cur_f = f_ot
        # if doc_cnt > 100000:
        #     break
    f_ov.close()
    f_ot.close()


if __name__ == '__main__':
    main()

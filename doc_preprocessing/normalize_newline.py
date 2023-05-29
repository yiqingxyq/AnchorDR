import sys

cont_nl_cnt = 0
max_nl_cnt = 2
for line in sys.stdin:
    if line[:4] == 'Url:':
        sys.stdout.write(line)
    else:
        try:
            line = line.strip()
            if line:
                sys.stdout.write(line + '\n')
                cont_nl_cnt = 0
            else:
                if cont_nl_cnt < max_nl_cnt:
                    sys.stdout.write('\n')
                cont_nl_cnt += 1
        except Exception as e:
            pass
            #print(line, e, file=sys.stderr)

# newline between dataset
for i in range(max_nl_cnt):
    sys.stdout.write('\n')
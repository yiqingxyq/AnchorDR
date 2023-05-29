import sys
from numba import jit

def alpha_cnt(line):
    return sum(c.isalpha() for c in line)

min_len = 3
min_alpha_char = 2
min_char = 20
max_len = 200

def get(line):
    tokens = line.split()
    cnt = alpha_cnt(line)
    size = len(''.join(tokens))
    return tokens, cnt, size

def write(buf):
    if len(buf) == 0:
        pass
    elif len(buf) == 1:
        sys.stdout.write(' '.join(buf[0][0]) + '\n')
    else:
        assert len(buf) == 2
        if len(buf[1][0]) <= min_len and buf[1][1] > min_alpha_char and buf[1][2] <= min_char and len(buf[1][0]) + len(buf[0][0]) < max_len:
            sys.stdout.write(' '.join(buf[0][0] + buf[1][0]) + '\n')
        else:
            sys.stdout.write(' '.join(buf[0][0]) + '\n')
            sys.stdout.write(' '.join(buf[1][0]) + '\n')
    sys.stdout.write('\n')

def process(buf):
    assert len(buf) == 3
    if len(buf[1][0]) <= min_len and buf[1][1] > min_alpha_char and buf[1][2] <= min_char:
        len1 = len(buf[0][0]) + len(buf[1][0])
        len2 = len(buf[1][0]) + len(buf[2][0])
        if len1 < len2 and len1 < max_len:
            #print(len1, len2, buf[1][0], file=sys.stderr)
            buf[1] = buf[0][0] + buf[1][0], buf[0][1] + buf[1][1], buf[0][2] + buf[1][2]
            #sys.stdout.write(' '.join(buf[0][0] + buf[1][0]) + '\n')
            return buf[-2:]
        elif len2 < max_len:
            #print(len1, len2, buf[1][0], file=sys.stderr)
            # sys.stdout.write(' '.join(buf[0][0]) + '\n')
            # sys.stdout.write(' '.join(buf[1][0] + buf[2][0]) + '\n')
            buf[1] = buf[1][0] + buf[2][0], buf[1][1] + buf[2][1], buf[1][2] + buf[2][2]
            return buf[:2]
    sys.stdout.write(' '.join(buf[0][0]) + '\n')
    return buf[-2:]


def main():
    buf = []
    for line in sys.stdin:
        if line[:4] == 'Url:':
            sys.stdout.write(line)
        else:
            line = line.strip()
            if not line:
                write(buf)
                buf = []
            else:
                buf.append(get(line))
                if len(buf) == 3:
                    buf = process(buf)
                    assert len(buf) <= 2
    write(buf)

if __name__ == '__main__':
    main()
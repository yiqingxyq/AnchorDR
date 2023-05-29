import re
import sys
import io

for line in sys.stdin:
    if line[:4] == 'Url:':
        sys.stdout.write(line)
    else:
        # citation
        line = re.sub(r'^>+', ' ', line)
        line = re.sub(r'^-+', ' ', line)
        line = re.sub(r'^\++', ' ', line)
        line = ' '.join(line.strip().split())
        sys.stdout.write(line + '\n')
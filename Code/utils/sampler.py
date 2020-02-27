#!/usr/bin/python
import sys
import os

if len(sys.argv) == 4:
    dir_path = sys.argv[3]
    files = os.listdir(dir_path)
else:
    sys.exit("Usage:  python sampler.py <lines> <?dir>")

start = int(sys.argv[1])
end = int(sys.argv[2])
sample = []

for f in files:
    file_path = os.path.join(dir_path, f)
    if os.path.isfile(file_path):
        with open(file_path, 'rb') as fp:
            for i, line in enumerate(fp):
                if start <= i < end:
                    sys.stdout.write(line)
                elif i > end:
                    break

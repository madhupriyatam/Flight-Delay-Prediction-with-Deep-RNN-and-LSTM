__author__ = 'jordon'

import sys

def main(argv):
    # get files
    input = open(sys.argv[1], 'r')
    output = open(sys.argv[1][:-4] + '_subset.csv', 'w')

    line_num = 0
    for line in input:
        if line_num % 1000 == 0:
            output.write(line)
        line_num += 1


if __name__ == "__main__":
    main(sys.argv)

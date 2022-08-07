import argparse as a
import nltk
import re


def parse_args():
    parser = a.ArgumentParser()
    parser.add_argument("file", type=str)
    parser.add_argument("--out", "-o", type=str)
    return parser.parse_args()


def main():
    args = parse_args()

    f = open(args.file, 'r')
    lines = f.readlines()
    f.close()

    cleaned_lines = []
    for line in lines:
        line = line.replace(r"\\", ' ')
        sep = line.find('-')
        if sep != -1:
            content = line[sep:]
        else:
            content = line
        words = content.split(' ')
        cleaned_line = []
        for word in words:
            match = re.match(r"""^[\'\"]*([a-zA-Z]+)[\,\.\?\!\;\']*$""", word)
            if not match:
                continue
            word = match.group(1).lower()
            if word in nltk.corpus.stopwords.words('english'):
                continue
            cleaned_line.append(word)

        cleaned_lines.append(' '.join(cleaned_line))

    f = open(args.out, 'w')
    f.write('\n'.join(cleaned_lines))
    f.close()


if __name__ == "__main__":
    main()

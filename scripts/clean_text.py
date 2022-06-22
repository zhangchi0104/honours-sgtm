import argparse as a
import re
def parse_args():
    parser = a.ArgumentParser()
    parser.add_argument("--file", "-f", type=str)
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
        sep = line.find('-') + 1
        content = line[sep:-2]
        cleaned_line = [word.lower() for word in content.split(' ') if re.match('^[a-zA-Z\,\.\?\!]+$', word)]
        cleaned_lines.append(' '.join(cleaned_line))

    f = open(args.out, 'w')
    f.write('\n'.join(cleaned_lines))
    f.close()

if __name__ == "__main__":
    main()

from argparse import ArgumentParser
from pathlib import Path

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-input', help='Input image dir', type=str, required=True)
    args = parser.parse_args()
    input_dir = Path(args.input)
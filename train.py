from penn_treebank_reader import *
from dataset import DatasetReader
from dataset import make_batch_iterator


def main(options):
    tr_reader = JSONLReader(options.tr_file)
    tr_dataset = DatasetReader(tr_reader).build()
    batch_iterator = make_batch_iterator(None, tr_dataset)

    for batch_map in batch_iterator.get_iterator():
        print(batch_map)
        break


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--tr_file', default=os.path.expanduser('~/data/ptb/valid.jsonl'), type=str)
    options = parser.parse_args()

    main(options)

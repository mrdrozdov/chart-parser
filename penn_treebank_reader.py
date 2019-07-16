import json
import os

import nltk
from nltk.corpus import ptb

# For tests.
import unittest


####################
# Reader (raw PTB) #
####################

class RawPTBReader(object):
    def __init__(self):
        section2fileid = {}

        for fileid in ptb.fileids():
            if not fileid.startswith('WSJ'):
                continue
            section = int(fileid.split('/')[1])
            section2fileid.setdefault(section, []).append(fileid)

        self.tr_sections = [x for x in range(0, 22)]
        self.va_sections = [x for x in range(22, 23)]
        self.te_sections = [x for x in range(23, 24)]
        self.section2fileid = section2fileid

    def read_sections(self, sections):
        for section in sections:
            for fileid in self.section2fileid[section]:
                for s in ptb.parsed_sents(fileid):
                    yield s

    def read_tr(self):
        return self.read_sections(self.tr_sections)

    def read_va(self):
        return self.read_sections(self.va_sections)

    def read_te(self):
        return self.read_sections(self.te_sections)


################################
# Converter (raw PTB -> jsonl) #
################################

def tree_to_string(tree):
    def helper(tree):
        if isinstance(tree, str):
            return tree
        out = '({}'.format(tree.label())
        for x in tree:
            out += ' {}'.format(helper(x))
        return out + ')'
    return helper(tree)


def tree_to_spans(tree):
    spans = []
    def helper(tree, pos=0):
        if isinstance(tree, str):
            return 1
        size = 0
        for x in tree:
            spansize = helper(x, pos+size)
            size += spansize
        label = tree.label().split('-')[0] # TODO: This is wrong!
        spans.append((pos, size, label))
        return size
    helper(tree)
    return spans


class RawToJSONLConverter(object):
    def __init__(self, saveto):
        super(RawToJSONLConverter, self).__init__()
        self.reader = RawPTBReader()
        self.saveto = saveto

        if not os.path.exists(self.saveto):
            raise Exception('The `saveto` directory does not exist. ' + \
                'Run: `mkdir -p {}`'.format(self.saveto))

    def to_object(self, tree, example_id):
        o = {}
        o['example_id'] = example_id
        o['sentence'] = tree.leaves()
        o['parse'] = tree_to_string(tree)
        o['spans'] = tree_to_spans(tree)
        return o

    def run(self):
        count = 0

        savepath = os.path.join(self.saveto, 'train.jsonl')
        data = self.reader.read_tr()
        with open(savepath, 'w') as f:
            for tree in data:
                f.write('{}\n'.format(json.dumps(self.to_object(tree, example_id='ptb{}'.format(count)))))
                count += 1

        savepath = os.path.join(self.saveto, 'valid.jsonl')
        data = self.reader.read_va()
        with open(savepath, 'w') as f:
            for tree in data:
                f.write('{}\n'.format(json.dumps(self.to_object(tree, example_id='ptb{}'.format(count)))))
                count += 1

        savepath = os.path.join(self.saveto, 'test.jsonl')
        data = self.reader.read_te()
        with open(savepath, 'w') as f:
            for tree in data:
                f.write('{}\n'.format(json.dumps(self.to_object(tree, example_id='ptb{}'.format(count)))))
                count += 1


##################
# Reader (jsonl) #
##################

class JSONLReader(object):
    def __init__(self, path):
        super(JSONLReader, self).__init__()
        self.path = path

    def read(self):
        with open(self.path) as f:
            for line in f:
                yield json.loads(line)
        

#########
# Tests #
#########

class TestPTBReader(object):

    def __init__(self):
        self.reader = RawPTBReader()

    def run(self):
        self.test_dataset_count()

    def test_num_examples(self):
        tr = [s for s in self.reader.read_tr()]
        assert len(tr) == 43746
        va = [s for s in self.reader.read_va()]
        assert len(va) == 1700
        te = [s for s in self.reader.read_te()]
        assert len(te) == 2416

        assert len(tr) + len(va) + len(te) == 47862


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--saveto', default=os.path.expanduser('~/data/ptb'), type=str)
    parser.add_argument('--mode', default='test', choices=('test', 'convert', 'demo'))
    options = parser.parse_args()
    
    if options.mode == 'test':
        TestPTBReader().run()
    if options.mode == 'convert':
        RawToJSONLConverter(options.saveto).run()
    if options.mode == 'demo':
        print(next(JSONLReader(os.path.join(options.saveto, 'train.jsonl')).read()))

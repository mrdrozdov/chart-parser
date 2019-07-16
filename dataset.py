from batch_iterator import BatchIterator


class DatasetReader(object):
    def __init__(self, reader, lowercase=True, key_sentence='sentence', key_extra={'labels': 'spans'}, config={}):
        super(DatasetReader, self).__init__()
        self.reader = reader
        self.key_sentence = key_sentence
        self.key_extra = key_extra
        self.config = config

    def build(self):
        sentences = []
        extra = {}
        metadata = {}
        word2idx = {}
        label2idx = {}

        lowercase = self.config.get('lowercase', True)
        max_len = self.config.get('max_len', 0)

        def preprocess(s):
            if lowercase:
                return [w.lower() for w in s]
            return s

        for example in self.reader.read():
            s = example[self.key_sentence]
            if max_len > 0 and len(s) > max_len:
                continue
            sentences.append(preprocess(s))
            for k_to, k_from in self.key_extra.items():
                extra.setdefault(k_to, []).append(example[k_from])

        for s in sentences:
            for w in s:
                if w not in word2idx:
                    word2idx[w] = len(word2idx)

        for spans in extra['labels']:
            for pos, size, label in spans:
                if label not in label2idx:
                    label2idx[label] = len(label2idx)

        def indexify_sentences(s):
            return [word2idx[w] for w in s]

        def indexify_labels(lst):
            return [(pos, size, label2idx[label]) for pos, size, label in lst]

        sentences = [indexify_sentences(s) for s in sentences]
        extra['labels'] = [indexify_labels(lst) for lst in extra['labels']]

        metadata['word2idx'] = word2idx
        metadata['label2idx'] = label2idx

        dataset = {
            'sentences': sentences,
            'extra': extra,
            'metadata': metadata
        }

        return dataset


def make_batch_iterator(options, dataset):
    sentences = dataset['sentences']
    extra = dataset['extra']
    word2idx = dataset['metadata']['word2idx']

    vocab_size = len(word2idx)

    batch_iterator = BatchIterator(sentences, extra=extra)

    return batch_iterator


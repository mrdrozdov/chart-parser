from batch_iterator import BatchIterator


class DatasetReader(object):
    def __init__(self, reader, lowercase=True, key_sentence='sentence', key_extra={}):
        super(DatasetReader, self).__init__()
        self.reader = reader
        self.key_sentence = key_sentence
        self.key_extra = key_extra
        self.lowercase = lowercase

    def build(self):
        sentences = []
        extra = {}
        metadata = {}
        word2idx = {}

        def preprocess(s):
            if self.lowercase:
                return [w.lower() for w in s]
            return s

        for example in self.reader.read():
            sentences.append(preprocess(example[self.key_sentence]))
            for k_to, k_from in self.key_extra.items():
                extra.setdefault(k_to, []).append(example[k_from])

        for s in sentences:
            for w in s:
                if w not in word2idx:
                    word2idx[w] = len(word2idx)

        def indexify(s):
            return [word2idx[w] for w in s]

        sentences = [indexify(s) for s in sentences]

        metadata['word2idx'] = word2idx

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


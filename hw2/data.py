import os
import torch
from torch.utils import data
from .constants import MAX_SENT_LENGTH, MAX_SENT_LENGTH_PLUS_SOS_EOS, PAD_INDEX, SOS_INDEX, EOS_INDEX


def read_sentence_file(filename):
    """
    Reads all sentences in a file, parsing them into a list by splitting on spaces.

    :param filename: Absolute or relative path to the file
    :return: List containing the parsed tokens
    """
    sentences_list = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            sentences_list.append(line.strip().split())
    return sentences_list


def read_vocab_file(filename):
    """
    Read a vocabulary file into a list of tokens

    :param filename: Absolute or relative path to the file
    :return: Set containing the vocabulary tokens
    """
    with open(filename, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f)


def load_vocabulary(path):
    """
    Load both english (vocab.en) and vietnamese (vocab.vi) vocabulary files

    :param path: Path to the folder containing both files.
    :return: Tuple with both vocabularies sets.
    """
    src_vocab_set = read_vocab_file(os.path.join(path, "vocab.vi"))
    trg_vocab_set = read_vocab_file(os.path.join(path, "vocab.en"))
    return src_vocab_set, trg_vocab_set


def filter_data(src_sentences_list, trg_sentences_list, max_len):
    """
    Filter the elegible sentences (Smaller than maximum sentence size)

    :param src_sentences_list: List containing all vietnamese sentences.
    :param trg_sentences_list: List containing all english sentences.
    :param max_len: Maximum length to keep.
    :return: Tuple containings lists of the filtered sentences.
    """
    new_src_sentences_list, new_trg_sentences_list = [], []
    for src_sent, trg_sent in zip(src_sentences_list, trg_sentences_list):
        if max_len >= len(src_sent) > 0 and max_len >= len(trg_sent) > 0:
            new_src_sentences_list.append(src_sent)
            new_trg_sentences_list.append(trg_sent)
    return new_src_sentences_list, new_trg_sentences_list


def load_data(path):
    """
    Load all sentences and split them into train (90%) and test (10%) sets
    :param path: Path that contains all sentences files.
    :return: Tuple containing train source sentences, train target sentences, test source sentences and
        test target sentences.
    """
    train_src_sentences_list = read_sentence_file(os.path.join(path, "train.vi"))
    train_trg_sentences_list = read_sentence_file(os.path.join(path, "train.en"))
    assert len(train_src_sentences_list) == len(train_trg_sentences_list)

    test_src_sentences_list = read_sentence_file(os.path.join(path, "tst2013.vi"))
    test_trg_sentences_list = read_sentence_file(os.path.join(path, "tst2013.en"))
    assert len(test_src_sentences_list) == len(test_trg_sentences_list)

    train_src_sentences_list, train_trg_sentences_list = filter_data(
        train_src_sentences_list,
        train_trg_sentences_list,
        max_len=MAX_SENT_LENGTH
    )
    test_src_sentences_list, test_trg_sentences_list = filter_data(
        test_src_sentences_list,
        test_trg_sentences_list,
        max_len=MAX_SENT_LENGTH
    )

    num_val = int(len(train_src_sentences_list) * 0.1)
    val_src_sentences_list = train_src_sentences_list[:num_val]
    val_trg_sentences_list = train_trg_sentences_list[:num_val]
    train_src_sentences_list = train_src_sentences_list[num_val:]
    train_trg_sentences_list = train_trg_sentences_list[num_val:]

    return train_src_sentences_list, train_trg_sentences_list, val_src_sentences_list, val_trg_sentences_list


class MTDataset(data.Dataset):
    """
    Helper class that load and pre-process pairs of source-target sentences. This class not only reads the
    sentences from the lists but also take care of converting the strings into their corresponding indexed
    values. Indices for words are assigned based on their position on the vocabulary set. This class also
    make sure that all sentences are padded to the same length.
    """
    def __init__(self, src_sentences, src_vocabs, trg_sentences, trg_vocabs, sampling=1.):
        """
        Creates the dataset object

        :param src_sentences: List containing the source sentences.
        :param src_vocabs: Set containing the source vocabulary
        :param trg_sentences: List containing the target sentences
        :param trg_vocabs: Set containing the target vocabulary
        :param sampling: Percentage of the data to load.
        """
        self.src_sentences = src_sentences[:int(len(src_sentences) * sampling)]
        self.trg_sentences = trg_sentences[:int(len(src_sentences) * sampling)]

        self.max_src_seq_length = MAX_SENT_LENGTH_PLUS_SOS_EOS
        self.max_trg_seq_length = MAX_SENT_LENGTH_PLUS_SOS_EOS

        self.src_vocabs = src_vocabs
        self.trg_vocabs = trg_vocabs

        self.src_v2id = {v: i for i, v in enumerate(src_vocabs)}
        self.src_id2v = {val: key for key, val in self.src_v2id.items()}
        self.trg_v2id = {v: i for i, v in enumerate(trg_vocabs)}
        self.trg_id2v = {val: key for key, val in self.trg_v2id.items()}

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, index):
        src_sent = self.src_sentences[index]
        src_len = len(src_sent) + 2   # add <s> and </s> to each sentence
        src_id = []
        for w in src_sent:
            if w not in self.src_vocabs:
                w = '<unk>'
            src_id.append(self.src_v2id[w])
        src_id = ([SOS_INDEX] + src_id + [EOS_INDEX] + [PAD_INDEX] * (self.max_src_seq_length - src_len))

        trg_sent = self.trg_sentences[index]
        trg_len = len(trg_sent) + 2
        trg_id = []
        for w in trg_sent:
            if w not in self.trg_vocabs:
                w = '<unk>'
            trg_id.append(self.trg_v2id[w])
        trg_id = ([SOS_INDEX] + trg_id + [EOS_INDEX] + [PAD_INDEX] *  (self.max_trg_seq_length - trg_len))

        return torch.tensor(src_id), src_len, torch.tensor(trg_id), trg_len


def load_dataset(path, sampling=1.):
    """
    Load all data into the Dataset class

    :param path: The path that contains the sentence files
    :param sampling: Percentage of data to load.
    :return: Tuple with train dataset and test dataset.
    """
    train_src, train_trg, test_src, test_trg = load_data(path)
    vocab_src, vocab_trg = load_vocabulary(path)

    train_ds = MTDataset(train_src, vocab_src, train_trg, vocab_trg, sampling=sampling)
    test_ds = MTDataset(test_src, vocab_src, test_trg, vocab_trg, sampling=sampling)

    return train_ds, test_ds
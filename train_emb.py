import argparse
import os
import json
import glob
import time
from collections import Counter
from multiprocessing import Pool
from os.path import join as pjoin

from tqdm import tqdm

from data_processor.others.vocab_wrapper import VocabWrapper
import spacy

from data_processor.preprocess_text import WhiteSpacePreprocessingStopwords

nlp = spacy.load('en_core_sci_lg')
import nltk
from nltk.corpus import stopwords
sws = stopwords.words('english')

import torch

from transformers import LEDTokenizer
tokenizer = LEDTokenizer.from_pretrained("allenai/led-large-16384-arxiv")

class Vocab(object):
    def __init__(self, vocab_size=3000):
        self.vocab_size = vocab_size
        self.vocab_mapping = {}
        self.all_counter = Counter()

    def _build_vocab(self, contextualized_vocab_ids):
        counter = 0
        all_vocabs = {}
        for token_id in contextualized_vocab_ids:
            if token_id not in all_vocabs.keys():
                all_vocabs[token_id] = counter
                counter += 1

        self.vocab_mapping = all_vocabs
        assert len(self.vocab_mapping) == self.vocab_size, f"Vocab mapping should have {self.vocab_size}, but it has {len(self.vocab_mapping)}!!"

    def _v2i(self, contextualized_id):
        try:
            return self.vocab_mapping[contextualized_id]
        except:
            return None

    def _update_vocab_counter(self, counter):
        counter = [c for c in counter if c in self.vocab_mapping.keys()]
        self.all_counter.update(counter)


    def save_vocab(self, path):
        torch.save(self.vocab_mapping, path)


def get_sentence_tokens(text):
    doc_nlp = nlp(text)
    ret = []
    for doc_sentence in doc_nlp.sents:
        sent = []
        for tkn in doc_sentence:
            sent.append(tkn.text)
            ret.append(tkn.text)
        # ret.append(sent)
    return ret

def get_sentence_tokens_contextualized_tokenizer(text):
    tokens = tokenizer.tokenize(text.lower())
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    return token_ids, tokens



def train_emb(args):
    data_dir = os.path.abspath(args.data_path)
    print("Preparing to process %s ..." % data_dir)

    ex_num = 0
    vocab_wrapper = Vocab(vocab_size=3000)

    file_ex = []
    instances = []

    for corpus_type in ['train', 'val', 'test']:
        for json_f in glob.glob(f'{data_dir}/' + f'/{corpus_type}/' + '*.json'):
            instances.append(json.load(open(json_f)))


    documents = []
    for ins in instances:
        pr_instances = []
        for section in ins['sections_txt_tokenized']:
            pr_instances.append(section['text'])
        documents.append(' '.join(pr_instances))
    tic = time.perf_counter()
    print()
    print(f'\t Processing All documents: {len(documents)} \t ')


    sp = WhiteSpacePreprocessingStopwords(documents=documents, vocabulary_size=3000, min_words=2, stopwords_list=sws)
    vocabulary_ids = sp.preprocess(tokenizer=tokenizer)

    print(f'\t Length of distinct vocab_ids: {len(set(vocabulary_ids))}')

    toc = time.perf_counter()
    print(f'\t {toc - tic:0.4f} seconds -- Processing Done!')


    tic = time.perf_counter()
    print(f'\t Tokenizing documents...')
    vocab_wrapper._build_vocab(vocabulary_ids)

    pool = Pool(16)
    for tokens in tqdm(pool.imap_unordered(get_sentence_tokens_contextualized_tokenizer, documents), total=len(documents)):
        vocab_wrapper._update_vocab_counter(list(tokens[0]))

    toc = time.perf_counter()
    print(f'\t {toc - tic:0.4f} seconds -- Tokenization Done!')

    tic = time.perf_counter()
    print(f'\t Building vocabularies...')
    print(f'\t Vocab len: {len(vocabulary_ids)}')

    toc = time.perf_counter()
    print(f'\t {toc - tic:0.4f} seconds -- Vocab builiding Done!')

    vocab_wrapper.save_vocab(args.vocab_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_path", default="/disk1/sajad/datasets/sci/mup/single_tokenized/", type=str)
    parser.add_argument("-vocab_path", default="/disk1/sajad/w2v_embeds/mup_vocab.pt", type=str)

    args = parser.parse_args()

    train_emb(args)

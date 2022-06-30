import argparse
import os
import json
import glob
from multiprocessing import Pool
from os.path import join as pjoin

from tqdm import tqdm

from data_processor.others.vocab_wrapper import VocabWrapper
import spacy

from data_processor.preprocess_text import WhiteSpacePreprocessingStopwords

nlp = spacy.load('en_core_sci_lg')

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

def _parse_paper(param):
    ex = param
    example_sents = get_sentence_tokens(ex['paper']['abstractText'])
    ex['paper']['abstract_tokens'] = example_sents.copy()

    new_sects = []
    for e in ex['paper']['sections']:
        # example_sents.extend()
        e['tokens'] = get_sentence_tokens(e['text'])
        new_sects.append(e)

    # ex['sentences'] = example_sents
    ex['paper']['sections'] = new_sects

    return ex


def train_emb(args):
    data_dir = os.path.abspath(args.data_path)
    print("Preparing to process %s ..." % data_dir)
    raw_files = glob.glob(pjoin(data_dir, '*_complete.jsonl'))
    # raw_files = [g for g in glob.glob(pjoin(data_dir, '*.jsonl')) if 'train' not in g]

    ex_num = 0
    vocab_wrapper = VocabWrapper(args.mode, args.emb_size)
    vocab_wrapper.init_model()

    file_ex = []
    instances = []

    for corpus_type in ['train', 'val']:
        for json_f in glob.glob('/disk1/sajad/datasets/sci/mup/single_files/' + f'/{corpus_type}/' + '*.json'):
            instances.append(json.load(open(json_f)))


    print(f'All instances: {len(instances)}')
    documents = []
    for ins in instances:
        pr_instances = []
        for sent_tokens in ins['source_sents']:
            pr_instances.append(' '.join(sent_tokens))
        documents.append(' '.join(pr_instances))

        for summary in ins['summary']:
            # summ_sent_tokens = get_sentence_tokens(summary)
            # for summ_sent_token in summ_sent_tokens:
                pr_instances.append(summary)

        documents.append(' '.join(pr_instances))

    sp = WhiteSpacePreprocessingStopwords(documents=documents, vocabulary_size=3000)
    preprocessed_docs, unpreprocessed_docs, vocabulary, retained_indices = sp.preprocess()

    pool = Pool(16)
    for tokens in tqdm(pool.imap_unordered(get_sentence_tokens, preprocessed_docs), total=len(preprocessed_docs)):
        file_ex.append(tokens)

    # for doc in tqdm(preprocessed_docs, total=len(preprocessed_docs)):
    #     tokens = get_sentence_tokens(doc)
    #     file_ex.append(tokens)

    print("Training embeddings...")
    vocab_wrapper.train(file_ex)
    vocab_wrapper.report()
    print("Datasets size: %d" % ex_num)
    vocab_wrapper.save_emb(args.emb_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-mode", default='word2vec', type=str, choices=['glove', 'word2vec'])
    parser.add_argument("-data_path", default="data_processor/mup/", type=str)
    parser.add_argument("-emb_size", default=100, type=int)
    parser.add_argument("-emb_path", default="/disk1/sajad/w2v_embeds/w2v_mup_reduced.emb", type=str)

    args = parser.parse_args()

    train_emb(args)

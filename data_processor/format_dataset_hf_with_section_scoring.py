import glob
import json
import math
import os
from multiprocessing import Pool

import numpy as np
import spacy
from tqdm import tqdm
import torch

nlp = spacy.load('en_core_sci_lg')
from rouge_score import rouge_scorer
metrics = ['rouge1', 'rouge2', 'rougeL']
scorer = rouge_scorer.RougeScorer(metrics, use_stemmer=True)

def get_sentence_tokens(text):
    doc_nlp = nlp(text)
    ret = []
    for doc_sentence in doc_nlp.sents:
        sent = []
        for tkn in doc_sentence:
            sent.append(tkn.text)
        ret.append(sent)
    return ret

def _parse_paper(param):
    ex = param
    example_sents = get_sentence_tokens(ex['source'])
    ex['source_sents'] = example_sents.copy()

    # new_sects = []
    # for e in ex['paper']['sections']:
    #     example_sents.extend()
        # e['tokens'] = get_sentence_tokens(e['text'])
        # new_sects.append(e)

    # ex['sentences'] = example_sents
    # ex['paper']['sections'] = new_sects

    return ex

def _cal_rg_sect(params):
    p_id, sect_sents, sect_sents_tokenized, summaries, ext_labels = params
    # adding section scores
    section_scores = []
    for sect_idx, sect in enumerate(sect_sents):
        sums_sect_score = []
        for summ_idx, summ in enumerate(summaries):
            sent_sect_scores = []
            for sect_sent in sect:
                sents_scores = scorer.score(summ.strip(), sect_sent.strip())
                sents_scores = [sents_scores['rouge1'].recall, sents_scores['rouge2'].recall, sents_scores['rougeL'].recall]
                avg_sent_score = np.average(sents_scores)
                sent_sect_scores.append(avg_sent_score)

            sect_sum_labels = ext_labels[sect_idx][summ_idx]
            sect_sents_ = sect_sents[sect_idx]
            sect_tokens = sect_sents_tokenized[sect_idx]
            # import pdb;pdb.set_trace()
            rg_avrg = (np.average(sent_sect_scores)+0.01) * ((sum(sect_sum_labels)+1) / (math.sqrt(len(sect_tokens.split())) * len(sect_sents_)))
            # import pdb;pdb.set_trace()
            """
            sectScoreV1.parquet: 
                -   rg_avrg = np.average(sent_sect_scores) * ((sum(sect_sum_labels)) / math.sqrt(len(sect_tokens.split())/len(sect_sents_))) 
            """
            # weighting by sect_len and number of important sentences


            sums_sect_score.append(rg_avrg)
        section_scores.append(sums_sect_score.copy())

    section_scores = np.array(section_scores)

    # section_scores / np.moveaxis(section_scores,0, -1).sum(axis=1)[None, :]
    return (p_id, section_scores)

def load_topic_info(se):
    ret = {}
    for file in glob.glob(f"/disk1/sajad/datasets/sci/mup/bert_data_scores2/{se}.*.pt"):
        file_dict = torch.load(file)
        ret.update(file_dict)
    return ret

if __name__ == '__main__':

    for se in ['train', 'val']:
        json_ents_dict = {}
        with open(f"/disk1/sajad/datasets/sci/mup/{se}_complete.json") as fR:
            for l in fR:
                ent = json.loads(l)
                if ent['paper_id'] not in json_ents_dict.keys():
                    json_ents_dict[ent['paper_id']] = {}
                    source = ent['paper']['abstractText']
                    for sect_text in ent['paper']['sections']:
                        source += ' '
                        source += sect_text['text']
                    source = source.strip()

                    json_ents_dict[ent['paper_id']]['source'] = source
                    json_ents_dict[ent['paper_id']]['summary'] = [ent['summary']]

                else:
                    json_ents_dict[ent['paper_id']]['summary'].append(ent['summary'])


        topic_info_dict = load_topic_info(se)
        hf_format = []
        hf_df = {
            'paper_id': [],
            'source': [],
            'section_headings': [],
            'source_tokenized': [],
            'ext_labels': [],
            'section_scores': [],
            'summary': [],
            'topic_info_global': [],
            'topic_info_section': [],
        }

        count = 0
        for paper_id, paper_ent in json_ents_dict.items():

            hf_df['paper_id'].append(paper_id)
            if len(topic_info_dict[paper_id]['section_text']) != len((topic_info_dict[paper_id]['topic_info_section'])):
                import pdb;pdb.set_trace()

            hf_df['source'].append(topic_info_dict[paper_id]['sections_sents'])
            hf_df['section_headings'].append(topic_info_dict[paper_id]['section_headings'])
            hf_df['source_tokenized'].append(topic_info_dict[paper_id]['section_text'])
            hf_df['summary'].append(paper_ent['summary'])
            hf_df['ext_labels'].append(topic_info_dict[paper_id]['ext_labels'])
            hf_df['topic_info_section'].append(json.dumps(topic_info_dict[paper_id]['topic_info_section']))
            hf_df['topic_info_global'].append(json.dumps(topic_info_dict[paper_id]['topic_info_global']))
            count += 1
            # if count == 12:
            #     break

        print('Calculating section scores...')
        pool = Pool(16)

        section_scores_lst = [0 for _ in range(len(hf_df['paper_id']))]

        mp_instances = [(p_id, src, src_tokenized, summaries, ext_labels) for p_id, src, src_tokenized, summaries, ext_labels in zip(hf_df['paper_id'], hf_df['source'],
                                                                                                       hf_df['source_tokenized'], hf_df['summary'], hf_df['ext_labels'])]
        # for m in mp_instances:
        #     if len(m[-1]) > 1:
        #         _cal_rg_sect(m)

        paper_ids_indices = hf_df['paper_id'].copy()

        for ret in tqdm(pool.imap_unordered(_cal_rg_sect, mp_instances), total=len(mp_instances)):
            p_id = ret[0]
            section_scores = ret[1]
            # if section_scores.shape[1] > 1:
            #     import pdb;pdb.set_trace()
            paper_idx = paper_ids_indices.index(p_id)
            section_scores_lst[paper_idx] = section_scores.tolist()

        hf_df['section_scores'] = section_scores_lst
        hf_df.pop('source_tokenized')

        print('Writing HF files...')

        try:
            os.makedirs('/disk1/sajad/datasets/sci/mup/hf_format/')
        except:
            pass

        try:
            os.makedirs(f'/disk1/sajad/datasets/sci/mup/single_files/{se}')
        except:
            pass

        import pandas as pd
        df = pd.DataFrame(hf_df)
        df.to_parquet(f"/disk1/sajad/datasets/sci/mup/hf_format/{se}-sectScoreV1.parquet")

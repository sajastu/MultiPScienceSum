import glob
import json
import os
from multiprocessing import Pool

import spacy
from tqdm import tqdm
import torch

nlp = spacy.load('en_core_sci_lg')

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


def load_topic_info(se):
    ret = {}
    for file in glob.glob(f"/disk1/sajad/datasets/sci/mup/bert_data/{se}.*.pt"):
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
            'summary': [],
            'topic_info': [],
            'topic_summ_info': []
        }

        for paper_id, paper_ent in json_ents_dict.items():
            # import pdb;pdb.set_trace()
            hf_format.append(
                {
                    'paper_id': paper_id,
                    'source': paper_ent['source'],
                    'summary': paper_ent['summary'],
                    'topic_info': topic_info_dict[paper_id]['topic_info'],
                    'topic_summ_infos': topic_info_dict[paper_id]['topic_summ_info']
                }
            )

            hf_df['paper_id'].append(paper_id)
            hf_df['source'].append(paper_ent['source'])
            hf_df['summary'].append(paper_ent['summary'])
            # import pdb;pdb.set_trace()
            hf_df['topic_info'].append(json.dumps(topic_info_dict[paper_id]['topic_info']))
            hf_df['topic_summ_info'].append(json.dumps(topic_info_dict[paper_id]['topic_summ_info']))

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
        df.to_parquet(f"/disk1/sajad/datasets/sci/mup/hf_format/{se}.parquet")

        # with open(f'/disk1/sajad/datasets/sci/mup/hf_format/{se}.json', mode='w') as fW:
        #     for hf_frmt in hf_format:
        #         json.dump(hf_frmt, fW)
        #         fW.write('\n')
        #
        # print('Tokenizing sentences...')

        # for hf_frmt in hf_format:
        # pool = Pool(15)
        # tokenized_hfs = []
        # for ret in tqdm(pool.imap_unordered(_parse_paper, hf_format), total=len(hf_format)):
        #     tokenized_hfs.append(ret)



        # print('Writing to single files...')
        # for g in tokenized_hfs:
        #     with open(f'/disk1/sajad/datasets/sci/mup/single_files/{se}/{g["paper_id"]}.json', mode="w") as fW:
        #         json.dump(g, fW)

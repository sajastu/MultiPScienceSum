import glob
import json
import os
from multiprocessing import Pool

import spacy
from tqdm import tqdm

nlp = spacy.load('en_core_sci_lg')

def get_sentence_tokens(text):
    doc_nlp = nlp(text)
    ret = []
    for doc_sentence in doc_nlp.sents:
        sent = []
        for tkn in doc_sentence:
            sent.append(tkn.text)
            # ret.append(tkn.text)
        ret.append(sent)
    return ret

def _parse_paepr(params):
    paper_id, sections, summaries = params
    ret = []

    for section in sections:
        ret.append(
            get_sentence_tokens(section)
        )

    return paper_id, sections, summaries, ret

def main():
    print("Preparing to process %s ..." % '/disk1/sajad/datasets/sci/mup')
    raw_files = glob.glob('/disk1/sajad/datasets/sci/mup' + '/*_complete.json')
    # open raw_files
    for file in raw_files:
        debug_counter = 0

        final_list = {}
        with open(file) as fR:
            for li in fR:
                ent = json.loads(li)
                paper_id = ent['paper_id']
                paper_summary = ent['summary']

                if paper_id not in final_list:
                    paper_info = ent['paper']
                    abstract_text = paper_info['abstractText']
                    paper_sects_list = []
                    for paper_section in paper_info['sections']:

                        section_text = paper_section['text'].strip()

                        if len(section_text) > 0:
                            paper_sects_list.append(section_text)

                    final_list[paper_id] = {
                        'sections': [abstract_text] + paper_sects_list,
                        'summaries': [paper_summary]
                    }
                else:
                    final_list[paper_id]['summaries'].append(paper_summary)

                debug_counter+=1
                # if debug_counter ==11:
                #     break

        # now tokenizing paper source...
        mp_list = []
        for paper_id, paper_info in final_list.items():
            mp_list.append(
                (paper_id, paper_info['sections'], paper_info['summaries'])
            )

        pool = Pool(16)
        new_dict_saved = {}
        for ret in tqdm(pool.imap_unordered(_parse_paepr, mp_list), total=len(mp_list)):
            paper_id = ret[0]
            sects = ret[1]
            summaries = ret[2]
            tokenized_sects = ret[3]
            saved_dict = {
                'paper_id': paper_id,
                'sections_txt_tokenized': [{'text':sect_text, 'tokenized': sect_toknized} for sect_text, sect_toknized in zip(sects, tokenized_sects)],
                'summaries': summaries
            }
            new_dict_saved[paper_id] = saved_dict

        # now save the processed files....
        appr_set = "train" if "train" in file.split("/")[-1] else "val"
        WR_DIR = f'/disk1/sajad/datasets/sci/mup/single_tokenized/{appr_set}'

        try:
            os.makedirs(WR_DIR)
        except:
            pass

        print(f'Writing the single files to {WR_DIR}/...')
        for paper_id, paper_info in new_dict_saved.items():
            with open(f'{WR_DIR}/{paper_id}.json', mode='w') as fW:
                json.dump(paper_info, fW)



if __name__ == '__main__':
    main()
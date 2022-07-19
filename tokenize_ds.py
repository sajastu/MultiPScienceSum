import glob
import json
import os
import re
import uuid
from multiprocessing import Pool

import pandas as pd
import spacy
from tqdm import tqdm
from unidecode import unidecode

nlp = spacy.load('en_core_sci_lg')


BAD_SECTIONS = ['acknowledgment', 'acknowledgments' , 'acknowledgements',
         'fund' ,'funding' ,
     'appendices' , 'proof of' ,
        'related work' , 'previous works' , 'references',
         'figure captions' , 'acknowledgement' , 'appendix']

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
            get_sentence_tokens(section['text'])
        )

    return paper_id, sections, summaries, ret

def remove_related(text):
    # print(text)
    # paranth_regex = "[A-Za-z]?\s?\(\s??[A-Za-z]\s?(\|[A-Za-z])?\)"
    # matches = re.findall(paranth_regex, text)
    # # print(matches)
    # for m in matches:
    #     text = text.replace(f'({m})', '')
    #     text = text.replace(m, '')
    #
    # # math formula including ~, = and +
    # paranth_regex = "\w*\s?[~|=|+]\w*?"
    # matches = re.findall(paranth_regex, text)
    # # print(matches)
    # for m in matches:
    #     id = f'{str(uuid.uuid4())[:5]}'
    #     text = text.replace(f'({m})', f'@xmath{id}')
    #     text = text.replace(m, f'@xmath{id}')
    #
    #
    # paranth_regex = "\(\s[A-Z]\s\)?"
    # matches = re.findall(paranth_regex, text)
    # # print(matches)
    # for m in matches:
    #     id = f'{str(uuid.uuid4())[:5]}'
    #     text = text.replace(f'({m})', f'@xmath{id}')
    #     text = text.replace(m, f'@xmath{id}')
    #
    # paranth_regex = "\(?\s[A-Z]\s\)"
    # matches = re.findall(paranth_regex, text)
    # # print(matches)
    # for m in matches:
    #     id = f'{str(uuid.uuid4())[:5]}'
    #     text = text.replace(f'({m})', f'@xmath{id}')
    #     text = text.replace(m, f'@xmath{id}')
    #
    # # p(x)
    # paranth_regex = "[A-Za-z]\([A-Za-z]\)"
    # matches = re.findall(paranth_regex, text)
    # # print(matches)
    # for m in matches:
    #     id = f'{str(uuid.uuid4())[:5]}'
    #     text = text.replace(f'({m})', f'@xmath{id}')
    #     text = text.replace(m, f'@xmath{id}')
    #
    # # all brackets

    #
    # # p(x, z)
    # paranth_regex = "[A-Za-z]?\([A-Za-z]\,\s?[A-Za-z]\)"
    # matches = re.findall(paranth_regex, text)
    # # print(matches)
    # for m in matches:
    #     id = f'{str(uuid.uuid4())[:5]}'
    #     text = text.replace(f'({m})', f'@xmath{id}')
    #     text = text.replace(m, f'@xmath{id}')

    # paranth_regex = "\[.*\]"
    # matches = re.findall(paranth_regex, text)
    # print(matches)
    # for m in matches:
    #     text = text.replace(f'({m})', '')
    #     text = text.replace(m, '')

    ## related_work
    author = "(?:[A-Za-z][A-Za-z'`-]+)"
    etal = "(?:et al.?)"
    additional = "(?:,? (?:(?:and |& )?" + author + "|" + etal + "))"
    year_num = "(?:19|20)[0-9][0-9]"
    page_num = "(?:, p.? [0-9]+)?"  # Always optional
    year = "(?:, *" + year_num + page_num + "| *\(" + year_num + page_num + "\))"
    # year_index = "*[a|b|c]"
    year_index = "[a|b|c]?"
    regex = "(" + author + additional + "*" + year +  year_index + ";?" + ")"
    # regex += "(" + ";?" + regex + "?" + ")"

    matches = re.findall(regex, text)
    # print(matches)
    for m in matches:
        text = text.replace(f'({m})', f'@xcite_{str(uuid.uuid4())[:5]}')
        text = text.replace(m, f'@xcite_{str(uuid.uuid4())[:5]}')

    author = "(?:[A-Z][A-Za-z'`-]+)"
    etal = "(?:et al.?)"
    additional = "(?:,? (?:(?:and |& )?" + author + "|" + etal + "))"
    regex = "\(" + author + additional + "\)"
    # regex += "(" + ";?" + regex + "?" + ")"

    matches = re.findall(regex, text)
    # print(matches)
    for m in matches:
        text = text.replace(f'({m})', f'@xcite_{str(uuid.uuid4())[:5]}')
        text = text.replace(m, f'@xcite_{str(uuid.uuid4())[:5]}')
    # print(text)

    paranth_regex = "\(;?\s+\)"
    matches = re.findall(paranth_regex, text)
    # print(matches)
    for m in matches:
        text = text.replace(f'({m})', '')
        text = text.replace(m, '')

    return text

def ngrams(input, n):
    input = input.split(' ')
    output = []
    for i in range(len(input)-n+1):
        output.append(input[i:i+n])
    return output

conc_keywords = pd.read_csv('/home/sajad/packages/summarization/mup/data_processor/heading_keyword.csv')
conc_keywords = conc_keywords['conclusion'].dropna().tolist()

def is_couc(heading):
    while heading[0].isdigit() or heading[0] == '.':
        heading = heading[1:]
    heading = heading.strip()
    one_grams = ngrams(heading.lower(), 1)
    ongrams = []
    for oneg in one_grams:
        ongrams.append(oneg[0])

    for one_gram in ongrams:
        for bad_sect in conc_keywords:
            if one_gram.strip() == bad_sect.strip():
                return True

    if len(heading.split(' ')) > 1:
        two_grams = ngrams(heading.lower(), 2)
        twograms = []
        for tg in two_grams:
            twograms.append(f'{tg[0]} {tg[1]}')

        for two_gram in twograms:
            for bad_sect in conc_keywords:
                if two_gram.strip() == bad_sect.strip():
                    return True



def is_bad_section(heading):

    # appendix
    if heading.startswith(('A ', 'B ', 'C ', 'D ', 'E ', 'F ', 'G ')):
        return True

    while heading[0].isdigit() or heading[0] == '.':
        heading = heading[1:]
    heading = heading.strip()

    one_grams = ngrams(heading.lower(), 1)
    ongrams = []
    for oneg in one_grams:
        ongrams.append(oneg[0])

    for one_gram in ongrams:
        for bad_sect in BAD_SECTIONS:
            if one_gram.strip() == bad_sect.strip():
                return True

    if len(heading.split(' ')) > 1:
        two_grams = ngrams(heading.lower(), 2)
        twograms = []
        for tg in two_grams:
            twograms.append(f'{tg[0]} {tg[1]}')

        for two_gram in twograms:
            for bad_sect in BAD_SECTIONS:
                if two_gram.strip() == bad_sect.strip():
                    return True
    return False


def main():
    print("Preparing to process %s ..." % '/disk1/sajad/datasets/sci/mup')
    raw_files = glob.glob('/disk1/sajad/datasets/sci/mup' + '/*_complete.json')
    # open raw_files
    for file in raw_files:
        # if 'train' in file:
        #     continue
        debug_counter = 0

        final_list = {}
        with open(file) as fR:
            for li in fR:
                ent = json.loads(li)
                paper_id = ent['paper_id']
                paper_summary = ent['summary']

                # if paper_id != 'SP:23084f30a4183f6965ef97e4cba2082bf2fffd64':
                #     continue
                # else:
                #     continue

                if paper_id not in final_list:
                    paper_info = ent['paper']
                    abstract_text = paper_info['abstractText']
                    paper_sects_list = []

                    all_section_headings = []
                    section_mask = []

                    for j, e in enumerate(paper_info['sections']):

                        if 'heading' in e.keys():
                            all_section_headings.append(e['heading'])
                            section_mask.append(True)

                        elif j==0 and 'heading' not in e.keys():
                            # check if it is abstract
                            abs_chars = None
                            text_chars = None
                            try:
                                abs_chars = ''.join([a[0] for a in abstract_text.split(' ')[:20]])
                                text_chars = ''.join([a[0] for a in e['text'].split(' ')[:20]])
                            except:
                                pass

                            if (abs_chars is not None and text_chars is not None) and abs_chars.strip() == text_chars.strip():
                                all_section_headings.append('NA')
                                section_mask.append(False)
                                continue

                            all_section_headings.append('NA')
                            section_mask.append(True)
                        else:
                            all_section_headings.append('NA')
                            section_mask.append(True)



                    main_sects = {}
                    for jsect, sect_heading in enumerate(all_section_headings):
                        if sect_heading != 'NA':
                            if sect_heading.startswith(('1 ', '2 ', '3 ', '4 ', '5 ', '6 ', '7 ', '8 ',
                                                        '9 ', '10 ', '11 ', '12 ', '13 ', '14 ', '15 ')): #main section:
                                main_sects[re.findall(r'^\d+', sect_heading)[0]] = sect_heading.replace(re.findall(r'^\d+', sect_heading)[0].strip(), '').strip()




                    for jsect, paper_section in enumerate(paper_info['sections']):
                        if not section_mask[jsect]:
                            continue

                        section_text = paper_section['text'].strip()
                        section_heading = paper_section['heading'].strip() if all_section_headings[jsect] != 'NA' and 'heading' in paper_section.keys() else 'NA'
                        heading_text = 'NA'
                        if section_heading != 'NA':
                            sect_num = re.findall("^\d+", section_heading)


                            if len(sect_num) > 0:
                                is_subsect = False
                                referring_sect = sect_num[0].strip()

                            else:
                                sect_num = re.findall("^\d+\.\d+", section_heading)
                                if len(sect_num) > 0:
                                    is_subsect = True
                                    referring_sect = sect_num[0].split('.')[0]
                                else:
                                    is_subsect = False
                            # sect_num = re.findall(r'\d+', section_heading)[0]

                            try:
                                heading_text = referring_sect + ' ' + main_sects[referring_sect].strip() if \
                                    referring_sect in main_sects.keys() else 'NA' + ' > ' + paper_section['heading'] if is_subsect else paper_section['heading']
                            except:
                                import pdb;pdb.set_trace()

                        # if paper_id == 'SP:23084f30a4183f6965ef97e4cba2082bf2fffd64':
                        #     import pdb;
                        #     pdb.set_trace()
                        # else:
                        #     continue

                        # if the current section is conclusion, append and then just break!!
                        if len(section_text.strip()) > 0 and is_couc(section_heading):
                            paper_sects_list.append({'heading': heading_text,
                                                     'text': remove_related(
                                                         unidecode(section_text.replace('\n', ' ').strip()))})
                            break

                        if len(section_text.strip()) > 0 and not is_bad_section(section_heading):
                            paper_sects_list.append({'heading': heading_text,
                                                    'text': remove_related(unidecode(section_text.replace('\n', ' ').strip()))})

                    final_list[paper_id] = {
                        'sections': [{'heading': 'abstract', 'text': remove_related(unidecode(abstract_text))}] + paper_sects_list,
                        'summaries': [paper_summary]
                    }
                else:
                    final_list[paper_id]['summaries'].append(paper_summary)

                debug_counter+=1

        # now tokenizing paper source...
        mp_list = []
        # import pdb;pdb.set_trace()
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
                'sections_txt_tokenized': [{'heading':sect_text['heading'], 'text':sect_text['text'], 'tokenized': sect_toknized} for sect_text, sect_toknized in zip(sects, tokenized_sects)],
                'summaries': summaries
            }
            new_dict_saved[paper_id] = saved_dict

        # now save the processed files....
        appr_set = "train" if "train" in file.split("/")[-1] else "val"
        WR_DIR = f'/disk1/sajad/datasets/sci/mup/single_tokenized_prep2/{appr_set}'

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
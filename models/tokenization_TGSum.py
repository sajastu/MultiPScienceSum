
import itertools
import math
from typing import Optional, Union, List, Dict, Any, Sequence, Tuple

import numpy as np

from transformers import TensorType, LEDTokenizer, is_tf_available, is_torch_available, is_flax_available
import torch

from transformers.tokenization_utils_base import TruncationStrategy, BatchEncoding, EncodingFast, TextInput, \
    TextInputPair, PreTokenizedInput, PreTokenizedInputPair, EncodedInput, EncodedInputPair
from transformers.utils import logging
from transformers.utils.generic import _is_jax, _is_numpy, PaddingStrategy, _is_tensorflow, _is_torch, to_py_obj
import transformers

logger = logging.get_logger(__name__)
transformers.logging.set_verbosity_info()

class BatchEncoding(BatchEncoding):

    def __init__(
            self,
            data: Optional[Dict[str, Any]] = None,
            encoding: Optional[Union[EncodingFast, Sequence[EncodingFast]]] = None,
            tensor_type: Union[None, str, TensorType] = None,
            prepend_batch_axis: bool = False,
            n_sequences: Optional[int] = None,
            is_target = False,
    ):
        self.is_target = is_target
        super(BatchEncoding, self).__init__(
            data,
            encoding,
            tensor_type,
            prepend_batch_axis,
            n_sequences
        )

    def convert_to_tensors(
        self, tensor_type: Optional[Union[str, TensorType]] = None, prepend_batch_axis: bool = False
    ):
        """
        Convert the inner content to tensors.

        Args:
            tensor_type (`str` or [`~file_utils.TensorType`], *optional*):
                The type of tensors to use. If `str`, should be one of the values of the enum
                [`~file_utils.TensorType`]. If `None`, no modification is done.
            prepend_batch_axis (`int`, *optional*, defaults to `False`):
                Whether or not to add the batch dimension during the conversion.
        """
        if tensor_type is None:
            return self

        # Convert to TensorType
        if not isinstance(tensor_type, TensorType):
            tensor_type = TensorType(tensor_type)

        # Get a function reference for the correct framework
        if tensor_type == TensorType.TENSORFLOW:
            if not is_tf_available():
                raise ImportError(
                    "Unable to convert output to TensorFlow tensors format, TensorFlow is not installed."
                )
            import tensorflow as tf

            as_tensor = tf.constant
            is_tensor = tf.is_tensor
        elif tensor_type == TensorType.PYTORCH:
            if not is_torch_available():
                raise ImportError("Unable to convert output to PyTorch tensors format, PyTorch is not installed.")
            import torch

            as_tensor = torch.tensor
            is_tensor = torch.is_tensor
        elif tensor_type == TensorType.JAX:
            if not is_flax_available():
                raise ImportError("Unable to convert output to JAX tensors format, JAX is not installed.")
            import jax.numpy as jnp  # noqa: F811

            as_tensor = jnp.array
            is_tensor = _is_jax
        else:
            as_tensor = np.asarray
            is_tensor = _is_numpy

        # Do the tensor conversion in batch

        for key, value in self.items():

            if key == 'doc_ids':
                tensor = value
                self[key] = tensor
                continue

            elif key=='summ_bow':
                # test_label = [1,2,3,4,5,6,7,8,9]
                lbls = []
                for batch_values in value:
                    batch_labels = []
                    for v in batch_values:
                        try:
                            batch_labels.append(as_tensor(v))
                        except:
                            import pdb;pdb.set_trace()
                    # batch_labels.append(as_tensor(test_label))
                    lbls.append(batch_labels)

                self[key] = lbls
                continue

            elif key=='ext_labels':
                # test_label = [1,2,3,4,5,6,7,8,9]
                lbls = []
                for batch_values in value:
                    batch_labels = []
                    for v in batch_values:
                        try:
                            batch_labels.append(as_tensor(v))
                        except:
                            import pdb;pdb.set_trace()
                    # batch_labels.append(as_tensor(test_label))
                    lbls.append(batch_labels)

                self[key] = lbls
                continue

            elif key=='section_scores':
                # test_label = [1,2,3,4,5,6,7,8,9]
                lbls = []
                for batch_values in value:
                    batch_labels = []
                    for v in batch_values:
                        try:
                            batch_labels.append(as_tensor(v))
                        except:
                            import pdb;pdb.set_trace()
                    # batch_labels.append(as_tensor(test_label))
                    lbls.append(batch_labels)

                self[key] = lbls
                continue

            elif key=='labels':
                lbls = []
                for batch_values in value:
                    batch_labels = []
                    for v in batch_values:
                        try:
                            batch_labels.append(as_tensor(v))
                        except:
                            import pdb;
                            pdb.set_trace()
                    # batch_labels.append(as_tensor(test_label))
                    lbls.append(batch_labels)

                self[key] = lbls
                continue
            else:
                try:
                    if prepend_batch_axis:
                        value = [value]

                    if not is_tensor(value):
                        try:
                            tensor = as_tensor(value)
                        except:
                            import pdb;pdb.set_trace()
                        # Removing this for now in favor of controlling the shape with `prepend_batch_axis`
                        # # at-least2d
                        # if tensor.ndim > 2:
                        #     tensor = tensor.squeeze(0)
                        # elif tensor.ndim < 2:
                        #     tensor = tensor[None, :]

                        self[key] = tensor
                except:  # noqa E722
                    if key == "overflowing_tokens":
                        raise ValueError(
                            "Unable to create tensor returning overflowing tokens of different lengths. "
                            "Please see if a fast version of this tokenizer is available to have this feature available."
                        )
                    raise ValueError(
                        "Unable to create tensor, you should probably activate truncation and/or padding "
                        "with 'padding=True' 'truncation=True' to have batched tensors with the same length."
                    )

        return self

class TGSumTokenizer(LEDTokenizer):

    # topic_info = torch.load
    def set_global_idf(self):
        self.idf_info_global = torch.load("/disk1/sajad/datasets/sci/mup/bert_data/idf_info_global.pt")
        self.idf_info_section = torch.load("/disk1/sajad/datasets/sci/mup/bert_data/idf_info_section.pt")

    def generate_src_bow(self, topic_src_info_global, input_ids, doc_id, ids):
        all_bows_section = []
        truncate_section = input_ids.count(self.EOSECT_ID)
        vocab_size = self.idf_info_global["voc_size"]
        all_file_counter_section = self.idf_info_section["all"]
        all_file_counter_global = self.idf_info_global["all"]
        file_num_global = self.idf_info_global["num"]
        file_num_section = self.idf_info_section["num"]

        ### creating section bow

        # for topic_src_info_section in topic_src_info_sections[:truncate_section]:
        #     all_bow = torch.zeros([vocab_size], dtype=torch.float)
        #     all_counter = topic_src_info_section
        #     all_counter_sum = sum(all_counter.values())
        #
        #     for key, value in all_counter.items():
        #         all_tf = value / all_counter_sum
        #         all_file_count = all_file_counter_section[int(key)]
        #         all_idf = math.log(file_num_section / (all_file_count + 1.))
        #         all_bow[int(key)] = all_tf * all_idf
        #
        #     all_bows_section.append(all_bow)

        ### creating global bow
        all_bow_global = torch.zeros([vocab_size], dtype=torch.float)
        all_counter_global = topic_src_info_global
        all_counter_global_sum = sum(all_counter_global.values())
        for key, value in all_counter_global.items():
            all_tf = value / all_counter_global_sum
            all_file_count_global = all_file_counter_global[int(key)]
            all_idf = math.log(file_num_global / (all_file_count_global + 1.))
            all_bow_global[int(key)] = all_tf * all_idf

        # if len(all_bows_section) != input_ids.count(self.EOSECT_ID):
        #     import pdb;pdb.set_trace()
        # assert len(all_bows_section) == input_ids.count(self.EOSECT_ID), "N/A equal sections"

        # return all_bows_section, all_bow_global
        return all_bow_global

    def generate_summ_bow(self, topic_summ_infos):
        all_bows = []
        for topic_summ_info in topic_summ_infos:
            vocab_size = self.idf_info_global["voc_size"]
            all_bow = torch.zeros([vocab_size], dtype=torch.float)
            all_file_counter = self.idf_info_global["all"]
            all_counter = topic_summ_info["all"]
            for key in all_counter.keys():
                # all_file_count = all_file_counter[key]
                # if not self.args.use_idf:
                # if all_file_count > self.args.max_word_count or \
                #   all_file_count < self.args.min_word_count:
                #     continue
                all_bow[int(key)] = 1
            all_bows.append(all_bow)
        return all_bows


    def truncate_labels(self, encoded_inputs, doc_id):
        # if encoded_inputs['input_ids'][-2] != 2:

        # if doc_id == 'SP:d8cd0216bc99e82a957d527a342bcefc5b69ec3c':
        #     import pdb;pdb.set_trace()

            # encoded_inputs['input_ids'][-2] = 2
            # encoded_inputs['input_ids'][-1] = self.EOSECT_ID

        # if encoded_inputs['input_ids'][-2] != 2:
            # encoded_inputs['input_ids'][-2] = 2
            # encoded_inputs['input_ids'][-1] = self.EOSECT_ID

        encoded_input_id = np.array(encoded_inputs['input_ids'])
        indices = np.where(encoded_input_id == self.EOSECT_ID)

        try:
            if len(np.diff(indices)[0]) > 0 and np.diff(indices)[0][-1] < 20:
                encoded_inputs['input_ids'] = np.array(encoded_inputs['input_ids'])[:np.diff(indices)[0][-2]].tolist()
        except:
            import pdb;pdb.set_trace()
        input_ids = encoded_inputs['input_ids']
        pre_ext = encoded_inputs['ext_labels']
        pre_sect_scores = encoded_inputs['section_scores']
        sect_Len = input_ids.count(self.BOSECT_ID)

        post_ext = pre_ext[:sect_Len]
        post_scores = pre_sect_scores[:sect_Len]
        encoded_inputs['ext_labels'] = post_ext


        if encoded_inputs['input_ids'].count(self.BOSECT_ID) == len(encoded_inputs['ext_labels']) \
            and encoded_inputs['input_ids'].count(0) != sum([len(l[0]) for l in encoded_inputs['ext_labels']]):
            # should truncate sentence_labels from from the last section
            last_sect_Labels = encoded_inputs['ext_labels'][-1]
            should_remove_sents = sum([len(l[0]) for l in encoded_inputs['ext_labels']])-encoded_inputs['input_ids'].count(0)

            # get section len
            # encoded_input_id = np.array(encoded_inputs['input_ids'])
            # indices = np.where(encoded_input_id == self.EOSECT_ID)

            # if should_remove_sents < len(last_sect_Labels[0]):
            last_sect_Labels = [l[:-should_remove_sents] for l in last_sect_Labels]
            encoded_inputs['ext_labels'] = encoded_inputs['ext_labels'][:-1] + [last_sect_Labels]
            # elif should_remove_sents == len(last_sect_Labels[0]):
            #     encoded_inputs['ext_labels'] = encoded_inputs['ext_labels'] [:-1]
            # else:
            #     import pdb;pdb.set_trace()
            #     encoded_inputs['ext_labels'] = encoded_inputs['ext_labels'][:-1]
            #     last_sect_Labels = encoded_inputs['ext_labels'][-1]
            #
            #     from_2_last = should_remove_sents - len(last_sect_Labels[0])
            #
            #     last_sect_Labels = [l[:-from_2_last] for l in last_sect_Labels]
            #     encoded_inputs['ext_labels'] = encoded_inputs['ext_labels'][:-1] + [last_sect_Labels]

            # if len(last_sect_Labels[0]) == 0:


            encoded_inputs['input_ids'] = encoded_inputs['input_ids']


        # normalize post_scores
        sum_sects_scores = [[] for _ in range(len(post_scores[0]))]
        for sect_score in post_scores:
            for j, sum_sect_scores in enumerate(sect_score):
                sum_sects_scores[j].append(sum_sect_scores)

        for j, s in enumerate(sum_sects_scores):
            sum_ = sum(s)
            sum_sects_scores[j] = sum_

        normalized_sect_scores = []
        for sect_score in post_scores:
            norm_sect_scores = []
            for j in range(len(sum_sects_scores)):
                norm_sect_scores.append(sect_score[j] / sum_sects_scores[j])
            normalized_sect_scores.append(norm_sect_scores.copy())

        sum([s[0] for s in normalized_sect_scores])

        encoded_inputs['section_scores'] = normalized_sect_scores


        # input_ids_tmp = encoded_inputs['input_ids']
        # try:
        #     input_ids_tmp.remove(self.BOSECT_ID)
        #     input_ids_tmp.remove(self.EOSECT_ID)
        # except:
        #     import pdb;pdb.set_trace()
        sent_start_idx = [i for i, j in enumerate(encoded_inputs['input_ids']) if j == 0]
        sent_end_idx = [i for i, j in enumerate(encoded_inputs['input_ids']) if j == 2]
        encoded_inputs['sent_len'] = [e-s for e, s in zip(sent_end_idx, sent_start_idx)]
        # encoded_inputs['input_ids'] = [self.BOSECT_ID] + encoded_inputs['input_ids']
        encoded_inputs['section_len'] = [len(l[0]) for l in encoded_inputs['ext_labels']]

        # if doc_id == 'SP:d8cd0216bc99e82a957d527a342bcefc5b69ec3c':
        #     import pdb;pdb.set_trace()
        #################
        #################
        #################
        #################
        #
        # if encoded_inputs['section_len'][-1] == 0:
        #     encoded_input_id = np.array(encoded_inputs['input_ids'])
        #     indices = np.where(encoded_input_id == self.EOSECT_ID)[0]
        #     encoded_inputs['input_ids'] = encoded_input_id[:indices[-2]+1].tolist()
        #
        #     input_ids = encoded_inputs['input_ids']
        #     pre_ext = encoded_inputs['ext_labels']
        #     pre_sect_scores = encoded_inputs['section_scores']
        #     sect_Len = input_ids.count(self.BOSECT_ID)
        #
        #     post_ext = pre_ext[:sect_Len]
        #     post_scores = pre_sect_scores[:sect_Len]
        #     encoded_inputs['ext_labels'] = post_ext
        #
        #     if encoded_inputs['input_ids'].count(self.BOSECT_ID) == len(encoded_inputs['ext_labels']) \
        #             and encoded_inputs['input_ids'].count(0) != sum([len(l[0]) for l in encoded_inputs['ext_labels']]):
        #         # should truncate sentence_labels from from the last section
        #         last_sect_Labels = encoded_inputs['ext_labels'][-1]
        #         should_remove_sents = sum([len(l[0]) for l in encoded_inputs['ext_labels']]) - encoded_inputs[
        #             'input_ids'].count(0)
        #         last_sect_Labels = [l[:-should_remove_sents] for l in last_sect_Labels]
        #         encoded_inputs['ext_labels'] = encoded_inputs['ext_labels'][:-1] + [last_sect_Labels]
        #
        #         # if len(last_sect_Labels[0]) == 0:
        #
        #         encoded_inputs['input_ids'] = encoded_inputs['input_ids']
        #
        #     # normalize post_scores
        #     sum_sects_scores = [[] for _ in range(len(post_scores[0]))]
        #     for sect_score in post_scores:
        #         for j, sum_sect_scores in enumerate(sect_score):
        #             sum_sects_scores[j].append(sum_sect_scores)
        #
        #     for j, s in enumerate(sum_sects_scores):
        #         sum_ = sum(s)
        #         sum_sects_scores[j] = sum_
        #
        #     normalized_sect_scores = []
        #     for sect_score in post_scores:
        #         norm_sect_scores = []
        #         for j in range(len(sum_sects_scores)):
        #             norm_sect_scores.append(sect_score[j] / sum_sects_scores[j])
        #         normalized_sect_scores.append(norm_sect_scores.copy())
        #
        #     sum([s[0] for s in normalized_sect_scores])
        #
        #     encoded_inputs['section_scores'] = normalized_sect_scores

            # input_ids_tmp = encoded_inputs['input_ids']
            # try:
            #     input_ids_tmp.remove(self.BOSECT_ID)
            #     input_ids_tmp.remove(self.EOSECT_ID)
            # except:
            #     import pdb;pdb.set_trace()
            # sent_start_idx = [i for i, j in enumerate(encoded_inputs['input_ids']) if j == 0]
            # sent_end_idx = [i for i, j in enumerate(encoded_inputs['input_ids']) if j == 2]
            # encoded_inputs['sent_len'] = [e - s for e, s in zip(sent_end_idx, sent_start_idx)]
            ## encoded_inputs['input_ids'] = [self.BOSECT_ID] + encoded_inputs['input_ids']
            # encoded_inputs['section_len'] = [len(l[0]) for l in encoded_inputs['ext_labels']]
        ###########################
        ###########################
        ###########################
        ###########################

        # if sum(encoded_inputs['section_len']) != input_ids.count(0):
        #     import pdb;pdb.set_trace()

        # further truncating input_ids
        # encoded_input_id = np.array(encoded_inputs['input_ids'])
        # indices = np.where(encoded_input_id == self.EOSECT_ID)[0]
        # indices = np.concatenate(([0], indices))
        # sect_token_len = np.diff(indices)
        # sect_token_len = np.cumsum(sect_token_len)
        #
        # for j, sect_tkn_len in enumerate(sect_token_len):
        #
        #     encoded_inputs['input_ids'][:sect_token_len]




        # return encoded_inputs['input_ids'], encoded_inputs['ext_labels'], encoded_inputs['section_scores']
        return encoded_inputs

    def truncate_sequences(
        self,
        ids: List[int],
        pair_ids: Optional[List[int]] = None,
        num_tokens_to_remove: int = 0,
        truncation_strategy: Union[str, TruncationStrategy] = "longest_first",
        stride: int = 0,
    ) -> Tuple[List[int], List[int], List[int]]:

        if num_tokens_to_remove <= 0:
            return ids, pair_ids, []

        if not isinstance(truncation_strategy, TruncationStrategy):
            truncation_strategy = TruncationStrategy(truncation_strategy)

        overflowing_tokens = []
        if truncation_strategy == TruncationStrategy.ONLY_FIRST or (
            truncation_strategy == TruncationStrategy.LONGEST_FIRST and pair_ids is None
        ):
            if len(ids) > num_tokens_to_remove:
                window_len = min(len(ids), stride + num_tokens_to_remove)
                if self.truncation_side == "left":
                    overflowing_tokens = ids[:window_len]
                    ids = ids[num_tokens_to_remove:]
                elif self.truncation_side == "right":
                    overflowing_tokens = ids[-window_len:]
                    ids = ids[:-num_tokens_to_remove]

                else:
                    raise ValueError(f"invalid truncation strategy: {self.truncation_side}, use 'left' or 'right'.")

            else:
                error_msg = (
                    f"We need to remove {num_tokens_to_remove} to truncate the input "
                    f"but the first sequence has a length {len(ids)}. "
                )
                if truncation_strategy == TruncationStrategy.ONLY_FIRST:
                    error_msg = (
                        error_msg
                        + "Please select another truncation strategy than "
                        f"{truncation_strategy}, for instance 'longest_first' or 'only_second'."
                    )
                logger.error(error_msg)
        elif truncation_strategy == TruncationStrategy.LONGEST_FIRST:
            logger.warning(
                "Be aware, overflowing tokens are not returned for the setting you have chosen,"
                f" i.e. sequence pairs with the '{TruncationStrategy.LONGEST_FIRST.value}' "
                "truncation strategy. So the returned list will always be empty even if some "
                "tokens have been removed."
            )
            for _ in range(num_tokens_to_remove):
                if pair_ids is None or len(ids) > len(pair_ids):
                    if self.truncation_side == "right":
                        ids = ids[:-1]
                    elif self.truncation_side == "left":
                        ids = ids[1:]
                    else:
                        raise ValueError("invalid truncation strategy:" + str(self.truncation_side))
                else:
                    if self.truncation_side == "right":
                        pair_ids = pair_ids[:-1]
                    elif self.truncation_side == "left":
                        pair_ids = pair_ids[1:]
                    else:
                        raise ValueError("invalid truncation strategy:" + str(self.truncation_side))
        elif truncation_strategy == TruncationStrategy.ONLY_SECOND and pair_ids is not None:
            if len(pair_ids) > num_tokens_to_remove:
                window_len = min(len(pair_ids), stride + num_tokens_to_remove)
                if self.truncation_side == "right":
                    overflowing_tokens = pair_ids[-window_len:]
                    pair_ids = pair_ids[:-num_tokens_to_remove]
                elif self.truncation_side == "left":
                    overflowing_tokens = pair_ids[:window_len]
                    pair_ids = pair_ids[num_tokens_to_remove:]
                else:
                    raise ValueError("invalid truncation strategy:" + str(self.truncation_side))
            else:
                logger.error(
                    f"We need to remove {num_tokens_to_remove} to truncate the input "
                    f"but the second sequence has a length {len(pair_ids)}. "
                    f"Please select another truncation strategy than {truncation_strategy}, "
                    "for instance 'longest_first' or 'only_first'."
                )

        return (ids, pair_ids, overflowing_tokens)

    def prepare_for_model(
            self,
            ids: List[int],
            pair_ids: Optional[List[int]] = None,
            doc_ids=None,
            section_scores=None,
            ext_labels=None,
            topic_info_tuple=None,
            add_special_tokens: bool = False,
            padding: Union[bool, str, PaddingStrategy] = False,
            truncation: Union[bool, str, TruncationStrategy] = False,
            max_length: Optional[int] = None,
            stride: int = 0,
            pad_to_multiple_of: Optional[int] = None,
            return_tensors: Optional[Union[str, TensorType]] = None,
            return_token_type_ids: Optional[bool] = None,
            return_attention_mask: Optional[bool] = None,
            return_overflowing_tokens: bool = False,
            return_special_tokens_mask: bool = False,
            return_offsets_mapping: bool = False,
            return_length: bool = False,
            verbose: bool = True,
            prepend_batch_axis: bool = False,
            **kwargs
    ) -> BatchEncoding:
        # Backward compatibility for 'truncation_strategy', 'pad_to_max_length'
        padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            verbose=verbose,
            **kwargs,
        )


        self.set_global_idf()
        pair = bool(pair_ids is not None)
        len_ids = len(ids)
        len_pair_ids = len(pair_ids) if pair else 0

        if return_token_type_ids and not add_special_tokens:
            raise ValueError(
                "Asking to return token_type_ids while setting add_special_tokens to False "
                "results in an undefined behavior. Please set add_special_tokens to True or "
                "set return_token_type_ids to None."
            )

        if (
                return_overflowing_tokens
                and truncation_strategy == TruncationStrategy.LONGEST_FIRST
                and pair_ids is not None
        ):
            raise ValueError(
                "Not possible to return overflowing tokens for pair of sequences with the "
                "`longest_first`. Please select another truncation strategy than `longest_first`, "
                "for instance `only_second` or `only_first`."
            )

        # Load from model defaults
        if return_token_type_ids is None:
            return_token_type_ids = "token_type_ids" in self.model_input_names
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names

        encoded_inputs = {}

        # Compute the total size of the returned encodings
        add_special_tokens = False if 'tgt' not in doc_ids else True
        total_len = len_ids + len_pair_ids + (self.num_special_tokens_to_add(pair=pair) if add_special_tokens else 0)

        # Truncation: Handle max sequence length
        overflowing_tokens = []
        if truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE and max_length and total_len > max_length:
            ids, pair_ids, overflowing_tokens = self.truncate_sequences(
                ids,
                pair_ids=pair_ids,
                num_tokens_to_remove=total_len - max_length,
                truncation_strategy=truncation_strategy,
                stride=stride,
            )

            if ids[-1] != self.EOSECT_ID:
                ids[-1] = self.EOSECT_ID
                if ids[-2] != 2:
                    ids[-2] = 2


        if return_overflowing_tokens:
            encoded_inputs["overflowing_tokens"] = overflowing_tokens
            encoded_inputs["num_truncated_tokens"] = total_len - max_length

        # Add special tokens
        if 'tgt' in doc_ids:
            sequence = self.build_inputs_with_special_tokens(ids, pair_ids)
            token_type_ids = self.create_token_type_ids_from_sequences(ids, pair_ids)
        else:
            sequence = ids + pair_ids if pair else ids
            token_type_ids = [0] * len(ids) + ([0] * len(pair_ids) if pair else [])

        # Build output dictionary
        encoded_inputs["input_ids"] = sequence
        # print(len(sequence))
        # import pdb;pdb.set_trace()
        # if doc_ids == 'SP:f76f1289d7b47dd1bd381108f5b86a410613af9e':
        #     import pdb;
        #     pdb.set_trace()

        if encoded_inputs['input_ids'][-3] == self.EOSECT_ID:
            encoded_inputs['input_ids'] = encoded_inputs['input_ids'][:-3] + [self.EOSECT_ID]

        if encoded_inputs['input_ids'][-3] == 2:
            encoded_inputs['input_ids'] = encoded_inputs['input_ids'][:-2] + [self.EOSECT_ID]

        if encoded_inputs['input_ids'][-3] == self.BOSECT_ID and encoded_inputs['input_ids'][-2] == 2 and  encoded_inputs['input_ids'][-1] == self.EOSECT_ID:
            encoded_inputs['input_ids'] = encoded_inputs['input_ids'][:-3]
            # encoded_inputs['input_ids'][-4] = self.EOSECT_ID
            # encoded_inputs['input_ids'][-5] = 2

        if encoded_inputs['input_ids'][-1] != self.EOSECT_ID and encoded_inputs['input_ids'][-2] != 2:
            encoded_inputs['input_ids'][-1] = self.EOSECT_ID
            encoded_inputs['input_ids'][-2] = 2



        # section len should be eq to number of sents.



            # print(len(sequence))
        # print('no----')
        if ext_labels is not None:
            encoded_inputs['ext_labels'] = ext_labels
        if section_scores is not None:
            encoded_inputs['section_scores'] = section_scores
        if ext_labels is not None and section_scores is not None:
            encoded_inputs = self.truncate_labels(encoded_inputs, doc_ids)



        # print(len(encoded_inputs['input_ids']))
        if return_token_type_ids:
            encoded_inputs["token_type_ids"] = token_type_ids
        if return_special_tokens_mask:
            if add_special_tokens:
                encoded_inputs["special_tokens_mask"] = self.get_special_tokens_mask(ids, pair_ids)
            else:
                encoded_inputs["special_tokens_mask"] = [0] * len(sequence)


        # if sub_graph is not None:
            # encoded_inputs["subgraphs"] = self._prepare_subgraph(sub_graph)

        encoded_inputs["doc_ids"] = doc_ids
        # assert special tokens...

        # now truncate section scores and ext_labels based on truncated input_ids


        if topic_info_tuple is not None:
            # encoded_inputs["src_bow_section"], encoded_inputs['src_bow_global'] = \
            encoded_inputs['src_bow_global'] = \
                self.generate_src_bow(topic_info_tuple, encoded_inputs["input_ids"], doc_ids, ids)



        # Check lengths
        self._eventual_warn_about_too_long_sequence(encoded_inputs["input_ids"], max_length, verbose)
        # Padding
        if padding_strategy != PaddingStrategy.DO_NOT_PAD or return_attention_mask:
            encoded_inputs = self.pad(
                encoded_inputs,
                max_length=max_length,
                padding=padding_strategy.value,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
            )

        if 'tgt' not in doc_ids:
            idx = 0
            while idx < len(encoded_inputs['input_ids']):
                id = encoded_inputs['input_ids'][idx]
                if idx + 1 < len(encoded_inputs['input_ids']):
                    if id == self.BOSECT_ID:
                        assert encoded_inputs['input_ids'][idx + 1] == 0, f"after {self.BOSECT_ID} should be 0"
                    if id == 2:
                        assert ((encoded_inputs['input_ids'][idx + 1] == self.EOSECT_ID) or (
                                encoded_inputs['input_ids'][idx + 1] == 0)), f"after 2 should be {self.EOSECT_ID} or 0"
                    if id == self.EOSECT_ID:
                        if encoded_inputs['input_ids'][idx + 1] != self.BOSECT_ID:
                            import pdb;
                            pdb.set_trace()
                        assert encoded_inputs['input_ids'][
                                   idx + 1] == self.BOSECT_ID, f'after {self.EOSECT_ID} should be {self.BOSECT_ID}, not {encoded_inputs["input_ids"][idx + 1]}'
                idx += 1

            if 'section_len' in encoded_inputs.keys() and sum(encoded_inputs['section_len']) != encoded_inputs['input_ids'].count(0):
                import pdb;pdb.set_trace()

            if 'section_len' not in encoded_inputs.keys():
                import pdb;pdb.set_trace()

            assert sum(encoded_inputs['section_len']) == encoded_inputs['input_ids'].count(0), 'Discrep in sentence len'


        # print(len(encoded_inputs['input_ids']))

        if return_length:
            encoded_inputs["length"] = len(encoded_inputs["input_ids"])
        batch_outputs = BatchEncoding(
            encoded_inputs, tensor_type=return_tensors, prepend_batch_axis=prepend_batch_axis
        )
        return batch_outputs

    def _batch_encode_plus(
        self,
        batch_text_or_text_pairs= None,
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        doc_ids=None,
        is_target=False,
        topic_info_tuple=None,
        ext_labels=None,
        section_scores=None,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs
    ) -> BatchEncoding:

        self.BOSECT_ID, self.EOSECT_ID = self.convert_tokens_to_ids(['<sect>', '</sect>'])

        def get_input_ids(text):
            if isinstance(text, str):
                tokens = self.tokenize(text, **kwargs)
                return self.convert_tokens_to_ids(tokens)
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], str):
                if is_split_into_words:
                    tokens = list(
                        itertools.chain(*(self.tokenize(t, is_split_into_words=True, **kwargs) for t in text))
                    )
                    return self.convert_tokens_to_ids(tokens)
                else:
                    return self.convert_tokens_to_ids(text)
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], int):
                return text
            else:
                raise ValueError(
                    "Input is not valid. Should be a string, a list/tuple of strings or a list/tuple of integers."
                )

        if return_offsets_mapping:
            raise NotImplementedError(
                "return_offset_mapping is not available when using Python tokenizers. "
                "To use this feature, change your tokenizer to one deriving from "
                "transformers.PreTrainedTokenizerFast."
            )

        input_ids = []
        ext_labels_set = []
        if 'tgt' not in doc_ids[0]:
            for idx, ids_or_pair_ids in enumerate(batch_text_or_text_pairs):

                # if doc_ids[idx] == 'SP:d8cd0216bc99e82a957d527a342bcefc5b69ec3c':
                #     import pdb;
                #     pdb.set_trace()
                sections, pair_ids = ids_or_pair_ids, None
                first_ids_all = []
                old_ext_label = ext_labels[idx]
                new_ext_labels = []
                # sections
                for sect_idx, sect_sentences in enumerate(sections):
                    first_ids = [self.BOSECT_ID, 0]
                    # sentences
                    # import pdb;
                    # pdb.set_trace()
                    old_ext_label_section = old_ext_label[sect_idx]
                    new_ext_label_section = [[] for _ in range(len(old_ext_label_section))]
                    sect_sentences = [i.strip() for i in sect_sentences.split(' <SENTTT> ')]

                    for jsent, sent in enumerate(sect_sentences):
                        if len(sent.strip()) > 0:
                            first_id = get_input_ids(sent)
                            if len(first_id) > 1:
                                first_ids += first_id + [2, 0]
                                for sum_idx in range(len(old_ext_label_section)):
                                    new_ext_label_section[sum_idx].append(old_ext_label_section[sum_idx][jsent])

                            # else:
                            #     if doc_ids[idx] == 'SP:d8cd0216bc99e82a957d527a342bcefc5b69ec3c':
                            #         import pdb;
                            #         pdb.set_trace()
                        # else:
                        #     if doc_ids[idx] == 'SP:d8cd0216bc99e82a957d527a342bcefc5b69ec3c':
                        #         import pdb;
                        #         pdb.set_trace()

                    new_ext_labels.append(new_ext_label_section.copy())

                    first_ids = first_ids[:-1]

                    # first_ids = first_ids.remove(self.BOSECT_ID) if self.BOSECT_ID in first_id else first_id
                    # first_ids = first_ids.remove(self.EOSECT_ID) if self.EOSECT_ID in first_id else first_id

                    first_ids = first_ids + [self.EOSECT_ID]

                    first_ids_all.extend(first_ids.copy())
                first_ids = first_ids_all
                # if doc_ids[idx] == 'SP:d8cd0216bc99e82a957d527a342bcefc5b69ec3c':
                #     import pdb;
                #     pdb.set_trace()
                # first_ids =  first_ids
                second_ids = get_input_ids(pair_ids) if pair_ids is not None else None
                input_ids.append((first_ids, second_ids))
                ext_labels_set.append(new_ext_labels.copy())
        else:
            for ids_or_pair_ids in batch_text_or_text_pairs:
                if not isinstance(ids_or_pair_ids, (list, tuple)):
                    sections, pair_ids = ids_or_pair_ids, None
                elif is_split_into_words and not isinstance(ids_or_pair_ids[0], (list, tuple)):
                    sections, pair_ids = ids_or_pair_ids, None
                else:
                    sections, pair_ids = ids_or_pair_ids

                first_ids = get_input_ids(sections)
                second_ids = get_input_ids(pair_ids) if pair_ids is not None else None
                input_ids.append((first_ids, second_ids))
        batch_outputs = self._batch_prepare_for_model(
            input_ids,
            doc_ids=doc_ids,
            ext_labels=ext_labels_set,
            section_scores=section_scores,
            topic_info_tuple=topic_info_tuple,
            add_special_tokens=add_special_tokens,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
            return_token_type_ids=return_token_type_ids,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_length=return_length,
            return_tensors=return_tensors,
            verbose=verbose,
        )

        return BatchEncoding(batch_outputs, is_target=is_target)


    def prepare_seq2seq_batch(
                self,
                src_texts: List[str],
                tgt_texts: Optional[List[str]] = None,
                max_length: Optional[int] = None,
                max_target_length: Optional[int] = None,
                padding: str = "longest",
                return_tensors: str = None,
                truncation: bool = True,
                **kwargs,
        ) -> BatchEncoding:

            # warnings.warn(formatted_warning, FutureWarning)
            # mBART-specific kwargs that should be ignored by other models.
            kwargs.pop("src_lang", None)
            kwargs.pop("tgt_lang", None)
            if max_length is None:
                max_length = self.model_max_length

            # import pdb;pdb.set_trace()
            model_inputs = self(
                src_texts,
                add_special_tokens=True,
                return_tensors=return_tensors,
                max_length=max_length,
                padding=padding,
                truncation=truncation,
                **kwargs,
            )
            if tgt_texts is None:
                return model_inputs
            # Process tgt_texts
            if max_target_length is None:
                max_target_length = max_length
            with self.as_target_tokenizer():
                labels = self(
                    tgt_texts,
                    add_special_tokens=True,
                    return_tensors=return_tensors,
                    padding=padding,
                    max_length=max_target_length,
                    truncation=truncation,
                    **kwargs,
                )
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs


    def _batch_prepare_for_model(
        self,
        batch_ids_pairs: List[Union[PreTokenizedInputPair, Tuple[List[int], None]]],
        sub_graphs = None,
        doc_ids=None,
        ext_labels=None,
        section_scores=None,
        topic_info_tuple=None,
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[str] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_length: bool = False,
        verbose: bool = True,
    ) -> BatchEncoding:
        """
        Prepares a sequence of input id, or a pair of sequences of inputs ids so that it can be used by the model. It
        adds special tokens, truncates sequences if overflowing while taking into account the special tokens and
        manages a moving window (with user defined stride) for overflowing tokens

        Args:
            batch_ids_pairs: list of tokenized input ids or input ids pairs
        """

        batch_outputs = {}
        try:
            for idx, (first_ids, second_ids) in enumerate(batch_ids_pairs):
                outputs = self.prepare_for_model(
                    first_ids,
                    second_ids,
                    sub_graph=sub_graphs[idx] if sub_graphs is not None else None,
                    doc_ids=doc_ids[idx] if doc_ids is not None else None,
                    section_scores=section_scores[idx] if section_scores is not None else None,
                    ext_labels=ext_labels[idx] if 'tgt' not in doc_ids[idx] else None,
                    topic_info_tuple=topic_info_tuple["topic_info_global"][idx] if topic_info_tuple is not None else None,
                    add_special_tokens=add_special_tokens,
                    padding=PaddingStrategy.DO_NOT_PAD.value,  # we pad in batch afterward
                    truncation=truncation_strategy.value,
                    max_length=max_length,
                    stride=stride,
                    pad_to_multiple_of=None,  # we pad in batch afterward
                    return_attention_mask=False,  # we pad in batch afterward
                    return_token_type_ids=return_token_type_ids,
                    return_overflowing_tokens=return_overflowing_tokens,
                    return_special_tokens_mask=return_special_tokens_mask,
                    return_length=return_length,
                    return_tensors=None,  # We convert the whole batch to tensors at the end
                    prepend_batch_axis=False,
                    verbose=verbose,
                )

                for key, value in outputs.items():
                    if key not in batch_outputs:
                        batch_outputs[key] = []
                    batch_outputs[key].append(value)
        except:
            import pdb;pdb.set_trace()
        batch_outputs = self.pad(
            batch_outputs,
            padding=padding_strategy.value,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
        )

        batch_outputs = BatchEncoding(batch_outputs, tensor_type=return_tensors)

        return batch_outputs

    def pad(
        self,
        encoded_inputs: Union[
            BatchEncoding,
            List[BatchEncoding],
            Dict[str, EncodedInput],
            Dict[str, List[EncodedInput]],
            List[Dict[str, EncodedInput]],
        ],
        padding: Union[bool, str, PaddingStrategy] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        verbose: bool = True,
    ) -> BatchEncoding:
        # If we have a list of dicts, let's convert it in a dict of lists
        # We do this to allow using this method as a collate_fn function in PyTorch Dataloader
        if isinstance(encoded_inputs, (list, tuple)) and isinstance(encoded_inputs[0], (dict, BatchEncoding)):
            encoded_inputs = {key: [example[key] for example in encoded_inputs] for key in encoded_inputs[0].keys()}

        # try:
        # The model's main input name, usually `input_ids`, has be passed for padding
        if self.model_input_names[0] not in encoded_inputs:
            raise ValueError(
                "You should supply an encoding or a list of encodings to this method "
                f"that includes {self.model_input_names[0]}, but you provided {list(encoded_inputs.keys())}"
            )

        required_input = encoded_inputs[self.model_input_names[0]]

        if not required_input:
            if return_attention_mask:
                encoded_inputs["attention_mask"] = []
            return encoded_inputs

        # If we have PyTorch/TF/NumPy tensors/arrays as inputs, we cast them as python objects
        # and rebuild them afterwards if no return_tensors is specified
        # Note that we lose the specific device the tensor may be on for PyTorch

        first_element = required_input[0]
        if isinstance(first_element, (list, tuple)):
            # first_element might be an empty list/tuple in some edge cases so we grab the first non empty element.
            for item in required_input:
                if len(item) != 0:
                    first_element = item[0]
                    break
        # At this state, if `first_element` is still a list/tuple, it's an empty one so there is nothing to do.
        if not isinstance(first_element, (int, list, tuple)):
            if is_tf_available() and _is_tensorflow(first_element):
                return_tensors = "tf" if return_tensors is None else return_tensors
            elif is_torch_available() and _is_torch(first_element):
                return_tensors = "pt" if return_tensors is None else return_tensors
            elif isinstance(first_element, np.ndarray):
                return_tensors = "np" if return_tensors is None else return_tensors
            else:
                raise ValueError(
                    f"type of {first_element} unknown: {type(first_element)}. "
                    f"Should be one of a python, numpy, pytorch or tensorflow object."
                )

            for key, value in encoded_inputs.items():
                encoded_inputs[key] = to_py_obj(value)

        # Convert padding_strategy in PaddingStrategy
        padding_strategy, _, max_length, _ = self._get_padding_truncation_strategies(
            padding=padding, max_length=max_length, verbose=verbose
        )

        required_input = encoded_inputs[self.model_input_names[0]]
        if required_input and not isinstance(required_input[0], (list, tuple)):
            encoded_inputs = self._pad(
                encoded_inputs,
                max_length=max_length,
                padding_strategy=padding_strategy,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
            )
            return BatchEncoding(encoded_inputs, tensor_type=return_tensors)

        batch_size = len(required_input)
        assert all(
            len(v) == batch_size for v in encoded_inputs.values()
        ), "Some items in the output dictionary have a different batch size than others."

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = max(len(inputs) for inputs in required_input)
            padding_strategy = PaddingStrategy.MAX_LENGTH

        batch_outputs = {}
        for i in range(batch_size):
            inputs = dict((k, v[i]) for k, v in encoded_inputs.items())
            outputs = self._pad(
                inputs,
                max_length=max_length,
                padding_strategy=padding_strategy,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
            )

            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)

        return BatchEncoding(batch_outputs, tensor_type=return_tensors)

    def _pad(
        self,
        encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
        max_length: Optional[int] = None,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
    ) -> dict:
        """
        Pad encoded inputs (on left/right and up to predefined length or max length in the batch)

        Args:
            encoded_inputs:
                Dictionary of tokenized inputs (`List[int]`) or batch of tokenized inputs (`List[List[int]]`).
            max_length: maximum length of the returned list and optionally padding length (see below).
                Will truncate by taking into account the special tokens.
            padding_strategy: PaddingStrategy to use for padding.

                - PaddingStrategy.LONGEST Pad to the longest sequence in the batch
                - PaddingStrategy.MAX_LENGTH: Pad to the max length (default)
                - PaddingStrategy.DO_NOT_PAD: Do not pad
                The tokenizer padding sides are defined in self.padding_side:

                    - 'left': pads on the left of the sequences
                    - 'right': pads on the right of the sequences
            pad_to_multiple_of: (optional) Integer if set will pad the sequence to a multiple of the provided value.
                This is especially useful to enable the use of Tensor Core on NVIDIA hardware with compute capability
                >= 7.5 (Volta).
            return_attention_mask:
                (optional) Set to False to avoid returning attention mask (default: set to model specifics)
        """
        # Load from model defaults
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names

        required_input = encoded_inputs[self.model_input_names[0]]

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = len(required_input)

        if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        needs_to_be_padded = padding_strategy != PaddingStrategy.DO_NOT_PAD and len(required_input) != max_length

        # Initialize attention mask if not present.
        if return_attention_mask and "attention_mask" not in encoded_inputs:
            encoded_inputs["attention_mask"] = [1] * len(required_input)

        if needs_to_be_padded:
            difference = max_length - len(required_input)

            if self.padding_side == "right":
                if return_attention_mask:

                    encoded_inputs["attention_mask"] = encoded_inputs["attention_mask"] + [0] * difference
                if "token_type_ids" in encoded_inputs:
                    encoded_inputs["token_type_ids"] = (
                        encoded_inputs["token_type_ids"] + [self.pad_token_type_id] * difference
                    )
                if "special_tokens_mask" in encoded_inputs:
                    encoded_inputs["special_tokens_mask"] = encoded_inputs["special_tokens_mask"] + [1] * difference
                encoded_inputs[self.model_input_names[0]] = required_input + [self.pad_token_id] * difference
            elif self.padding_side == "left":
                if return_attention_mask:
                    encoded_inputs["attention_mask"] = [0] * difference + encoded_inputs["attention_mask"]
                if "token_type_ids" in encoded_inputs:
                    encoded_inputs["token_type_ids"] = [self.pad_token_type_id] * difference + encoded_inputs[
                        "token_type_ids"
                    ]
                if "special_tokens_mask" in encoded_inputs:
                    encoded_inputs["special_tokens_mask"] = [1] * difference + encoded_inputs["special_tokens_mask"]
                encoded_inputs[self.model_input_names[0]] = [self.pad_token_id] * difference + required_input
            else:
                raise ValueError("Invalid padding strategy:" + str(self.padding_side))

        if return_attention_mask and "global_attention_mask" in encoded_inputs:
            required_input = encoded_inputs[self.model_input_names[0]]
            # `global_attention_mask` need to have the same length as other (sequential) inputs.
            needs_to_be_padded = len(encoded_inputs["global_attention_mask"]) != len(required_input)

            if needs_to_be_padded:
                difference = len(required_input) - len(encoded_inputs["global_attention_mask"])

                if self.padding_side == "right":
                    # Use `-1` since `0` in `global_attention_mask` means `local attention` instead of `not to attend`
                    encoded_inputs["global_attention_mask"] = (
                        encoded_inputs["global_attention_mask"] + [-1] * difference
                    )
                elif self.padding_side == "left":
                    encoded_inputs["global_attention_mask"] = [-1] * difference + encoded_inputs[
                        "global_attention_mask"
                    ]
                else:
                    raise ValueError("Invalid padding strategy:" + str(self.padding_side))

        return encoded_inputs

import itertools
import math
from typing import Optional, Union, List, Dict, Any, Sequence, Tuple

import numpy as np

from transformers import TensorType, LEDTokenizer, is_tf_available, is_torch_available, is_flax_available
import torch

from transformers.tokenization_utils_base import TruncationStrategy, BatchEncoding, EncodingFast, TextInput, \
    TextInputPair, PreTokenizedInput, PreTokenizedInputPair, EncodedInput, EncodedInputPair
from transformers.utils.generic import _is_jax, _is_numpy, PaddingStrategy, _is_tensorflow, _is_torch, to_py_obj


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

    def generate_src_bow(self, topic_src_info_global, topic_src_info_sections, input_ids, doc_id, ids):
        all_bows_section = []
        truncate_section = min(input_ids.count(0), len(topic_src_info_sections))

        if input_ids.count(0) > truncate_section:
            # import pdb;pdb.set_trace()
            truncated_input_ids = []
            diff = input_ids.count(0) - truncate_section
            traveresed = 0
            for idd in reversed(input_ids):
                if traveresed < diff:
                    if idd == 0:
                        traveresed += 1
                    continue
                truncated_input_ids.append(idd)
            input_ids = [l for l in reversed(truncated_input_ids)]


        vocab_size = self.idf_info_global["voc_size"]
        all_file_counter_section = self.idf_info_section["all"]
        all_file_counter_global = self.idf_info_global["all"]
        file_num_global = self.idf_info_global["num"]
        file_num_section = self.idf_info_section["num"]

        ### creating section bow

        for topic_src_info_section in topic_src_info_sections[:truncate_section]:
            all_bow = torch.zeros([vocab_size], dtype=torch.float)
            all_counter = topic_src_info_section
            all_counter_sum = sum(all_counter.values())

            for key, value in all_counter.items():
                all_tf = value / all_counter_sum
                all_file_count = all_file_counter_section[int(key)]
                all_idf = math.log(file_num_section / (all_file_count + 1.))
                all_bow[int(key)] = all_tf * all_idf

            all_bows_section.append(all_bow)

        ### creating global bow
        all_bow_global = torch.zeros([vocab_size], dtype=torch.float)
        all_counter_global = topic_src_info_global
        all_counter_global_sum = sum(all_counter_global.values())
        for key, value in all_counter_global.items():
            all_tf = value / all_counter_global_sum
            all_file_count_global = all_file_counter_global[int(key)]
            all_idf = math.log(file_num_global / (all_file_count_global + 1.))
            all_bow_global[int(key)] = all_tf * all_idf

        if len(all_bows_section) != input_ids.count(0):
            import pdb;pdb.set_trace()
        assert len(all_bows_section) == input_ids.count(0), "N/A equal sections"

        return all_bows_section, all_bow_global

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

    def prepare_for_model(
            self,
            ids: List[int],
            pair_ids: Optional[List[int]] = None,
            doc_ids=None,
            topic_info_tuple=None,
            add_special_tokens: bool = True,
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

        if return_overflowing_tokens:
            encoded_inputs["overflowing_tokens"] = overflowing_tokens
            encoded_inputs["num_truncated_tokens"] = total_len - max_length

        # Add special tokens
        if add_special_tokens:
            sequence = self.build_inputs_with_special_tokens(ids, pair_ids)
            token_type_ids = self.create_token_type_ids_from_sequences(ids, pair_ids)
        else:
            sequence = ids + pair_ids if pair else ids
            token_type_ids = [0] * len(ids) + ([0] * len(pair_ids) if pair else [])

        # Build output dictionary
        encoded_inputs["input_ids"] = sequence
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

        # try:
        if topic_info_tuple is not None:
            encoded_inputs["src_bow_section"], encoded_inputs['src_bow_global'] = \
            self.generate_src_bow(topic_info_tuple[0], topic_info_tuple[1], encoded_inputs["input_ids"], doc_ids, ids)
        # except:
            # import pdb;pdb.set_trace()
            # encoded_inputs["summ_bow"] = self.generate_summ_bow(topic_info_tuple[1])

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
        if 'tgt' not in doc_ids[0]:
            for idx, ids_or_pair_ids in enumerate(batch_text_or_text_pairs):
                # if not isinstance(ids_or_pair_ids, (list, tuple)):
                #     ids, pair_ids = ids_or_pair_ids, None
                # elif is_split_into_words and not isinstance(ids_or_pair_ids[0], (list, tuple)):
                ids, pair_ids = ids_or_pair_ids, None

                # if doc_ids[idx] == "SP:bd9472600b9e7e4b407b0b2572179bc8cab7f272":
                #     import pdb;
                #     pdb.set_trace()
                # else:
                #     ids, pair_ids = ids_or_pair_ids

                # ids is a list of lists (sections)

                first_ids = []
                for id in ids:
                    first_id = get_input_ids(id)
                    first_ids += first_id + [2, 0]
                first_ids = first_ids[:-1]
                # if doc_ids[idx] == "SP:bd9472600b9e7e4b407b0b2572179bc8cab7f272":
                #     import pdb;
                #     pdb.set_trace()
                second_ids = get_input_ids(pair_ids) if pair_ids is not None else None
                input_ids.append((first_ids, second_ids))
        else:
            for ids_or_pair_ids in batch_text_or_text_pairs:
                if not isinstance(ids_or_pair_ids, (list, tuple)):
                    ids, pair_ids = ids_or_pair_ids, None
                elif is_split_into_words and not isinstance(ids_or_pair_ids[0], (list, tuple)):
                    ids, pair_ids = ids_or_pair_ids, None
                else:
                    ids, pair_ids = ids_or_pair_ids

                first_ids = get_input_ids(ids)
                second_ids = get_input_ids(pair_ids) if pair_ids is not None else None
                input_ids.append((first_ids, second_ids))

        batch_outputs = self._batch_prepare_for_model(
            input_ids,
            doc_ids=doc_ids,
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

            import pdb;pdb.set_trace()
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
        for idx, (first_ids, second_ids) in enumerate(batch_ids_pairs):
            # try:

            outputs = self.prepare_for_model(
                first_ids,
                second_ids,
                sub_graph=sub_graphs[idx] if sub_graphs is not None else None,
                doc_ids=doc_ids[idx] if doc_ids is not None else None,
                topic_info_tuple=(topic_info_tuple["topic_info_global"][idx], topic_info_tuple["topic_info_section"][idx]) if topic_info_tuple is not None else None,
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
            # except:
            #     import pdb;pdb.set_trace()

            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)


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
        """
        Pad a single encoded input or a batch of encoded inputs up to predefined length or to the max sequence length
        in the batch.

        Padding side (left/right) padding token ids are defined at the tokenizer level (with `self.padding_side`,
        `self.pad_token_id` and `self.pad_token_type_id`)

        <Tip>

        If the `encoded_inputs` passed are dictionary of numpy arrays, PyTorch tensors or TensorFlow tensors, the
        result will use the same type unless you provide a different tensor type with `return_tensors`. In the case of
        PyTorch tensors, you will lose the specific device of your tensors however.

        </Tip>

        Args:
            encoded_inputs ([`BatchEncoding`], list of [`BatchEncoding`], `Dict[str, List[int]]`, `Dict[str, List[List[int]]` or `List[Dict[str, List[int]]]`):
                Tokenized inputs. Can represent one input ([`BatchEncoding`] or `Dict[str, List[int]]`) or a batch of
                tokenized inputs (list of [`BatchEncoding`], *Dict[str, List[List[int]]]* or *List[Dict[str,
                List[int]]]*) so you can use this method during preprocessing as well as in a PyTorch Dataloader
                collate function.

                Instead of `List[int]` you can have tensors (numpy arrays, PyTorch tensors or TensorFlow tensors), see
                the note above for the return type.
            padding (`bool`, `str` or [`~file_utils.PaddingStrategy`], *optional*, defaults to `True`):
                 Select a strategy to pad the returned sequences (according to the model's padding side and padding
                 index) among:

                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                  lengths).
            max_length (`int`, *optional*):
                Maximum length of the returned list and optionally padding length (see above).
            pad_to_multiple_of (`int`, *optional*):
                If set will pad the sequence to a multiple of the provided value.

                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
                >= 7.5 (Volta).
            return_attention_mask (`bool`, *optional*):
                Whether to return the attention mask. If left to the default, will return the attention mask according
                to the specific tokenizer's default, defined by the `return_outputs` attribute.

                [What are attention masks?](../glossary#attention-mask)
            return_tensors (`str` or [`~file_utils.TensorType`], *optional*):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
            verbose (`bool`, *optional*, defaults to `True`):
                Whether or not to print more information and warnings.
        """
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
        # except:
        #     import pdb;pdb.set_trace()

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
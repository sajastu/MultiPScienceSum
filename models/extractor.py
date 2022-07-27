import torch
import torch.nn as nn

from transformers import RobertaModel


class TokenLevelEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pad = 1
        self.encoder = RobertaModel.from_pretrained('roberta-large', config=config)
        self.encoder.gradient_checkpointing = True

    def _create_token_position_ids(self, input_ids):
        # input_ids : bsz, length
        shp = input_ids.shape
        device = input_ids.device
        token_position_ids = torch.arange(shp[-1], dtype=torch.long, device=device, requires_grad=False)
        token_position_ids = token_position_ids.unsqueeze(0).expand(shp)
        return token_position_ids

    def _create_token_type_ids(self, input_ids):
        shp = input_ids.shape
        device = input_ids.device
        token_type_ids = torch.zeros(shp, dtype=torch.long, device=device, requires_grad=False)
        return token_type_ids

    def _create_token_attention_mask(self, input_ids):
        token_attention_mask = (input_ids != self.pad).float()
        return token_attention_mask

    def forward(self,
                input_ids=None,
                position_ids=None,
                token_type_ids=None,
                attention_mask=None,
                hidden=None,
                ):

        if hidden is None:
            # Convert Inputs to 2 D
            bsz, num, length = input_ids.shape
            n = bsz * num
            input_ids = input_ids.reshape(n, length)
            if position_ids is None:
                position_ids = self._create_token_position_ids(input_ids)
            else:
                position_ids = position_ids.reshape(n, length)

            if token_type_ids is None:
                token_type_ids = self._create_token_type_ids(input_ids)
            else:
                token_type_ids = token_type_ids.reshape(n, length)

            if attention_mask is None:
                attention_mask = self._create_token_attention_mask(input_ids)
            else:
                attention_mask = attention_mask.reshape(n, length)


            encoder_output = self.encoder(
                input_ids=input_ids,
                position_ids=position_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask
            )
            # sss=  [((input_ids[xx] == 0).nonzero(as_tuple=True)[0]) for xx in range(input_ids.shape[0])]

            whole_sents = (input_ids == 0).sum()
            try:
                hidden = torch.masked_select(encoder_output[0], (input_ids == 0).unsqueeze(-1).expand_as(encoder_output[0])).view(1, whole_sents, encoder_output[0].shape[-1])
            except:
                import pdb;pdb.set_trace()
        return hidden


class ChunkLevelEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_encoder = TokenLevelEncoder(config)

    def _create_sent_position_ids(self, input_hidden):
        # input_hidden: bsz, length, dim
        shp = input_hidden.shape
        device = input_hidden.device
        sent_position_ids = torch.arange(shp[-2], dtype=torch.long, device=device, requires_grad=False)
        sent_position_ids = sent_position_ids.unsqueeze(0).expand(shp[:-1])
        return sent_position_ids

    def forward(self,
                token_input_ids=None,
                token_position_ids=None,
                token_type_ids=None,
                token_attention_mask=None,
                chunk_input_hidden=None,
                chunk_position_ids=None,
                chunk_attention_mask=None,
                ):
        """
            input_ids: Batch x Number x length
            token_position_ids: Batch x Number x length
            token_type_ids: Batch x Number x length
            token_attention_mask: Batch x Number x length
            sent_position_ids: Batch x Number
            sent_attention_mask: Batch x Number x Number
        """
        if token_input_ids is None and chunk_input_hidden is None:
            raise ValueError("both token_input_ids and sent_input_hidden are None")
        elif chunk_input_hidden is None:
            chunk_input_hidden = self.token_encoder(
                input_ids=token_input_ids,
                position_ids=token_position_ids,
                token_type_ids=token_type_ids,
                attention_mask=token_attention_mask
            )

        if chunk_position_ids is None:
            chunk_position_ids = self._create_sent_position_ids(chunk_input_hidden)

        if chunk_attention_mask is None:
            raise ValueError("sent_attention_mask should not be None")

        chunk_level_outputs = self.chunk_encoder(
            input_emb=chunk_input_hidden,
            position_ids=chunk_position_ids,
            attention_mask=chunk_attention_mask
        )

        chunk_level_outputs = chunk_level_outputs[0][-1]

        return chunk_level_outputs



class ExtractorFirstStep(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = TokenLevelEncoder(config)
        self.saliency = BinaryClsHead(config.hidden_size, 1, 0.1)

    def forward(self,
                chunk_input_ids=None,
                chunk_hidden=None,
                chunk_attention_mask=None
                ):

        hidden = self.encoder(
            input_ids=chunk_input_ids,
            hidden=chunk_hidden
        )

        return self.saliency(hidden)


class BinaryClsHead(nn.Module):
    def __init__(self, n_in, n_out, dropout):
        super().__init__()
        self.dense = nn.Linear(n_in, n_out)
        self.dropout = nn.Dropout(p=dropout)
        self.out_proj = nn.Linear(n_out, 1)
        self.sigmoid = torch.sigmoid

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return torch.sigmoid(x.squeeze(-1))

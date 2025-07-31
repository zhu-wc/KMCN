import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from models.transformer.attention import MultiHeadAttention,MultiHeadAttentionWithHidden
from models.transformer.utils import sinusoid_encoding_table, PositionWiseFeedForward
from models.containers import Module, ModuleList


class DecoderLayer(Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, self_att_module=None,
                 enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
        super(DecoderLayer, self).__init__()
        self.self_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=True,
                                           attention_module=self_att_module,
                                           attention_module_kwargs=self_att_module_kwargs)
        self.enc_att1 = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=False,
                                          attention_module=enc_att_module,
                                          attention_module_kwargs=enc_att_module_kwargs)
        self.enc_att2 = MultiHeadAttentionWithHidden(d_model, d_k, d_v, h, dropout, can_be_stateful=False,
                                          attention_module=enc_att_module,
                                          attention_module_kwargs=enc_att_module_kwargs)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)

        # self.mlp1 = Hidden_MLP(d_model,128,dropout)
        # self.mlp2 = Hidden_MLP(d_model, 128, dropout)

        self.dropout1 = nn.Dropout(dropout)
        self.lnorm1 = nn.LayerNorm(d_model)

        # self.dropout2 = nn.Dropout(dropout)
        # self.lnorm2 = nn.LayerNorm(d_model)

        self.dropout3 = nn.Dropout(dropout)
        self.lnorm3 = nn.LayerNorm(d_model)


    def forward(self, input, enc_output, hidden_visual,hidden_ctx,mask_pad, mask_self_att, mask_enc_att):
        Visual_Feature = enc_output[:,:99] # bs 99 512
        Visual_mask = mask_enc_att[:,:,:,:99] # bs 1 1 99
        Ctx_Feature = enc_output[:,99:] # bs 48 512
        Ctx_mask = mask_enc_att[:,:,:,99:] # bs 1 1 48

        #Masked Self-Attention && Dropout+Add+LayerNrom
        self_att = self.self_att(input, input, input, mask_self_att)
        self_att = self.lnorm1(input + self.dropout1(self_att))
        self_att = self_att * mask_pad

        #Cross-Attention && Dropout+Add+LayerNorm
        enc_att1 = self.enc_att1(self_att, Ctx_Feature, Ctx_Feature,Ctx_mask)  # 将hidden_visual作为传入参数，则K=V=CTX features
        hidden_ctx = (hidden_ctx + enc_att1)*0.5
        enc_att2 = self.enc_att2(self_att, Visual_Feature, Visual_Feature,hidden_ctx,Visual_mask)  # 将hidden_ctx作为传入参数，则K=V=visual features


        # enc_att1 = self.lnorm2(self_att + self.dropout2(enc_att1))
        # enc_att1 = enc_att1 * mask_pad


        # enc_att2 = self.lnorm3(self_att + self.dropout3(enc_att2))
        # enc_att2 = enc_att2 * mask_pad

        enc_att = (enc_att1 + enc_att2)*0.5
        enc_att = self.lnorm3(self_att + self.dropout3(enc_att))
        enc_att = enc_att * mask_pad

        # FFN
        ff = self.pwff(enc_att)
        ff = ff * mask_pad

        #相对独立的简单的MLP网络微调CA的输出，作为隐藏单元
        # out_hidden_ctx = self.mlp1(enc_att1)
        # out_hidden_visual = self.mlp2(enc_att2)

        return ff


class TransformerDecoder(Module):
    def __init__(self, vocab_size, max_len, N_dec, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
                 self_att_module=None, enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
        super(TransformerDecoder, self).__init__()
        self.d_model = d_model
        self.word_emb = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(max_len + 1, d_model, 0), freeze=True)
        self.layers = ModuleList(
            [DecoderLayer(d_model, d_k, d_v, h, d_ff, dropout, self_att_module=self_att_module,
                                enc_att_module=enc_att_module, self_att_module_kwargs=self_att_module_kwargs,
                                enc_att_module_kwargs=enc_att_module_kwargs) for _ in range(N_dec)])
        self.fc = nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len
        self.padding_idx = padding_idx
        self.N = N_dec

        self.register_state('running_mask_self_attention', torch.zeros((1, 1, 0)).bool())
        self.register_state('running_seq', torch.zeros((1,)).long())

    def forward(self, input, encoder_output, mask_encoder):
        # input (b_s, seq_len)
        b_s, seq_len = input.shape[:2]
        mask_queries = (input != self.padding_idx).unsqueeze(-1).float()  # (b_s, seq_len, 1)
        mask_self_attention = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8, device=input.device),
                                         diagonal=1)
        mask_self_attention = mask_self_attention.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        mask_self_attention = mask_self_attention + (input == self.padding_idx).unsqueeze(1).unsqueeze(1).byte()
        mask_self_attention = mask_self_attention.gt(0)  # (b_s, 1, seq_len, seq_len)
        if self._is_stateful:
            self.running_mask_self_attention = torch.cat([self.running_mask_self_attention, mask_self_attention], -1)
            mask_self_attention = self.running_mask_self_attention

        seq = torch.arange(1, seq_len + 1).view(1, -1).expand(b_s, -1).to(input.device)  # (b_s, seq_len)
        seq = seq.masked_fill(mask_queries.squeeze(-1) == 0, 0)
        if self._is_stateful:
            self.running_seq.add_(1)
            seq = self.running_seq

        out = self.word_emb(input) + self.pos_emb(seq)


        # 1.拆分EncoderOut
        Visual_Feature = encoder_output[:,:99] # bs 99 512
        Visual_mask = mask_encoder[:,:,:,:99] # bs 1 1 99
        Ctx_Feature = encoder_output[:,99:] # bs 48 512
        # Ctx_mask = mask_encoder[:,:,:,99:] # bs 1 1 48

        hidden_visual = torch.sum(Visual_Feature,dim=1,keepdim=True) / (99 - torch.sum(Visual_mask,dim=-1)) # bs 1 512
        init_hidden_visual = hidden_visual.repeat(1,input.shape[1],1) # bs 20 512

        hidden_ctx = torch.mean(Ctx_Feature,dim=1,keepdim=True) # bs 1 512
        init_hidden_ctx = hidden_ctx.repeat(1,input.shape[1],1)# bs 20 512

        # out_hidden_visual = init_hidden_visual
        # out_hidden_ctx = init_hidden_ctx
        for l in self.layers:
            out = l(out, encoder_output, init_hidden_visual,init_hidden_ctx,mask_queries, mask_self_attention,mask_encoder) #利用visual进行解码
            # out_hidden_visual = (init_hidden_visual + out_hidden_visual)*0.5
            # out_hidden_ctx = (init_hidden_ctx + out_hidden_ctx)*0.5
            # out2 = l2(out2, encoder_output, hidden_visual,mask_queries, mask_self_attention,mask_encoder) #利用CTX进行解码
            # hidden_ctx = out2
            # out = (out1 + out2) * 0.5

        out = self.fc(out)
        # out2 = self.fc2(out2)
        # out = (F.log_softmax(out1, dim=-1) + F.log_softmax(out2, dim=-1))*0.5
        return F.log_softmax(out, dim=-1)
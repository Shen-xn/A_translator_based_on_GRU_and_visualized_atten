import os
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as nn_utils

# 构建编码器 - 批量序列全部输入 
# input, hidden: (batch, seq_len),（num_layers * num_directions, batch, hidden_size）  
# -> output, hidden: (seq_len, batch, hidden_size） * 2), （batch, hidden_size)

class Norm_GRU(nn.Module):
    def __init__(self, 
                 hidden_size, # 隐藏层维度
                 ):
        super().__init__()
        self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size)
        self.ln_out = nn.LayerNorm(hidden_size)
        self.ln_hidden = nn.LayerNorm(hidden_size)
        
    def forward(self, input, hidden):
        output, hidden = self.gru(input, hidden)
        output = self.ln_out(output)
        hidden = self.ln_hidden(hidden)
        return output, hidden
        
        
class   EncoderRNN(nn.Module):
    # 初始化，引入需要的层和必要的参数
    def __init__(self, 
                 input_size,  # 输入维度 - 也是嵌入层的输出维度
                 hidden_size, # 隐藏层维度
                 device,
                 num_encoder_layers = 1, # 采用编码器的GRU层数
                 padding_index = 2,
                 ):
        super(EncoderRNN, self).__init__()

        self.device = device

        self.hidden_size = hidden_size
        self.num_encoder_layers = num_encoder_layers

        self.embedding = nn.Embedding(input_size, hidden_size, padding_idx=padding_index)  # 定义嵌入层，输入one_hot词编号 -> 输出连续词向量
        
        self.grus = nn.ModuleList([Norm_GRU(hidden_size) for _ in range(num_encoder_layers)])
        
        self.fc = nn.Linear(hidden_size, hidden_size)
        
        self.to(self.device)
        self.model_half = False

    def to(self, device):
        self.device = device
        return super().to(device)
    
    def half(self):
        self.model_half=True
        return super().half()

    # 定义向前传播
    def forward(self, input, hidden=None):
          
        # (batch, seq_len) -> (batch, seq_len, hidden_size)
        embedded = self.embedding(input)
        
        batch_size = input.size(0)
        if hidden is None:
            hidden = self.initHidden(batch_size)  
       
        # (batch, seq_len, hidden_size) -> (seq_len, batch, hidden_size)
        x = embedded.permute(1, 0, 2)
        
        # (seq_len, batch, input_size)，（num_layers * num_directions, batch, hidden_size）
        # -> (seq_len, batch, input_size*2)，（num_layers, batch, hidden_size）
        h_list = list(hidden.chunk(self.num_encoder_layers, dim=0))
        new_h_list = []
        for i, layer in enumerate(self.grus):
            x, hi = layer(x, h_list[i])         # x:(T,B,H) -> 下一层输入
            new_h_list.append(hi) 
            
        hidden = torch.cat(new_h_list)
        output = self.fc(x)

 
        return output, hidden  # (seq_len, batch, input_size), （batch, hidden_size）

    def initHidden(self, batch_size=1):
        if self.model_half:
            return torch.zeros(self.num_encoder_layers, batch_size, self.hidden_size, device=self.device).half()
        else:
            return torch.zeros(self.num_encoder_layers, batch_size, self.hidden_size, device=self.device)
    

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.Wq = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.Wk = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.Wv = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.ln = nn.LayerNorm(self.hidden_size)
        
    def forward(self, input_matrix, query_seed_vector = None):
            # input_matrix: (T,B,H) -> (B,T,H)
            input_matrix = input_matrix.permute(1, 0, 2).contiguous()  # (B,T,H)

            # K, V 来自输入
            K = self.Wk(input_matrix)  # (B,T,H)
            V = self.Wv(input_matrix)  # (B,T,H)

            scale = 1.0 / math.sqrt(self.hidden_size)

            if query_seed_vector is None:
                # 自注意力：Q 也来自输入
                Q = self.Wq(input_matrix)  # (B,T,H)

                # scores: (B,T,T)
                scores = torch.bmm(Q, K.transpose(1, 2)) * scale
                atten = F.softmax(scores, dim=-1)               # (B,T,T)

                #(B,T,H) -> (T,B,H)
                out_bt = torch.bmm(atten, V)                 # (B,T,H)
                out_tb = out_bt.permute(1, 0, 2).contiguous()  # (T,B,H)
                out_tb = self.ln(input_matrix + out_tb)
                
                return out_tb, atten

            else:
                if query_seed_vector.dim() == 1:
                    B = input_matrix.size(0)
                    q_bh = query_seed_vector.unsqueeze(0).expand(B, -1)  # (B,H)
                elif query_seed_vector.dim() == 2:
                    q_bh = query_seed_vector  # (B,H)
                else:
                    raise ValueError("query_seed_vector must be of shape (H,) or (B, H)")

                Q_b1h = self.Wq(q_bh).unsqueeze(1)             # (B,1,H)

                # scores: (B,1,T)
                scores = torch.bmm(Q_b1h, K.transpose(1, 2)) * scale
                atten = F.softmax(scores, dim=-1)                # (B,1,T)

                # 输出: (B,1,H) -> (1,B,H)
                out_b1h = torch.bmm(atten, V)                 # (B,1,H)
                out_tbh = out_b1h.permute(1, 0, 2).contiguous() # (1,B,H)
                out_tbh = self.ln(query_seed_vector + out_tbh)
                return out_tbh, atten
    

# 注意力机制解码器 - 直接受依次批次输入
# input, hidden, encoder_outputs: (batch, 1), （i, batch, hidden_size）, (seq_len, batch, hidden_size） * 2) 
# -> output, hidden, attn_weights: (1, batch，num_words), (num_layer, batch, hidden_size), (batch, max_length)
class AttnDecoderRNN(nn.Module):
    def __init__(
            self, 
            hidden_size, 
            output_size,
            device, 
            num_decoder_layers = 1, # 解码器的GRU层数``
            padding_index = 2,
            dropout_rate=0.1, 
            ):
        super(AttnDecoderRNN, self).__init__()

        self.device = device
        self.hidden_size = hidden_size
        self.num_decoder_layers = num_decoder_layers
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        self.embedding = nn.Embedding(self.output_size, self.hidden_size, padding_idx=padding_index)  
        self.attention_emb = Attention(self.hidden_size)
        self.attention_enc = Attention(self.hidden_size)
        # 行代码定义了另一个线性层，用于将当前GRU的输出和通过注意力机制加权后的编码器输出结合起来。
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.grus = nn.ModuleList([Norm_GRU(self.hidden_size) for _ in range(self.num_decoder_layers)])
        self.ln_embedding = nn.LayerNorm(self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.embedding_list = []
        self.to(self.device)
        self.model_half=False
        
    def to(self, device):
        self.device = device
        return super().to(device)
    
    def half(self):
        self.model_half=True
        return super().half()

    def forward(self, input, encoder_output, hidden=None):
        batch_size = encoder_output.size(1)
        if hidden==None:
            hidden = self.initHidden(batch_size)
            
        embedded = self.embedding(input)  # (batch, 1, hidden_size)
        embedded = embedded.permute(1, 0, 2) # (1, batch_size, hidden_size)
        
        self.embedding_list.append(embedded)
        embedding_mat = torch.cat(self.embedding_list, dim=0)
        attn_embedding, attn_emb = self.attention_emb(embedding_mat, embedded[0])  # (1, batch_size, hidden_size)
        attn_embedding = embedded + attn_embedding
        attn_embedding = self.ln_embedding(attn_embedding)
        
        attn_applied, attn_enc = self.attention_enc(encoder_output, attn_embedding[0])
       
        x = torch.cat((attn_embedding, attn_applied), 2)  # -> （1, batch，hidden_size * 2）

        # 将嵌入的输出和加权后的编码器输出拼接起来，并通过另一个线性层进行处理。
        x = self.attn_combine(x) # -> (1, batch，hidden_size)

        x = F.relu(x)  # -> (1, batch，hidden_size)
        
        # (seq_len, batch, input_size)，（num_layers * num_directions, batch, hidden_size）
        # -> (seq_len, batch, input_size*2)，（num_layers, batch, hidden_size）
        h_list = list(hidden.chunk(self.num_decoder_layers, dim=0))
        new_h_list = []
        for i, layer in enumerate(self.grus):
            x, hi = layer(x, h_list[i])         # x:(T,B,H) -> 下一层输入
            new_h_list.append(hi) 

        hidden = torch.cat(new_h_list, dim=0)  
        output = self.dropout(x)  # (1, batch，hidden_size)
        output = self.out(output)  # -> (1, batch，hidden_size)
        return output, hidden, attn_emb, attn_enc
        

    def initHidden(self, batch_size=1):
        if self.model_half:
            return torch.zeros(self.num_decoder_layers, batch_size, self.hidden_size, device=self.device).half()
        else: 
            return torch.zeros(self.num_decoder_layers, batch_size, self.hidden_size, device=self.device)
    
    def reset_state(self):
        self.embedding_list = []
    

class translator:
    def __init__(self,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 hidden_size: int,
                 device: str,
                 num_encoder_layers: int = 1, 
                 num_decoder_layers: int = 1,
                 dropout_rate: float = 0.1,
                 max_len: int = 50,
                 sos_idx: int = 0,
                 eos_idx: int = 1,
                 padding_idx: int = 2):

        self.device = device
        self.max_input_len = max_len
        self.max_output_len = max_len
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx


        self.encoder = EncoderRNN(
            input_size=src_vocab_size,
            hidden_size=hidden_size,
            device=device,
            num_encoder_layers=num_encoder_layers,
            padding_index = padding_idx
        )

        self.decoder = AttnDecoderRNN(
            hidden_size=hidden_size,
            output_size=tgt_vocab_size,
            device=device,
            num_decoder_layers=num_decoder_layers,
            dropout_rate=dropout_rate,
            padding_index = padding_idx
        )

        self.encoder.eval()
        self.decoder.eval()
    
    def to(self, device):
        self.encoder.to(device)
        self.decoder.to(device)
        self.device = device
        
    def half(self):
        self.encoder.half()
        self.decoder.half()


    # 单句推理（greedy）
    @torch.no_grad()
    def translate_sentence(self, input_ids: torch.Tensor): # input: (T, ) or (1, T)

        self.encoder.eval()
        self.decoder.eval()

        # 截断输入
        if input_ids.size(0) > self.max_input_len:
            input_ids = input_ids[:self.max_input_len]
            
        src = input_ids.view(1, -1).to(self.device) # (1, T_in)
        B = 1

        # 编码
        enc_hidden = self.encoder.initHidden(batch_size=B)
        enc_out, enc_hidden = self.encoder(src, enc_hidden)   # enc_out: (T,B,H), enc_hidden:(L,B,H)

        # 解码初始化
        self.decoder.reset_state()
        dec_hidden = self.decoder.initHidden(batch_size=B)                          # (L,B,H)
        dec_input = torch.full((B, 1), self.sos_idx, device=self.device, dtype=torch.long)

        outputs = []
        for _ in range(self.max_output_len):
            # decoder(input, encoder_output, hidden)
            dec_out, dec_hidden, attn_emb, atten_enc = self.decoder(dec_input, enc_out, dec_hidden)  # dec_out:(1,B,V)
            topv, topi = dec_out.topk(1, dim=2)                                       # (1,B,1)
            next_tok = topi.squeeze(0)                                                # (B,1)
            outputs.append(int(next_tok.item()))
            dec_input = next_tok.detach()                                             # (B,1)

            if (self.eos_idx is not None) and (outputs[-1] == self.eos_idx) or len(outputs)>self.max_output_len :
                break
        
        return outputs, attn_emb, atten_enc

    # 批量训练（单个 batch）
    def train_on_batch(self,
                       input_tensor: torch.Tensor,   # (B, T_in) padded
                       target_tensor: torch.Tensor,  # (B, T_out) padded
                       encoder_optimizer,
                       decoder_optimizer,
                       criterion,
                       teacher_forcing_ratio: float = 0.5):
        
        self.encoder.train()
        self.decoder.train()

        device = self.device
        B, _ = input_tensor.size()

        # 编码
        enc_hidden = self.encoder.initHidden(batch_size=B)
        enc_out, enc_hidden = self.encoder(input_tensor.to(device), enc_hidden)   # enc_out:(T,B,H)

        # 解码起始
        self.decoder.reset_state()
        dec_hidden = None                                                   # (L,B,H)
        dec_input = torch.full((B, 1), self.sos_idx, device=device, dtype=torch.long) # (L,B,1)

        use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio
        tgt_TB = target_tensor.to(device).t()   # (T_out, B)   
        dec_out_collector = []
        for t in range(tgt_TB.size(0)):
            target_word = tgt_TB[t].unsqueeze(1)                                  # (B,1)

            dec_out, dec_hidden, _, _ = self.decoder(dec_input, enc_out, dec_hidden)  # dec_out:(1,B,V)
            dec_out_collector.append(dec_out[0])
            # if not torch.isfinite(loss):
            #     raise ValueError("exploed")

            if use_teacher_forcing:
                dec_input = target_word                                           # Teacher forcing
            else:
                topv, topi = dec_out.topk(1, dim=2)                               # (1,B,1)
                dec_input = topi.squeeze(0).detach()                              # (B,1)
        dec_out_collector = torch.cat(dec_out_collector) # [T*B, V]
        target = tgt_TB.reshape(-1) # [T*B]
        loss = criterion(dec_out_collector, target)
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()
        nn_utils.clip_grad_norm_(self.encoder.parameters(), max_norm=1.0)
        nn_utils.clip_grad_norm_(self.decoder.parameters(), max_norm=1.0)
        # 返回时序平均损失
        return loss.item()

    # 批量评估（greedy 生成）
        # 输入: (B, T_in) LongTensor（已 padding）
        # 输出: (B, T_out_gen) 的 LongTensor（greedy，长度<=max_output_len）
    @torch.no_grad()
    def evaluate_batch(self,
                       input_tensor: torch.Tensor,   # (B, T_in) padded
                       target_tensor: torch.Tensor,  # (B, T_out) padded
                       criterion):
        self.encoder.eval()
        self.decoder.eval()

        device = self.device
        B, _ = input_tensor.size()

        # 编码
        enc_hidden = self.encoder.initHidden(batch_size=B)
        enc_out, enc_hidden = self.encoder(input_tensor.to(device), enc_hidden)   # enc_out:(T,B,H)

        # 解码起始
        self.decoder.reset_state()
        dec_hidden = None                                                   # (L,B,H)
        dec_input = torch.full((B, 1), self.sos_idx, device=device, dtype=torch.long) # (L,B,1)
        tgt_TB = target_tensor.to(device).t()   # (T_out, B)
        dec_out_collector = []
        for t in range(tgt_TB.size(0)):            
            dec_out, dec_hidden, _, _ = self.decoder(dec_input, enc_out, dec_hidden)  # dec_out:(1,B,V)
            dec_out_collector.append(dec_out[0])
            
            # dec_out.squeeze(0)->(B,V)
            _, topi = dec_out.topk(1, dim=2)                               # (1,B,1)
            dec_input = topi.squeeze(0).detach()                              # (B,1)
            
        dec_out_collector = torch.cat(dec_out_collector) # [T*B, V]
        target = tgt_TB.reshape(-1) # [T*B]
        loss = criterion(dec_out_collector, target)
        # 返回时序平均损失
        return loss.item()
    

    # 保存 / 加载
    def save(self, model_dir: str, extra: dict = None):
        os.makedirs(model_dir, exist_ok=True)
        torch.save(self.encoder.state_dict(), os.path.join(model_dir, "encoder.pt"))
        torch.save(self.decoder.state_dict(), os.path.join(model_dir, "decoder.pt"))
        cfg = {
            "device": self.device,
            "max_input_len": self.max_input_len,
            "max_output_len": self.max_output_len,
            "sos_idx": self.sos_idx,
            "eos_idx": self.eos_idx,
            "hidden_size": getattr(self.encoder, "hidden_size", None),
            "num_encoder_layers": getattr(self.encoder, "num_encoder_layers", None),
            "num_decoder_layers": getattr(self.decoder, "num_decoder_layers", None),
        }
        if extra is not None:
            cfg.update(extra)
        with open(os.path.join(model_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)

    def load(self, model_dir: str, map_location=None):
        """
        注意：此处保持与之前相同的行为——要求当前实例的
        encoder/decoder 超参与保存时一致；然后再加载权重。
        """
        enc_p = os.path.join(model_dir, "encoder.pt")
        dec_p = os.path.join(model_dir, "decoder.pt")
        self.encoder.load_state_dict(torch.load(enc_p, map_location=map_location or self.device))
        self.decoder.load_state_dict(torch.load(dec_p, map_location=map_location or self.device))
        cfg_p = os.path.join(model_dir, "config.json")
        if os.path.exists(cfg_p):
            with open(cfg_p, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            self.max_input_len = cfg.get("max_input_len", self.max_input_len)
            self.max_output_len = cfg.get("max_output_len", self.max_output_len)
            self.sos_idx = cfg.get("sos_idx", self.sos_idx)
            self.eos_idx = cfg.get("eos_idx", self.eos_idx)

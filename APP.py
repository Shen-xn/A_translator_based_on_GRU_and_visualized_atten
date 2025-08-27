# gui_translator.py
import os, json, re, pickle, tkinter as tk
from tkinter import ttk, filedialog, messagebox
import torch

# 你的工程内模块
from text_processing_lib.data_processing  import dictionary, Lang, normalizeString 
from RNN_with_atten.NN import translator

# ---------- 句子切分与后处理 ----------
_SPLIT = re.compile(r'([.!?])')

def split_sentences(text: str):
    text = normalizeString(text).strip()
    if not text: return []
    parts, cur, out = _SPIT(parts=None), "", []
def _SPIT(parts):
    return _SPLIT.split(parts) if parts is not None else []

def postprocess_sentence(s: str):
    s = s.strip()
    if not s: return s
    s = s[0].upper() + s[1:]
    if not re.search(r'[.!?]$', s):
        s += '.'
    return s

# ---------- 加载 tokenizer ----------
def _load_tokenizer_pkl(pkl_path: str):
    with open(pkl_path, "rb") as f:
        payload = pickle.load(f)
    lang1 = Lang(payload["lang1_name"])
    lang1.word2index = payload["lang1_word2index"]
    lang1.index2word = payload["lang1_index2word"]
    lang1.n_words = max(lang1.index2word.keys()) + 1

    lang2 = Lang(payload["lang2_name"])
    lang2.word2index = payload["lang2_word2index"]
    lang2.index2word = payload["lang2_index2word"]
    lang2.n_words = max(lang2.index2word.keys()) + 1
    return lang1, lang2

def _load_tokenizer_pt(pt_path: str):
    payload = torch.load(pt_path, map_location="cpu")
    # 兼容 torch.save(dict) 的情况
    lang1 = Lang(payload["lang1_name"])
    lang1.word2index = payload["lang1_word2index"]
    lang1.index2word = payload["lang1_index2word"]
    lang1.n_words = max(lang1.index2word.keys()) + 1

    lang2 = Lang(payload["lang2_name"])
    lang2.word2index = payload["lang2_word2index"]
    lang2.index2word = payload["lang2_index2word"]
    lang2.n_words = max(lang2.index2word.keys()) + 1
    return lang1, lang2

def find_and_load_tokenizer(model_dir: str, direction: str):
    """
    优先在模型目录查找：
      1) tokenizer.pt
      2) *.tokenizer.pkl（如 en_rus.tokenizer.pkl / rus_eng.tokenizer.pkl）
    再退回到 模型目录/Tokenizer/ 下寻找同名 pkl
    最后弹窗人工选择。
    """
    # 方向决定默认名字
    name_pair = "en_rus" if direction == "eng2rus" else "rus_eng"

    # 1) model_dir 直查
    cand = [
        os.path.join(model_dir, "tokenizer.pt"),
        os.path.join(model_dir, f"{name_pair}.tokenizer.pkl"),
    ]
    # 2) model_dir/Tokenizer
    cand += [
        os.path.join(model_dir, "Tokenizer", "tokenizer.pt"),
        os.path.join(model_dir, "Tokenizer", f"{name_pair}.tokenizer.pkl"),
    ]
    for p in cand:
        if os.path.exists(p):
            try:
                if p.endswith(".pt"):
                    return _load_tokenizer_pt(p)
                else:
                    return _load_tokenizer_pkl(p)
            except Exception:
                pass

    # 3) 让用户选择
    messagebox.showwarning("未找到 tokenizer", "请选择 tokenizer（.pt 或 .tokenizer.pkl）")
    p = filedialog.askopenfilename(title="选择 tokenizer 文件", filetypes=[("Tokenizer", "*.pt *.pkl")])
    if not p:
        raise FileNotFoundError("未提供 tokenizer")
    if p.endswith(".pt"):
        return _load_tokenizer_pt(p)
    else:
        return _load_tokenizer_pkl(p)

# ---------- 加载 agent ----------
def load_agent(direction: str):
    direction = direction.lower()
    assert direction in ("eng2rus", "rus2eng")
    # 模型目录：默认 applied_model/<direction>/
    default_dir = os.path.join("applied_model", direction)
    if not os.path.exists(os.path.join(default_dir, "config.json")):
        messagebox.showwarning("模型未找到", f"未在 {default_dir} 找到模型，将手动选择")
        default_dir = filedialog.askdirectory(title="选择模型目录")
        if not default_dir:
            raise FileNotFoundError("未选择模型目录")
    cfg_p = os.path.join(default_dir, "config.json")
    with open(cfg_p, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    hidden_size       = cfg.get("hidden_size")
    num_encoder_layers= cfg.get("num_encoder_layers", 1)
    num_decoder_layers= cfg.get("num_decoder_layers", 1)
    max_len           = cfg.get("max_input_len", 50)
    sos_idx           = cfg.get("sos_idx", 0)
    eos_idx           = cfg.get("eos_idx", 1)

    # 载入 tokenizer（双向）
    lang_src, lang_tgt = find_and_load_tokenizer(default_dir, direction)

    # 用词表初始化 translator，再加载权重
    agent = translator(
        src_vocab_size=lang_src.n_words,
        tgt_vocab_size=lang_tgt.n_words,
        hidden_size=hidden_size,
        device=("cuda" if (torch.cuda.is_available()) else "cpu"),
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dropout_rate=0.1,
        max_len=max_len,
        sos_idx=sos_idx,
        eos_idx=eos_idx,
    )
    agent.load(default_dir)  # 会同步 config 里的 max_len/sos/eos

    # 绑定词表，推理解码时复用 Lang 的 Tensor2Sentence
    agent.input_lang  = lang_src
    agent.output_lang = lang_tgt
    return agent

# ---------- 推理：分句 -> 逐句翻译 -> 拼接 ----------
def translate_paragraph(agent, text: str):
    # 分句（保留 .!?），逐句送入
    parts = _SPLIT.split(text)
    sents, cur = [], ""
    for p in parts:
        if _SPLIT.fullmatch(p):
            cur += p
            sents.append(cur.strip())
            cur = ""
        else:
            cur += p
    if cur.strip():
        sents.append(cur.strip())
    sents = [s for s in sents if s]

    outs = []
    for s in sents:
        # 编码到固定长度：不加 <SOS>（encoder 不需要），对齐到 max_input_len
        inp = agent.input_lang.Sentence2Tensor(
            normalizeString(s),
            alignment=True, max_length=agent.max_input_len, include_sos=False
        )
        # (T,) -> translate
        ids, _, _ = agent.translate_sentence(inp)
        hyp = agent.output_lang.Tensor2Sentence(ids)
        outs.append(postprocess_sentence(hyp))
    return " ".join(outs)

# ======================== Tkinter GUI ========================
class TranslatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Seq2Seq Translator")
        self.agent = None

        self.dir_var = tk.StringVar(value="eng2rus")

        top = ttk.Frame(root, padding=10); top.pack(fill=tk.BOTH, expand=True)

        row0 = ttk.Frame(top); row0.pack(fill=tk.X, pady=5)
        ttk.Label(row0, text="方向：").pack(side=tk.LEFT)
        ttk.OptionMenu(row0, self.dir_var, "eng2rus", "eng2rus", "rus2eng").pack(side=tk.LEFT, padx=6)
        ttk.Button(row0, text="加载模型", command=self.on_load).pack(side=tk.LEFT, padx=8)

        ttk.Label(top, text="输入（自动按 .!? 分句）：").pack(anchor="w")
        self.txt_in = tk.Text(top, height=8, width=84); self.txt_in.pack(fill=tk.BOTH, expand=True, pady=4)

        row2 = ttk.Frame(top); row2.pack(fill=tk.X, pady=6)
        ttk.Button(row2, text="翻译", command=self.on_translate).pack(side=tk.LEFT)
        self.status = tk.StringVar(value="未加载模型")
        ttk.Label(row2, textvariable=self.status, foreground="gray").pack(side=tk.RIGHT)

        ttk.Label(top, text="输出：").pack(anchor="w")
        self.txt_out = tk.Text(top, height=8, width=84); self.txt_out.pack(fill=tk.BOTH, expand=True, pady=4)

    def on_load(self):
        try:
            self.agent = load_agent(self.dir_var.get())
            self.status.set(f"已加载：{self.dir_var.get()}")
            messagebox.showinfo("加载成功", "模型与 tokenizer 已就绪")
        except Exception as e:
            self.agent = None
            self.status.set("未加载模型")
            messagebox.showerror("加载失败", str(e))

    def on_translate(self):
        if self.agent is None:
            messagebox.showwarning("未加载模型", "请先加载模型")
            return
        text = self.txt_in.get("1.0", tk.END).strip()
        if not text:
            self.txt_out.delete("1.0", tk.END)
            return
        try:
            out = translate_paragraph(self.agent, text)
            self.txt_out.delete("1.0", tk.END)
            self.txt_out.insert(tk.END, out)
        except Exception as e:
            messagebox.showerror("翻译失败", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    try:
        root.call("source", "sun-valley.tcl")
        root.call("set_theme", "light")
    except Exception:
        pass
    TranslatorGUI(root)
    root.mainloop()

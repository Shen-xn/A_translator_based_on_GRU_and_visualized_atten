import unicodedata
import re
import random
import torch
import pickle
import os
from typing import List, Tuple, Dict
import torch
from torch.utils.data import TensorDataset, DataLoader

# All kinds of spaces...
spaces = [
    '\u00A0',  # 不间断空格 (Non-Breaking Space)
    '\xA0',    # 不间断空格 (Non-Breaking Space)
    '\u0009',  # 制表符 (Tab)
    '\u000A',  # 换行符 (Line Feed)
    '\u000D',  # 回车符 (Carriage Return)
    '\u200B',  # 零宽空格 (Zero Width Space)
    '\u2002',  # En Space
    '\u2003',  # Em Space
    '\u2009',  # Thin Space
    '\u3000',  # 全角空格 (Ideographic Space)
    '\u202F',  # 非断行空格 (Narrow Non-Breaking Space)
]

# Replace various forms of spaces with normal
def replace_narrow_nonbreaking_space(text):  # {str -> str}
    for space in spaces:
        text = text.replace(space, ' ')
    return text

# Convert Unicode strings to pure ASCII 
def unicodeToAscii(s):  # {str -> str}
    return ''.join(
        c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn'
    )
    
# Lowercase, and nomalize characters
def normalizeString(s):  # {str -> str}
    s = replace_narrow_nonbreaking_space(unicodeToAscii(s.lower().strip()))
    s = re.sub(r"[^a-zA-Z,;.!?а-яА-ЯёЁ\u4e00-\u9fff']+", r" ", s)
    s = re.sub(r"(?<!\s)([.!?])", r" \1", s)
    return s

# To remove data that is too long in the language pair
def filterPairs(pairs, MAX_LENGTH):
    filted = []
    for pair in pairs:
        if len(pair[0].split(' ')) <= MAX_LENGTH and len(pair[1].split(' ')) <= MAX_LENGTH:
            filted.append(pair)        
    return filted

# Construct a language_class based on data set
class Lang:  
    def __init__(self, name):
        self.name = name
        self.word2index = {"<SOS>": 0, "<EOS>": 1, "<PAD>": 2, "<UNK>": 3}
        self.word2count = {}
        self.index2word = {0: "<SOS>", 1: "<EOS>", 2: "<PAD>", 3: "<UNK>"}
        self.n_words = 4

    def addWord(self, word):  # Add a word
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def addSentence(self, sentence):  # Statistics words in a sentence
        for word in sentence.split(' '):
            self.addWord(word)
            
    def Sentence2Tensor(self, sentence, alignment=False, max_length=None, include_sos=True):
        words = sentence.split(' ')
        # Only convert words that exist in the vocabulary, otherwise they are considered pad
        indexes = []
        if include_sos:
            indexes.append(self.word2index["<SOS>"])
        for w in words:
            indexes.append(self.word2index.get(w, self.word2index["<UNK>"]))
        indexes.append(self.word2index["<EOS>"])
        
        if alignment:
            if max_length is None:
                raise ValueError("Must input maxlength:int, when using alignment")
            if len(indexes) < max_length:
                indexes += [self.word2index['<PAD>']] * (max_length - len(indexes))  # Pad to max_length
            else:
                indexes = indexes[:max_length]
                    
        seq_tensor = torch.tensor(indexes, dtype=int)
        return seq_tensor
    
    def Tensor2Sentence(self, tensor):
        words = []
        # 允许 list/ndarray
        if isinstance(tensor, (list, tuple)):
            it = tensor
        else:
            it = tensor.tolist()
        for idx in it:
            tok = self.index2word.get(int(idx), "<UNK>")
            if tok == "<EOS>":
                break
            if tok in ("<SOS>", "<PAD>"):
                continue
            words.append(tok)
        return " ".join(words)        
            
class dictionary:
    def __init__(self, name, lang1, lang2, save_dir="./Tokenizer"):
        self.name = name
        self.lang_class1 = Lang(lang1)
        self.lang_class2 = Lang(lang2)
        self.data_pairs = []
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
    def AddData(self, path, MAX_LENGTH, reverse=False):
        
        lines = open(path, encoding='utf-8').read().strip().split('\n')
        pairs = []
        print("Reading data file")
        for l in lines:
            sentences = l.split('\t')[0:2]
            if len(sentences) < 2:
                continue  
            pairs.append([normalizeString(s) for s in sentences])
            
        print(f"Got {len(pairs)} sentence pairs")
        pairs = filterPairs(pairs, MAX_LENGTH)
        print(f"Trimmed to {len(pairs)} sentence pairs")
        if reverse:
            pairs = [[p[1], p[0]] for p in pairs]
        
        
        print(" Counting words...")
        for pair in pairs:
            self.lang_class1.addSentence(pair[0])
            self.lang_class2.addSentence(pair[1])
        print(" Counted words:\n")
        sample = random.randint(0, len(pairs))
        print(f"Input language:{self.lang_class1.name}, {self.lang_class1.n_words} words, random sample:{pairs[sample][0]}")
        print(f"Input language:{self.lang_class2.name}, {self.lang_class2.n_words} words random sample:{pairs[sample][1]}")
        self.data_pairs.extend(pairs)
    
    def save_tokenizer(self):
        fp = os.path.join(self.save_dir, f"{self.name}.tokenizer.pkl")
        payload = {
            "name": self.name,
            "lang1_name": self.lang_class1.name,
            "lang2_name": self.lang_class2.name,
            # Save tables to recover Langs
            "lang1_word2index": self.lang_class1.word2index,
            "lang1_index2word": self.lang_class1.index2word,
            "lang2_word2index": self.lang_class2.word2index,
            "lang2_index2word": self.lang_class2.index2word,
        }
        with open(fp, "wb") as f:
            pickle.dump(payload, f)
        print(f"[saved tokenizer] {fp}")

    def load_tokenizer(self, filepath=None):
        if filepath is None:
            filepath = os.path.join(self.save_dir, f"{self.name}.tokenizer.pkl")
        with open(filepath, "rb") as f:
            payload = pickle.load(f)
        # 复原两个 Lang
        self.lang_class1 = Lang(payload["lang1_name"])
        self.lang_class1.word2index = payload["lang1_word2index"]
        self.lang_class1.index2word = payload["lang1_index2word"]
        self.lang_class1.n_words = max(self.lang_class1.index2word.keys()) + 1

        self.lang_class2 = Lang(payload["lang2_name"])
        self.lang_class2.word2index = payload["lang2_word2index"]
        self.lang_class2.index2word = payload["lang2_index2word"]
        self.lang_class2.n_words = max(self.lang_class2.index2word.keys()) + 1
        print(f"[loaded tokenizer] {filepath}")



# sentense_pairs -> tensor_pairs(tensor1-size(num_data, MAX_LENGTH), tensor1-size(num_data, MAX_LENGTH))
def Pair2Standardtensors(pairs, input_lang: Lang, output_lang: Lang, MAX_LENGTH: int, include_sos=True):
    N = len(pairs)
    pad_in = input_lang.word2index["<PAD>"]
    pad_out = output_lang.word2index["<PAD>"]
    input_tensors = torch.full((N, MAX_LENGTH), pad_in, dtype=torch.long)
    target_tensors = torch.full((N, MAX_LENGTH), pad_out, dtype=torch.long)

    for i, (src, tgt) in enumerate(pairs):
        input_tensors[i]  = input_lang.Sentence2Tensor(src,  alignment=True, max_length=MAX_LENGTH, include_sos=include_sos)
        target_tensors[i] = output_lang.Sentence2Tensor(tgt, alignment=True, max_length=MAX_LENGTH, include_sos=include_sos)

    return input_tensors, target_tensors


def _pair_len_no_specials(pair: Tuple[str, str]) -> int:
    # 句长（不含 SOS/EOS），就按空格分词后的 token 数
    s, t = pair
    return max(len(s.split(' ')) if s else 0,
               len(t.split(' ')) if t else 0)

def split_pairs_four_buckets(pairs: List[Tuple[str, str]], MAX_LENGTH_no_specials: int) -> Dict[str, List[Tuple[str, str]]]:
    # 四个阈值（不含 SOS/EOS）
    b1 = MAX_LENGTH_no_specials // 4
    b2 = MAX_LENGTH_no_specials // 2
    b3 = (3 * MAX_LENGTH_no_specials) // 4
    b4 = MAX_LENGTH_no_specials

    buckets = {"q1": [], "q2": [], "q3": [], "q4": []}
    for p in pairs:
        L = _pair_len_no_specials(p)
        if L <= b1:
            buckets["q1"].append(p)
        elif L <= b2:
            buckets["q2"].append(p)
        elif L <= b3:
            buckets["q3"].append(p)
        else:  # L <= b4（你的 AddData 已保证不超 MAX_LENGTH）
            buckets["q4"].append(p)
    return buckets


def build_bucket_dataloaders(
    pairs: List[Tuple[str, str]],
    input_lang,
    output_lang,
    MAX_LENGTH_no_specials: int,
    include_sos: bool = True,
    batch_sizes: Dict[str, int] = None,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False
) -> Dict[str, DataLoader]:
    buckets = split_pairs_four_buckets(pairs, MAX_LENGTH_no_specials)
    b1 = MAX_LENGTH_no_specials // 4
    b2 = MAX_LENGTH_no_specials // 2
    b3 = (3 * MAX_LENGTH_no_specials) // 4
    b4 = MAX_LENGTH_no_specials
    add_specials = 2 if include_sos else 1

    pad_lens = {
        "q1": max(1, b1 + add_specials),
        "q2": max(1, b2 + add_specials),
        "q3": max(1, b3 + add_specials),
        "q4": max(1, b4 + add_specials),
    }
    if batch_sizes is None:
        batch_sizes = {"q1":128, "q2":96, "q3":64, "q4":48}

    loaders = {}
    for key in ["q1","q2","q3","q4"]:
        ps = buckets[key]
        if len(ps) == 0: 
            continue
        pad_len = pad_lens[key]
        src_t, tgt_t = Pair2Standardtensors(ps, input_lang, output_lang, pad_len, include_sos=include_sos)
        ds = TensorDataset(src_t, tgt_t)
        loaders[key] = DataLoader(
            ds,
            batch_size=batch_sizes.get(key, 64),
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    return loaders


def train_eval_split(
    pairs: List[Tuple[str, str]],
    MAX_LENGTH_no_specials: int,
    eval_ratio: float = 0.10,
    seed: int = 42
):
    rng = random.Random(seed)
    buckets = split_pairs_four_buckets(pairs, MAX_LENGTH_no_specials)
    train_pairs, eval_pairs = [], []
    for key in ["q1", "q2", "q3", "q4"]:
        ps = buckets[key]
        rng.shuffle(ps)
        k = max(1, int(len(ps) * eval_ratio)) if len(ps) > 0 else 0
        eval_pairs.extend(ps[:k])
        train_pairs.extend(ps[k:])
    rng.shuffle(train_pairs)
    rng.shuffle(eval_pairs)
    return train_pairs, eval_pairs


def mixed_bucket_iterator(loaders: Dict[str, DataLoader], seed: int = None):
    """
    将 {'q1':dl1, 'q2':dl2, ...} 的批次按“随机顺序跨桶交替”输出。
    - 每个桶内部 DataLoader 已 shuffle=True；
    - 本函数仅打乱桶的出场顺序，每轮从可用桶中随机挑一个拿一批。
    """
    rng = random.Random(seed)
    # 为每个 loader 创建迭代器
    iters = {k: iter(dl) for k, dl in loaders.items()}
    alive = list(iters.keys())
    while alive:
        k = rng.choice(alive)
        try:
            batch = next(iters[k])
            yield batch  # (src, tgt)
        except StopIteration:
            alive.remove(k)
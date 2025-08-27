import time
import datetime
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
import torch.nn as nn
from torch import optim
from text_processing_lib.data_processing import *
from RNN_with_atten.NN import *
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt
from datasets import load_dataset

# Warmup + CosinAnnealing
def build_cosine_with_warmup(optimizer, total_steps, warmup_ratio=0.1, min_lr_scale=0.2):
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        # 把最低学习率控制在初始 lr 的 min_lr_scale
        return max(min_lr_scale, cosine)
    return LambdaLR(optimizer, lr_lambda)

# ====== 评估：遍历 eval_loaders，返回平均 loss ======
@torch.no_grad()
def evaluate_all(agent, eval_loaders, criterion, device):
    agent.encoder.eval()
    agent.decoder.eval()
    total_loss, total_tokens = 0.0, 0
    for _, loader in eval_loaders.items():
        for data in loader:
            input = data[0].to(device)
            tg = data[1].to(device)
            loss = agent.evaluate_batch(input, tg, criterion)
            eval_batch_size = input.size(0)
            total_loss += loss * eval_batch_size
            total_tokens += eval_batch_size
    if total_tokens == 0:
        return float("inf")
    return total_loss / total_tokens


def train_iters(
    agent,                 
    train_loaders,            
    eval_loaders,               
    model_save_dir,
    device="cuda",
    epochs=5,
    lr=3e-4,
    weight_decay=0.01,
    warmup_ratio=0.1,
    print_every=50,     
    val_every=500,  
):
    os.makedirs(model_save_dir, exist_ok=True)
    enc_opt = AdamW(agent.encoder.parameters(), lr=lr, weight_decay=weight_decay)
    dec_opt = AdamW(agent.decoder.parameters(), lr=lr, weight_decay=weight_decay)

    # 遍历一次 loader 统计 batch 数
    total_train_steps = 0
    for _, dl in train_loaders.items():
        total_train_steps += len(dl)

    total_train_steps *= epochs # maximum total train steps
    enc_sch = build_cosine_with_warmup(enc_opt, total_steps=total_train_steps, warmup_ratio=warmup_ratio)
    dec_sch = build_cosine_with_warmup(dec_opt, total_steps=total_train_steps, warmup_ratio=warmup_ratio)

    pad_tgt = 2
    criterion = nn.CrossEntropyLoss(ignore_index=pad_tgt)
    """
    正确类的概率 = 1 - ε（例如 0.9,当 ε=0.1 时）
    其他类的概率 = ε / (num_classes - 1)（平均分配)
    """

    # ==== 日志与计时 ====
    run_dir = model_save_dir
    print(f"[Train] logs & checkpoints -> {run_dir}")

    history = {
        "iter": [],
        "train_loss": [],
        "val_iter": [],
        "val_loss": []
    }

    start_time = time.time()
    global_step = 0
    best_val = float("inf")

    # ==== 训练 ====
    for ep in range(1, epochs + 1):
        agent.encoder.train()
        agent.decoder.train()

        # 每个 epoch 重置 decoder 的状态（decoder 自己每步会 reset_state）
        running_loss = 0.0
        iters_in_epoch = 0
        epoch_start = time.time()

        # ——混合跨桶迭代：把四个 DataLoader 的 batch 随机交织
        for src_batch, tgt_batch in mixed_bucket_iterator(train_loaders):
            
            # Transfer batch_data to cuda
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)
            loss = agent.train_on_batch(src_batch, tgt_batch, enc_opt, dec_opt, criterion, teacher_forcing_ratio=0.5)

            # === scheduler 跟进一步 ===
            enc_sch.step()
            dec_sch.step()

            global_step += 1
            iters_in_epoch += 1
            running_loss += loss

            # ——日志：平均损失、耗时、ETA——
            if global_step % print_every == 0:
                elapsed = time.time() - start_time
                avg_loss = running_loss / iters_in_epoch
                # ETA
                speed = global_step / max(1.0, elapsed)  # iters/sec
                remain_steps = max(0, total_train_steps - global_step)
                eta_sec = remain_steps / max(1e-6, speed)
                h, m = divmod(int(eta_sec), 3600)
                m, s = divmod(m, 60)
                lr_enc = enc_opt.param_groups[0]["lr"]
                lr_dec = dec_opt.param_groups[0]["lr"]
                print(f"[ep {ep}/{epochs}] iter {global_step}/{total_train_steps} | "
                        f"train_loss(avg in epoch)={avg_loss:.4f} | "
                        f"lr(enc/dec)={lr_enc:.2e}/{lr_dec:.2e} | "
                        f"elapsed={int(elapsed)}s | ETA={h:02d}:{m:02d}:{s:02d}")
                history["iter"].append(global_step)
                history["train_loss"].append(avg_loss)

            # ——评估 & 保存——
            if global_step % val_every == 0:
                val_loss = evaluate_all(agent, eval_loaders, criterion, device)
                print(f"  [validate] iter {global_step} | val_loss={val_loss:.4f}")
                history["val_iter"].append(global_step)
                history["val_loss"].append(val_loss)

                # 保存最好模型
                if val_loss < best_val:
                    best_val = val_loss
                    ckpt_dir = os.path.join(run_dir, f"best_model")
                    os.makedirs(ckpt_dir, exist_ok=True)
                    agent.save(ckpt_dir, extra={"best_iter": global_step, "best_val": best_val})
                    print(f"  [save] best model -> {ckpt_dir}")

        # ——epoch 结束：本 epoch 汇总 & 也做一次 eval——
        avg_loss_ep = running_loss / max(1, iters_in_epoch)
        val_loss_ep = evaluate_all(agent, eval_loaders, criterion, device)
        history["iter"].append(global_step)
        history["train_loss"].append(avg_loss_ep)
        history["val_iter"].append(global_step)
        history["val_loss"].append(val_loss_ep)

        ep_time = time.time() - epoch_start
        print(f"[epoch {ep}] avg_train_loss={avg_loss_ep:.4f} | val_loss={val_loss_ep:.4f} | epoch_time={int(ep_time)}s")

        # 保存该 epoch 的 checkpoint
        ckpt_dir = os.path.join(run_dir, f"epoch_{ep:03d}")
        os.makedirs(ckpt_dir, exist_ok=True)
        agent.save(ckpt_dir, extra={"epoch": ep, "val_loss": val_loss_ep})

    # ==== 画图并保存 ====
    fig_path = os.path.join(run_dir, "loss_curve.png")
    plt.figure()
    if len(history["iter"]):
        plt.plot(history["iter"], history["train_loss"], label="train_loss")
    if len(history["val_iter"]):
        plt.plot(history["val_iter"], history["val_loss"], label="val_loss")
    plt.xlabel("iterations")
    plt.ylabel("loss")
    plt.title("Train/Val Loss vs Iterations")
    plt.legend()
    plt.grid(True)
    plt.savefig(fig_path, dpi=150)
    print(f"[figure saved] {fig_path}")

    # 保存训练日志
    with open(os.path.join(run_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print("[done] best_val=%.4f | run_dir=%s" % (best_val, run_dir))
    return history, run_dir


# Set device to CUDA if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Use calculate device: ", device, "\n")
MAX_LENGTH = 20 # This is maxlength of sentence! NOT INCLUDING EOS SOS...
print(f"Training data limit length (including punctuation marks): {MAX_LENGTH}")

data_path = "./dataset/rus.txt"
eval_ratio = 0.15
hidden_size = 1024
batch_size = 512
model_save_dir = "./temp_models"
data = load_dataset("wmt14", "ru-en")

if __name__ == "__main__":
    time_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(model_save_dir, time_stamp)
    
    
    dict_obj = dictionary(name="en_rus", lang1="en", lang2="rus", save_dir=save_dir)
    
    
    dict_obj.AddData(data, MAX_LENGTH=MAX_LENGTH, reverse=False)
    dict_obj.save_tokenizer()
    
    
    # 这里的数据和Lang对象在训练正反翻译器的时候是可复用的，我们只实现一次dictionary类
    language1 = dict_obj.lang_class1
    language2 = dict_obj.lang_class2
    vocab_size1 = language1.n_words
    vocab_size2 = language2.n_words
    
    train_pairs, eval_pairs = train_eval_split(dict_obj.data_pairs, MAX_LENGTH_no_specials=MAX_LENGTH, eval_ratio=eval_ratio)
    
    # Rlease memory
    del dict_obj
    import gc
    gc.collect()
    
    
    agent_f = translator(
            src_vocab_size=vocab_size1,
            tgt_vocab_size=vocab_size2,
            hidden_size=hidden_size,
            device="cpu",
            num_encoder_layers=2, 
            num_decoder_layers=1,
            dropout_rate= 0.1,
            sos_idx=language1.word2index["<SOS>"],
            eos_idx=language2.word2index["<EOS>"],
            padding_idx=language2.word2index["<PAD>"])
    agent_f.to(device)
    
    # agent_b = translator(
    #             src_vocab_size=vocab_size2,
    #             tgt_vocab_size=vocab_size1,
    #             hidden_size=hidden_size,
    #             device="cuda",
    #             num_encoder_layers=2, 
    #             num_decoder_layers=1,
    #             dropout_rate= 0.1,
    #             sos_idx=language2.word2index["SOS"],
    #             eos_idx=language1.word2index["EOS"])
    

    
    
    
    # 正向 loaders（lang1 -> lang2）# Attention! Here load data on cpu to save cuda memory
    train_loaders_f = build_bucket_dataloaders(
        train_pairs, language1, language2, MAX_LENGTH_no_specials=MAX_LENGTH,
        include_sos=False, batch_sizes={"q1":batch_size//4,"q2":batch_size//4,"q3":batch_size//4,"q4":batch_size//4},
        shuffle=True, pin_memory=(device.type=="cpu")
    )
    eval_loaders_f = build_bucket_dataloaders(
        eval_pairs, language1, language2, MAX_LENGTH_no_specials=MAX_LENGTH,
        include_sos=False, batch_sizes={"q1":batch_size//4,"q2":batch_size//4,"q3":batch_size//4,"q4":batch_size//4},  # eval可更大
        shuffle=False, pin_memory=(device.type=="cpu")
    )

    # # 3) 反向 loaders（lang2 -> lang1），同一套字典可直接用
    # train_loaders_b = build_bucket_dataloaders(
    #     [(t,s) for (s,t) in train_pairs], language2, language1, MAX_LENGTH_no_specials=MAX_LENGTH,
    #     include_sos=False, batch_sizes={"q1":batch_size//4,"q2":batch_size//4,"q3":batch_size//4,"q4":batch_size//4},
    #     shuffle=True, pin_memory=(device.type=="cuda")
    # )
    # eval_loaders_b = build_bucket_dataloaders(
    #     [(t,s) for (s,t) in eval_pairs], language2, language1, MAX_LENGTH_no_specials=MAX_LENGTH,
    #     include_sos=False, batch_sizes={"q1":batch_size//4,"q2":batch_size//4,"q3":batch_size//4,"q4":batch_size//4},
    #     shuffle=False, pin_memory=(device.type=="cuda")
    # )
                     
   
    history, run_dir = train_iters(
        agent=agent_f,
        train_loaders=train_loaders_f,
        eval_loaders=eval_loaders_f,
        # upper three need change direction!!
        device=device,
        epochs=8,
        lr=3e-4,
        weight_decay=0.01,
        warmup_ratio=0.1,
        print_every=100,
        val_every=1000,
        model_save_dir=save_dir
    )
    
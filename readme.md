# TRANSLATER_PROGRAM — README

A lightweight Seq2Seq translator (GRU + Attention) built with **PyTorch**.  
The project includes: data preprocessing, bucketed DataLoader, training script with warmup + cosine scheduler, inference interface, and a **tkinter GUI**.  
All core implementations are self-contained in this repository, without external complex dependencies.

---

## Directory Structure

- **applied_models/**  
  Pretrained or deployed models (`encoder.pt`, `decoder.pt`, `config.json`, `tokenizer.*`).

- **dataset/**  
  Example dataset (e.g., `rus.txt`, tab-separated sentence pairs).

- **RNN_with_atten/NN.py**  
  Model definitions and agent class: `EncoderRNN`, `AttnDecoderRNN`, `translator`.

- **text_processing_lib/data_processing.py**  
  Data preprocessing, normalization, vocabulary building, bucketing and DataLoader.

- **temp_models/**  
  Checkpoints and logs generated during training.

- **APP.py**  
  GUI program (tkinter-based translator).

- **model_train.py**  
  Training entry, including `train_iters`, evaluation, logging, and saving models.

- **requirements.txt**  
  Project dependencies.

- **test.py**  
  Optional test script.

---

## Quickstart

### 1. Environment

We recommend using a virtual environment (conda or venv):

```bash
pip install -r requirements.txt
```

Main dependencies:
- `torch`, `torchvision`, `torchaudio`
- `numpy`, `matplotlib`
- `scikit-learn`, `tqdm`
- `tkinter` (Python built-in, may require `python3-tk` on Linux)

---

### 2. Prepare Data

Place your bilingual dataset (e.g., `rus.txt`) under `dataset/`.  
Each line should contain two tab-separated sentences (source and target).

The script will:
- Normalize sentences (lowercase, remove special symbols, standardize spaces).
- Filter sentence pairs exceeding the maximum length.
- Build vocabularies with `<SOS>`, `<EOS>`, `<PAD>`, `<UNK>` tokens.

---

### 3. Train (Example: English → Russian)

Run:

```bash
python model_train.py
```

The script will:
- Train with bucketed DataLoaders (4 buckets by length).
- Use **AdamW + cosine scheduler with warmup**.
- Save checkpoints/logs to `./temp_models/<timestamp>/`.

In each run directory:
- `epoch_XXX/`: checkpoints after each epoch.
- `best_XXXXXX/`: best checkpoint by validation loss.
- `loss_curve.png`: loss vs. iterations.
- `history.json`: training/validation logs.

---

## Models

- **EncoderRNN**  
  - Embedding → multi-layer `Norm_GRU` (GRU + LayerNorm) → Linear projection.  
  - Input `(B, T)` → Output `(T, B, H)`.

- **Attention**  
  - Scaled dot-product attention.  
  - Supports self-attention and query-seed attention.

- **AttnDecoderRNN**  
  - At each decoding step:  
    1. Attention over historical + current embeddings (self-attention).  
    2. Attention over encoder outputs (context).  
  - Combine embedding + context → GRU → dropout → linear → log-softmax.

- **translator class**  
  - Wraps encoder/decoder.  
  - Provides training (`train_on_batch`), evaluation (`evaluate`), greedy inference (`translate_sentence`).  
  - Supports truncation by `max_input_len` and `max_output_len`.  
  - Save/load checkpoints with config and tokenizer.

---

## GUI

Run the GUI app:

```bash
python APP.py
```

Features:
- Select translation direction (`eng→rus` / `rus→eng`).
- Models are loaded automatically from `applied_models/`.
- Input box supports splitting text by `.?!` into sentences.
- Each sentence translated separately, then concatenated.
- Postprocessing: capitalize the first letter, append `.` if missing.

---

## Notes

- **Bucketing**: 4 buckets (`¼`, `½`, `¾`, full length) reduce padding waste.  
- **Loss**: `log_softmax + NLLLoss(ignore_index=<PAD>)`.  
- **Evaluation**: teacher-forcing loss per token for stable validation results.  
- **Inference**: greedy decoding until `<EOS>` or max length.

---

## FAQ

**Q: How to generate `requirements.txt`?**  
- `pip freeze > requirements.txt` (full list).  
- Or `pip install pipreqs && pipreqs . --force` (minimal list from imports).  

**Q: How to check CUDA availability?**  
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

**Q: How to deploy models?**  
Copy the best checkpoint into `applied_models/<direction>/`, ensuring it contains:
```
encoder.pt
decoder.pt
config.json
tokenizer.pkl
```

---

## Acknowledgements

The project is based on self-implemented modules:
- `RNN_with_atten/NN.py`
- `text_processing_lib/data_processing.py`
- `model_train.py`
- `APP.py`
#  Transformer DE→EN

Implementation of a Transformer-based model for German → English translation,
based on the seminal paper:

> Vaswani et al. (2017) - *Attention is All You Need*
> [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

This project covers the following main steps:

* Data preparation (WMT17) → `src/data/prepare_data.py`
* Tokenizer training using Byte-Pair Encoding (BPE) → `src/data/train_tokenizer.py`
* Transformer model training and evaluation → `src/train.py`
* Model testing → `src/test.py`

Disclaimer : This project is for educational purposes, so the model was trained on my own machine on one CPU, which is why the model is not very accurate.

---

## **Project structure**

```
Project_transformer_translation/
│
├── checkpoints/                # Saved models
│   ├── best_bleu.pt            # Model with the best BLEU score
│   ├── best_val_loss.pt        # Model with the best validation loss
│   └── last.pt                 # Last trained model
│
├── configs/
│   ├── config.yaml             # Main configuration file (OmegaConf)
│   └── tiny.yaml               # Lightweight model variant
│
├── data/
│   ├── raw/                    # WMT17 data in JSONL format
│   │   ├── train.jsonl
│   │   ├── valid.jsonl
│   │   └── test.jsonl
│   └── tokenizer/              # Trained tokenizer vocabulary
│       ├── bpe.json
│       └── specials.json
│
├── src/
│   ├── data/
│   │   ├── dataset.py          # JSONL dataset class + padding
│   │   ├── prepare_data.py     # Downloads and prepares datasets
│   │   └── train_tokenizer.py  # Trains the BPE tokenizer
│   ├── model/
│   │   ├── model.py            # Transformer architecture + positional encoding
│   │   ├── schedulers.py       # NoamLR learning rate scheduler
│   │   └── earlyStopping.py    # Early stopping mechanism
│   ├── utils/
│   │   ├── io.py               # Model save/load utilities
│   │   ├── text.py             # Detokenization utilities
│   │   └── __init__.py
│   ├── train.py                # Full model training script
│   └── test.py                 # Evaluation on the test set
│
├── requirements.txt
└── README.md
```

---

## **Installation**

### Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate  # on Windows
# or
source venv/bin/activate  # on Linux / macOS
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Main dependencies

| Library      | Purpose                                     |
| ------------ | ------------------------------------------- |
| `torch`      | Transformer model training                  |
| `tokenizers` | BPE tokenizer training                      |
| `datasets`   | Loading the WMT17 dataset                   |
| `evaluate`   | BLEU score computation                      |
| `omegaconf`  | Configuration and hyperparameter management |
| `tqdm`       | Progress bars for loops                     |

---

## **Data preparation**

Download and prepare the WMT17 German → English subsets:

```bash
python src/data/prepare_data.py
```

This script:

* Loads the `wmt17` dataset using Hugging Face `datasets`
* Creates the files `train.jsonl`, `valid.jsonl`, and `test.jsonl` under `data/raw/`

---

## **Tokenizer training (BPE)**

Train a Byte-Pair Encoding tokenizer on the corpus:

```bash
python src/data/train_tokenizer.py
```

This generates:

* `data/tokenizer/bpe.json`
* `data/tokenizer/specials.json`

---

## **Model training**

Train the Transformer model:

```bash
python src/train.py
```

* The NoamLR scheduler automatically adapts the learning rate
* Training stops automatically when `epoch_max` is reached or validation loss stops improving (early stopping)
* Checkpoints are saved in `checkpoints/`:

  * `last.pt`
  * `best_val_loss.pt`
  * `best_bleu.pt`

---

## **Evaluation**

Evaluate the model on the test set:

```bash
python src/test.py
```

Outputs:

* Display of loss and BLEU score
* Example detokenizations
* Loss curves plotted using `matplotlib`

---

## **Resume training**

If training is interrupted, it automatically resumes from:

```
checkpoints/last.pt
```

---

## **Configuration (OmegaConf)**

All hyperparameters, paths, and settings are centralized in `configs/config.yaml`.

You can override any value directly from the command line, for example:

```bash
python src/train.py model.d_model=512 train.epochs_max=5 data.batch_size=128
```

---

## **References**

* Vaswani et al. (2017) - [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762)
* PyTorch - [https://pytorch.org/](https://pytorch.org/)
* Hugging Face Tokenizers - [https://github.com/huggingface/tokenizers](https://github.com/huggingface/tokenizers)
* SacreBLEU - [https://github.com/mjpost/sacrebleu](https://github.com/mjpost/sacrebleu)
* OmegaConf - [https://omegaconf.readthedocs.io](https://omegaconf.readthedocs.io)

---

## **Author**

Project developed as part of an application.
Created and maintained by Thibault PAWLISZ, 2025.


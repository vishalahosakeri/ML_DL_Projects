# ML_DL_e2e

End-to-end PyTorch projects covering the core supervised learning + deep learning task types: regression, binary classification, multiclass classification, image classification (CNN), and sequence classification (RNN/LSTM/GRU).

## Projects

| File | Task | Highlights |
|---|---|---|
| `linearRegression_pytorch.py` | Linear regression | PyTorch training loop fundamentals — model, loss, optimizer, backprop |
| `binaryClassification_pytorch.py` | Binary classification | Non-linear decision boundary on `make_circles` data; `nn.Sequential` MLP with ReLU; feature scaling with `StandardScaler` |
| `multiclassification_pytorch.py` | Multiclass classification | 4-class blob classification (`make_blobs`); softmax output, `long`-typed labels for `CrossEntropyLoss` |
| `cnn_imageClassification.py` | Image classification | FashionMNIST; MLP baseline vs. CNN comparison, `DataLoader` batching, accuracy/confusion matrix via `torchmetrics` + `mlxtend` |
| `RNNLLSTM_GRU_SMSSpam_Classification.py` | Sequence / text classification | BiLSTM SMS spam classifier — custom vocab (train-only), `pack_padded_sequence` for variable-length sequences, stratified split for class imbalance. **F1 ≈ 0.927** |

## Getting started

```bash
pip install torch torchvision scikit-learn pandas numpy matplotlib torchmetrics mlxtend nltk
python linearRegression_pytorch.py
```

Each script is self-contained and runs end-to-end (data → model → training → evaluation) — no shared setup required beyond the dependencies above. `cnn_imageClassification.py` will download FashionMNIST automatically on first run.

## Notes

- `RNNLLSTM_GRU_SMSSpam_Classification.py` currently reads its input file from a hardcoded local path — update `file_path` to point at `../NLP/SMSSpamCollection.txt` (relative to this folder) before running it elsewhere.
- Progression across files: linear regression → binary → multiclass → CNN → RNN/LSTM/GRU, roughly in order of increasing model and data complexity.
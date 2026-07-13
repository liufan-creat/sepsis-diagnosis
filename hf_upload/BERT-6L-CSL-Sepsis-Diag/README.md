# BERT-6L-CSL

Sepsis in-hospital mortality prediction from ICD diagnosis text.

- **Mode**: diag
- **Architecture**: BERT-style Transformer (encoder_num=1, 6 layers per branch)
- **Total parameters**: 4,206,978
- **Vocabulary size**: 4805
- **Hidden size**: 128
- **Num attention heads**: 4
- **Feed-forward dim**: 2048
- **Max sequence length**: 30

## Loading the model

```python
import torch
from src.bert_classifier import BertClassifier

# Load checkpoint
ckpt = torch.load('pytorch_model.bin', map_location='cpu')
model = BertClassifier(
    vocab_size=4805,
    encoder_num=1,
)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# Inference
input_ids = tokenizer(text, return_tensors='pt')['input_ids']
logits    = model(input_ids)
pred      = torch.argmax(logits, dim=-1).item()
```

---

Built with [sepsis-diagnosis](https://github.com/liufan/sepsis-diagnosis).

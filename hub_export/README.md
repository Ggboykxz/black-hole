---
library_name: blackhole
tags: [llm, black-hole, from-scratch, gpt, rotary, gqa]
---

# Black Hole (tiny)

Decoder-only Transformer LM implemented from scratch (PyTorch).

| | |
|---|---|
| Layers | 4 |
| Dim | 192 |
| Heads | 6 (KV 2) |
| Context | 256 |
| Vocab | 516 |
| Params | 1.67M |

Load with the `blackhole` package:
```python
from blackhole.model import BlackHole
from blackhole.config import BlackHoleConfig
import json, torch
cfg = BlackHoleConfig.from_dict(json.load(open("config.json")))
model = BlackHole(cfg)
from safetensors.torch import load_file
sd = load_file("model.safetensors")
model.load_state_dict(sd, strict=False)  # lm_head tied to tok_emb -> omitted on purpose
```

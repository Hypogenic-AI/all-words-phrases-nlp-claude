# Datasets

This directory contains datasets for the implicit vocabulary research project. Data files are symlinked from the footprints code repository.

## Dataset 1: COUNTERFACT (Expanded)

### Overview
- **Source**: Originally from [ROME](https://rome.baulab.info/) (Meng et al., 2022), expanded by Feucht et al.
- **Size**: ~12K prompts (6.7MB CSV)
- **Format**: CSV with columns including prompt text and subject entities
- **Task**: Factual knowledge recall; used to test erasure on entity last tokens
- **License**: MIT (inherited from ROME project)

### Download Instructions
Already available in `code/footprints/data/counterfact_expanded.csv`.

Alternatively:
```bash
git clone https://github.com/sfeucht/footprints.git
# Data is in footprints/data/counterfact_expanded.csv
```

### Loading
```python
import pandas as pd
df = pd.read_csv("code/footprints/data/counterfact_expanded.csv")
```

## Dataset 2: The Pile (Subsets)

### Overview
- **Source**: [The Pile](https://pile.eleuther.ai/) (Gao et al., 2020)
- **Size**: train=1000 docs, val=500, test=500 (small subsets)
- **Format**: CSV with 'text' column
- **Task**: Probe training and vocabulary readout evaluation
- **License**: Various (see Pile documentation)

### Download Instructions
Already available in `code/footprints/data/`:
- `train_tiny_1000.csv` — probe training data
- `val_tiny_500.csv` — probe validation data
- `test_tiny_500.csv` — probe testing data

### Loading
```python
import pandas as pd
df = pd.read_csv("code/footprints/data/train_tiny_1000.csv")
```

## Dataset 3: Wikipedia (Subsets)

### Overview
- **Source**: [Wikipedia dump 20220301.en](https://huggingface.co/datasets/legacy-datasets/wikipedia)
- **Size**: 500-1000 articles per split
- **Format**: CSV with 'text' column
- **Task**: Multi-token word and entity evaluation
- **License**: CC BY-SA 3.0

### Download Instructions
Already available in `code/footprints/data/`:
- `wikipedia_test_500.csv` — primary evaluation set (used in paper)
- `wikipedia_train_1000.csv` — additional training data
- `wikipedia_val_500.csv` — validation set

### Loading
```python
import pandas as pd
df = pd.read_csv("code/footprints/data/wikipedia_test_500.csv")
```

## Dataset 4: Pre-trained Linear Probes

### Overview
- **Source**: HuggingFace (sfeucht/footprints)
- **Size**: 34 probe checkpoints per model (layers 0-32, offsets -3 to +1)
- **Format**: PyTorch checkpoint files (.ckpt)
- **Task**: Token prediction from hidden states (core to erasure score computation)

### Download Instructions
```python
from huggingface_hub import hf_hub_download

# Download a specific probe (e.g., layer 0, predict 3 tokens ago)
checkpoint_path = hf_hub_download(
    repo_id="sfeucht/footprints",
    filename="llama-2-7b/layer0_tgtidx-3.ckpt"
)
```

Available models: `llama-2-7b/`, `llama-3-8b/`
Available files per model: `layer{0-32}_tgtidx{-3,-2,-1,0,1}.ckpt`

### Notes
- Probes are linear models: input_size=4096, output_size=32000 (Llama-2) or 128256 (Llama-3)
- Training took 6-8 hours per probe on RTX-A6000
- Using pre-trained probes saves significant compute

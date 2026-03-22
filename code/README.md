# Cloned Repositories

## Repo 1: footprints
- **URL**: https://github.com/sfeucht/footprints
- **Purpose**: Official implementation of "Token Erasure as a Footprint of Implicit Vocabulary Items in LLMs" (Feucht et al., 2024)
- **Location**: `code/footprints/`
- **Key files**:
  - `scripts/segment.py` — Document segmentation using erasure scores (Algorithm 1)
  - `scripts/readout.py` — Read out implicit vocabulary from a dataset
  - `scripts/train_probe.py` — Train linear probes on hidden states
  - `scripts/test_probe.py` — Test probes on various datasets
  - `scripts/utils.py` — Utility functions including erasure score computation
  - `data/` — All datasets used in the paper (COUNTERFACT, Pile subsets, Wikipedia subsets)
- **Dependencies**: PyTorch, transformers, nnsight, spacy, pandas
- **Requirements**: GPU with ~16GB VRAM for Llama-2-7b inference
- **Notes**:
  - Pre-trained probe checkpoints available at HuggingFace `sfeucht/footprints`
  - Segment.py can be run on any text document with `--model meta-llama/Llama-2-7b-hf`
  - The readout.py script replicates the implicit vocabulary tables from the paper
  - Uses nnsight library for efficient access to model hidden states

## Additional Resources (Not Cloned)

### nnsight
- **URL**: https://github.com/ndif-team/nnsight
- **Purpose**: Library for accessing and intervening on neural network internals
- **Notes**: Install via `pip install nnsight`. Used by footprints for extracting hidden states.

### lm-compositionality
- **URL**: https://github.com/nightingal3/lm-compositionality
- **Purpose**: Code for "Are Representations Built from the Ground Up?" (Liu & Neubig, 2022)
- **Notes**: Contains CHIP dataset and composition probes. Could be used for complementary compositionality analysis.

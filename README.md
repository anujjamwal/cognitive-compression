# Cognitive Compression: Hierarchical Chain-of-Thought for Efficient LLM Reasoning

Chain-of-Thought (CoT) reasoning enables Large Language Models to solve com-
plex problems through step-by-step decomposition, but incurs linear growth in
context length and KV-cache memory. We present Cognitive Compression, a
method that restructures flat CoT into a hierarchical tree of subproblems using
three special tokens—[THOUGHT], [SOLUTION], and [RETURN]—and prunes com-
pleted reasoning branches during autoregressive generation. When a subproblem
is solved, the verbose intermediate tokens between [THOUGHT] and [SOLUTION]
are physically removed from the sequence, retaining only the concise solution
summary. We fine-tune Nemotron-1.5B on a dataset of ∼1,500 hierarchically
annotated mathematical reasoning traces, and implement a custom generation loop
with KV-cache–aware pruning. Our approach reduces token consumption while
preserving the logical state required for multi-step reasoning.

## Setup

Setting up the environment

```
conda env create -f env.yaml
```

```
conda activate cs224n
```
## Training Details

- **Base Model:** nvidia/OpenMath-Nemotron-1.5B (Qwen2.5-Math-1.5B architecture)
- **Training Method:** Prune-Aware SFT — training simulates inference by splitting examples at `[RETURN]` tokens and running separate forward passes per stage
- **Dataset:** [OpenMathReasoning-Sampled-Hierarchical-Cot-cleaned](https://huggingface.co/datasets/anujjamwal/OpenMathReasoning-Sampled-Hierarchical-Cot-cleaned), derived from NVIDIA's OpenMathReasoning with hierarchical annotations
- **Special Token Initialization:** Embeddings seeded from semantically similar phrases for stable fine-tuning

## Intended Use

Mathematical reasoning tasks where inference efficiency matters — the model produces structured reasoning that can be pruned in real-time, reducing token overhead without sacrificing answer quality.

## How to Use

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "anujjamwal/OpenMath-Nemotron-1.5B-PruneAware", 
    attn_implementation="flash_attention_2",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("anujjamwal/OpenMath-Nemotron-1.5B-PruneAware")

gen_out = model.generate(**inputs, tokenizer=tokenizer, max_new_tokens=2048)

# Use with the custom HCoT generation loop for pruning support
# See: https://github.com/anujjamwal/cognitive-compression
```


## HuggingFace

Dataset: anujjamwal/OpenMathReasoning-Sampled-Hierarchical-Cot
Model: anujjamwal/OpenMath-Nemotron-1.5B-PruneAware


## Citation

If you find this work useful, please cite:

```bibtex
@misc{jamwal2026cognitivecompression,
  title        = {Cognitive Compression: Hierarchical Chain-of-Thought for Efficient LLM Reasoning},
  author       = {Jamwal, Anuj},
  year         = {2026},
  howpublished = {\url{https://github.com/anujjamwal/cognitive-compression}},
  note         = {CS224N Winter 2026 Final Project, Stanford University}
}
```
# Context compression on the Google DeepMind Gemma repo (JAX)

## Context

We want to implement hierarchical-CoT context compression for Gemma 4
E4B by building directly against Google DeepMind's **gemma** repo
(https://github.com/google-deepmind/gemma) — a JAX/Flax library focused
exclusively on Gemma.

Context compression: during generation, the model wraps sub-reasoning
inside `<|channel>...<channel|>summary<return|>` blocks.  When the
model emits `<return|>`, the KV cache is *pruned* back to the position
of the matching `<|channel>`, so the detail inside the channel no
longer consumes attention budget.  Only the summary text survives
into the parent reasoning stream.  The model must be trained
(SFT + RL) to use this structure productively.

The gemma repo was chosen because it gives us primitives that make
the pruner dramatically simpler than in the HF transformers path we
had been prototyping against: one uniform cache shape across all
layers, an explicit `end_index` field on the cache (so "prune" = an
integer rewind, not a tensor re-slice), no `strip_thinking` chat-
template hazard, built-in LoRA + loss-mask + FSDP via Kauldron.  The
downside we accept: no built-in GRPO, and deployment is JAX-only
unless we manually export to HF.

This plan captures the findings from the repo, the key pruning
insight, and a four-phase implementation path from a single-prompt
parity check through public benchmarks.

---

## Findings from the Gemma repo

### What's genuinely simpler

1. **One cache shape, not a hybrid zoo.**  `LayerCache` is a plain
   `dict[str, jax.Array]` with fields `k`, `v`, `end_index`,
   `positions` (`/gemma/gm/nn/_modules.py:34,326–335`).  One dict per
   layer; both sliding and global layers use the same schema.
2. **`end_index` is an explicit field of the cache.**  Writes happen
   via `jax.lax.dynamic_update_slice(..., end_index % cache_size)`
   (`/gemma/gm/nn/_modules.py:214–234`).  `end_index` is the only
   stateful counter, and it lives in the cache dict.
3. **Attention pattern is a function, not a list.**
   `make_attention_layers_types()` in `/gemma/gm/nn/gemma4/_config.py`
   generates a repeating `(LOCAL_SLIDING, ..., GLOBAL)` pattern.  Only
   two enum values: `LOCAL_SLIDING` and `GLOBAL`.
4. **Single RoPE, two base frequencies.**  `apply_rope()` in
   `/gemma/gm/math/_positional_embeddings.py` takes `base_frequency`;
   the model passes `rope_base_frequency_local` or
   `rope_base_frequency_global` depending on attention type.  No
   per-layer-type `RotaryEmbedding` instances.
5. **LoRA + loss masking + sharding are built in.**  `gm.nn.LoRA`
   wraps any dense / einsum, `SoftmaxCrossEntropyWithIntLabels` takes
   a `mask` directly, Kauldron drives FSDP via `kd.sharding`.  No
   third-party PEFT/TRL gymnastics.
6. **Chat templating is explicit.**  No `strip_thinking` macro; no
   implicit history filtering.  `dialog.Conversation` objects are
   rendered with visible `<start_of_turn>` markers.

### What's different (not necessarily harder)

1. **JAX / Flax, not PyTorch.**  Functional state: the cache is passed
   in and returned; no in-place mutation.  For pruning this is
   actually cleaner — we return a new state with a rewound
   `end_index`.
2. **Tokenizer enum is minimal.**  `_Gemma4SpecialTokens` at
   `/gemma/gm/text/_tokenizer.py:136–158` enumerates only `PAD, EOS,
   BOS, UNK, MASK, START_OF_TURN=105, END_OF_TURN=106` and the
   multimodal tokens.  **`<|channel>`, `<channel|>`, `<|think|>` are
   NOT listed.**  However, the underlying SentencePiece binary is the
   same 262K-token Gemma 4 vocab that HF ships, so those tokens exist
   in the vocab and are addressable via `tokenizer.encode("<|channel>")`
   — they just aren't enumerated in the `SpecialTokens` enum.  We
   confirm their IDs at runtime.
3. **`custom_tokens: dict[int, str]` is supported** on the tokenizer
   (`_tokenizer.py:180–200, 322–334`).  Lets us register `<return|>`
   into an unused slot (0–98) without touching the SP binary.  This
   is the cleanest path for our new boundary token.

### What's harder

1. **The main decode loop is `jax.lax.while_loop` JIT-compiled** —
   shapes must be static, dynamic truncation of the cache is illegal
   inside it.  File: `/gemma/gm/text/_sampler_loop.py`, method
   `SamplerLoop._sample_loop()`.
2. **A Python streaming path exists** — `_stream_sample_loop()` yields
   `SamplingState` between steps, is not JIT-compiled as an outer
   loop (each `_sample_step` is still JIT-compiled individually), and
   is the hook we use for pruning.  Latency per step comparable; lose
   the outer-loop fusion win.
3. **No GRPO scaffolding.**  Kauldron has SFT, LoRA, DPO/NPO
   (`/examples/dpo.py`, `/examples/npo.py`) but not GRPO.  Either we
   port our HF GRPO worker's reward shaping into JAX, or do SFT in
   JAX and switch to PyTorch/TRL for RL (requires weight conversion).
4. **Deployment = JAX-only** unless we export.  No provided
   converter to HF checkpoints; vLLM / llama.cpp / TensorRT-LLM
   inference would require manual weight mapping.

---

## The key insight for pruning: rewind `end_index`, don't resize buffers

The cache is a **fixed-size ring buffer** whose current extent is
tracked by an explicit `end_index` field — not a variable-length
tensor that has to be re-sliced.  Pruning therefore reduces to
decrementing that integer.  To prune back to position `open_pos`:

```python
# Given: state.cache is a dict[layer_name -> LayerCache]
# LayerCache = {'k': ..., 'v': ..., 'end_index': [B], 'positions': [B, S]}
def prune_to(state, open_pos):
    new_cache = {
        name: {
            **layer,
            'end_index': jnp.array([open_pos], dtype=jnp.int32),
        }
        for name, layer in state.cache.items()
    }
    return state.replace(
        cache=new_cache,
        last_token_pos=jnp.array([open_pos], dtype=jnp.int32),
    )
```

That's it.  No buffer resizing, no head-dim branching, no per-layer-type
dispatch.  The K/V at positions `[open_pos .. old_end_index]` physically
remain in the ring buffer but are no longer addressable: the next
`_sample_step` writes at `update_index = end_index % cache_size =
open_pos % cache_size`, overwriting them; the sliding-window attention
mask is derived from `positions[:end_index]`, so unreached slots don't
leak into attention.

**Correctness caveats to verify:**
- Positions in `positions[open_pos:]` still hold stale absolute indices
  from the pre-prune pass.  The next `_sample_step` overwrites them
  via `dynamic_update_slice` as new tokens are written — but if the
  attention-mask builder ever reads beyond `end_index`, stale data
  would leak.  Sanity-check that `positions[end_index:]` is never
  consumed.  (From the code: `positions` is always read via
  `positions[:cumulative_length]`; `end_index` is the gate, so this
  should be safe — but verify with a test.)
- Under Option A (drop all three markers), renumber contiguously: the
  summary tokens re-processed after the prune get positions starting
  at `open_pos`, matching the rewound cache exactly.

---

## Phased implementation plan

Four phases, each gated on the previous one succeeding.  Scope is
intentionally narrow in Phase 1 (pruning correctness only — no
training, no behaviour change to stock outputs).  Later phases build
on a verified foundation.

### Proposed module layout (added incrementally across phases)

```
lib/gemma_jax/                 # added in Phase 1
  __init__.py
  setup.py                     # tokenizer customization + marker-ID resolution
  markers.py                   # token-name constants (port from lib/gemma/markers.py)
  prune_sampler.py             # PruningChatSampler subclassing gm.text.ChatSampler
  dataset.py                   # added in Phase 2: Seq2SeqTask adapter + collapse_nested
  rewards.py                   # added in Phase 3: correctness / compression / format
  bench.py                     # added in Phase 4: accuracy/len/memory reporter
scripts/
  gemma_jax_verify.py          # Phase 1: parity + prune-correctness tests
  gemma_jax_sample.py          # Phase 1: hand-run a prompt with pruning on/off
  gemma_jax_sft.py             # Phase 2: Kauldron Trainer driver
  gemma_jax_rl.py              # Phase 3: RL driver
  gemma_jax_benchmark.py       # Phase 4: MMLU / GSM8K / AIME / Polymath harness
```

`PruningChatSampler` subclasses `gm.text.ChatSampler`.  `ChatSampler`
is the stateful, multi-turn, multimodal-aware entry point that
auto-detects the model and dispatches to the right inner sampler
(`Gemma4Sampler` for Gemma 4, `Sampler` for earlier versions); it
also owns `last_state` and the `turns` log.  Subclassing means our
pruner slots in as a drop-in replacement — users write `chat(...)`,
get cached multi-turn, native `dialog.Conversation` formatting,
image/audio support, and transparent prune events on `<return|>`.

The `lib/gemma/dataprep/` pipeline we already built (stage-1 sample
Gemma traces, stage-2 Gemini hierarchization) stays as-is — its
output is a plain HF dataset of `{question, hierarchical_cot,
final_answer, ...}` records, which the JAX path consumes through a
thin `dataset.py` adapter in Phase 2.

### Trigger token: `<return|>` (decided)

The prune is signalled by a single custom `<return|>` token registered
into an unused `<unusedN>` SentencePiece slot.  Rationale:

- The underlying cache surgery (rewind `end_index` + re-forward
  summary) is the tested, trigger-agnostic mechanism.  The choice of
  trigger does not affect correctness.
- A single token is the cheapest representation (1 token per prune
  vs ~15-30 for a tool-call JSON payload) — best compression ratio.
- Upstream Gemma 4's `_Gemma4SpecialTokens` enum does not (yet)
  define `CUSTOM`, but the SentencePiece binary still ships
  `<unusedN>` slots; our `setup.py` patches around the missing enum
  member once and moves on.
- The tool-call alternative (model emits `<|tool_call>{"name":
  "prune","summary":"..."}<|tool_call|>`) was considered and rejected
  for Phase 1.  It would lean on Gemma 4's pre-trained tool-use
  behaviour but at a significant token-budget cost and with heavier
  SFT plumbing.  The cost/benefit didn't favour it enough to switch.
- `<|channel>` and `<channel|>` (native Gemma 4 pieces) remain the
  open/close delimiters of sub-chain-of-thought blocks; `<return|>`
  marks the prune event at the end.

---

### Phase 1 — Prune mechanism produces bit-equivalent output to stock sampler

**Goal.** Implement the pruning hook as a subclass of
`gm.text.ChatSampler` and prove it is a no-op when no `<return|>` is
generated.  On a standard Gemma 4 E4B prompt, `PruningChatSampler`
must yield the same tokens (or, with bf16, numerically equivalent
logits) as the stock `ChatSampler` run with the same seed.  Side
goal: learn whether the `Gemma4` model definition needs any changes
to accommodate our pruning (expected: none — confirm empirically).

**Work.**

1. **`setup.py`**: `gm.text.Gemma4Tokenizer(..., custom_tokens={N:
   "<return|>"})` for an unused slot `N`.  Resolve `<|channel>`,
   `<channel|>` IDs via `tokenizer.encode(...)`; assert single-token.
   Patch around the missing `_Gemma4SpecialTokens.CUSTOM` enum
   member (upstream oversight).

2. **`prune_sampler.py`**: `PruningChatSampler` subclasses
   `gm.text.ChatSampler`.  Override `chat(...)` to drive a segmented
   sampling loop:
   - Build the underlying `SamplerLoop` with `end_tokens =
     (EOS, END_OF_TURN, BEGIN_OF_TOOL_RESPONSE, return_id, ...stop)`.
   - Iterate: run the JIT-compiled `SamplerLoop._sample_loop` until
     one of those fires; inspect `state.last_token[0]`.
     - If it's `return_id`: reconstruct the matching `<|channel>` /
       `<channel|>` positions by scanning `predicted_tokens[0,
       :state.step]` with a simple balanced-bracket stack walk; apply
       the prune (rewind every layer's `end_index` to `open_pos` via
       `state.cache_info.set_end_index`, re-forward the summary one
       token at a time, sample `next_token` from the post-summary
       logits, collapse `predicted_tokens` so the pruned trace is
       `[..., summary, next_token, 0, ...]`, reset `state.done` to
       all-False).  Loop.
     - Otherwise: exit.  `ChatSampler.chat` then records the
       (prompt, response) turn and stashes `last_state` per its normal
       flow.
   - When pruning is disabled, skip the segment loop entirely and
     delegate straight to `super().chat(...)` — byte-identical
     to stock `ChatSampler`.

3. **`scripts/gemma_jax_verify.py`** — three tests gated in order:
   - *`--test token_init`*: tokenizer custom-token registration and
     marker-ID discovery succeed.
   - *`--test sampler_noop`*: run a prompt that emits ordinary text
     (no channel markers) through both stock `ChatSampler` and
     `PruningChatSampler` with the same seed and prefill.  Assert
     identical token streams (or, for bf16, assert logit-cosine >
     0.9999 at every step).  This is the "does not break anything"
     gate.
   - *`--test prune_parity`*: hand-construct `[prompt, <|channel>,
     thought, <channel|>, summary, <return|>]`, feed through the
     sampler to trigger a prune, force-continue one more token.
     Compare against a reference `model(...)` forward on `[prompt,
     summary]`.  Accept cosine > 0.99 and top-1 agreement.
4. **`scripts/gemma_jax_sample.py`**: CLI to run a prompt with
   pruning on vs off.  Human-readable smoke test; not asserted, but
   must produce coherent output in both modes.

**Deliverables.**  `lib/gemma_jax/{setup,markers,prune_sampler}.py`,
two scripts above, a short write-up naming any model-definition
change needed.

**Gate to Phase 2.**  All three `verify.py` tests pass.  If
`sampler_noop` fails, Phase 1 is not done — the prune code is
corrupting normal generation somehow.

**Status note (verified 2026-05-06).**  Phase 1 is complete on Gemma 4
E4B-IT, run on Modal (A100-80GB).  All three `gemma_jax_verify.py`
tests pass:

* `token_init` — `<|channel>`=100, `<channel|>`=101, `<return|>`=6,
  all round-trip to single ids on the 262144-vocab Gemma 4 tokenizer.
* `sampler_noop` — stock `gm.text.ChatSampler` and
  `PruningChatSampler(pruning_enabled=True)` emit byte-identical
  token streams (40/40 match) on a no-marker prompt.
* `prune_parity` — hand-crafted cache-rewind logic produces logits
  with cosine = 1.000000 and matching top-1 against a reference
  `[prompt, summary]` forward.

The implementation history: commit `c15bc89` shipped a first version
inheriting from `gm.text.Sampler` with a per-token streaming loop;
commit `47eac01` refactored it onto the `ChatSampler` segmented-JIT
base described above.  The verify run uncovered two real bugs that
were fixed in the harness:

* Two prefill sites in `gemma_jax_verify.py` (`_forward_logits` and
  `_teacher_forced_prune_logits`) constructed an attention mask of
  shape `(1, 1, cache_length)` instead of the
  `(1, L, cache_length)` causal mask Gemma 4's signature requires.
* The verify script's `--test all` path leaked GPU memory across
  tests; on E4B the second `load_params` OOMed even on A100-80GB.
  Fixed by running each test in its own subprocess via
  `scripts/_gemma_jax_verify_runner.py` (Modal-side wrapper).

Modal infrastructure: `scripts/modal_verify_gemma_jax.py` builds a
JAX-CUDA image and ships verify on a single A100-80GB.  GCE
metadata-service auth probes are explicitly suppressed
(`GOOGLE_AUTH_DISABLE_GCE_METADATA_LOOKUP=1`,
`GCE_METADATA_HOST=disabled`) so anonymous GCS reads of
`gs://gemma-data/...` succeed immediately without 60s probe stalls.

Phase 2 may begin.

---

### Phase 2 — SFT with hierarchical data, benchmark, bug-hunt

**Goal.**  Train Gemma 4 E4B (LoRA) on the hierarchical dataset
produced by `lib/gemma/dataprep/`.  Confirm the trained model emits
well-formed hierarchical structure; confirm the Phase-1 pruner fires
correctly on its generations; catch latent bugs in the dataset
pipeline, collapse logic, or sampler under realistic conditions.

**Work.**

1. **`lib/gemma_jax/dataset.py`**:
   - Port `collapse_nested` verbatim from `lib/gemma/dataset.py`
     (pure regex — no torch).
   - Adapter: HF dataset (`question`, `hierarchical_cot`,
     `final_answer`) → `Seq2SeqTask`-shaped dicts.  Emit two variants
     per record: `"full"` (hierarchical as-is) and `"collapsed"`
     (nested blocks replaced by summaries).
   - Tokenize prompt and response separately, build `loss_mask` = 0
     on prompt, 1 on response.
2. **`scripts/gemma_jax_sft.py`**:
   - `kd.Trainer` modelled on `/examples/lora.py`.
   - `gm.nn.LoRA(model=Gemma4_E4B(), rank=16)`.
   - `optax.adafactor` + `kd.optim.partial_updates(mask=kd.optim.select("lora"))`.
   - `kd.losses.SoftmaxCrossEntropyWithIntLabels(mask=loss_mask)`.
   - `kd.sharding.FSDPSharding()`.
   - Orbax checkpoints.
3. **Benchmarks for this phase** (internal, not Phase 4's public ones):
   - SFT train/eval loss curve.
   - Hold-out perplexity on a reserved split of the hierarchical
     dataset.
   - Sample-generation QA: pick 20 held-out prompts; for each, run
     the Phase-1 pruner on the SFT model and check (a) structural
     well-formedness via `lib/gemma/dataprep/validate.py` (port
     verbatim), (b) final-answer correctness vs the dataset's
     `expected_answer`, (c) number of prune events fired, (d) ratio
     of generated-tokens to pruned-tokens.
4. **Bug-hunt deliverable**: the sample-generation QA will surface
   failure modes (model emits markers but forgets to close, emits
   `<return|>` without an open, summary is empty, etc.).  Fix each in
   the dataset pipeline, sampler, or SFT loss masking as appropriate
   before advancing.

**Deliverables.**  `lib/gemma_jax/dataset.py`, `scripts/gemma_jax_sft.py`,
trained LoRA checkpoint, a one-page report with loss curves + the 4
numbers from step 3 above.

**Gate to Phase 3.**  SFT model produces structurally valid
hierarchical traces ≥ 80% of the time on the QA sample and its
final-answer accuracy on the held-out split is within 5% of a
non-hierarchical SFT baseline (i.e., we haven't destroyed base task
performance).

---

### Phase 3 — RL to encourage deeper thinking with correct output

**Goal.**  RL fine-tune the Phase-2 SFT checkpoint with rewards that
encourage (a) correct final answers, (b) meaningful compression via
real sub-CoT use, (c) well-formed hierarchical structure.  Avoid
reward-hacking modes (empty thought blocks, trivial summaries).

**Work.**

1. **`lib/gemma_jax/rewards.py`**:
   - *Correctness*: task-specific where possible (math-verify for
     math, exact-match for QA), LLM-judge fallback for open-ended
     prompts.  Gate all other rewards on this one.
   - *Compression*: `pruned_len / unpruned_len` clipped and
     transformed so that short thought blocks are discouraged and
     substantive summaries are rewarded.  Multiplied by correctness
     gate so that empty thoughts score 0.
   - *Format*: structural well-formedness from Phase 2's validator,
     as a bounded bonus.
2. **RL driver**: two options, pick one:
   - *Option A (native)*: port a minimal GRPO trainer to Kauldron /
     JAX.  Non-trivial — GRPO's group-relative advantage computation
     has to live inside the training step.
   - *Option B (hybrid)*: export the Phase-2 LoRA checkpoint to HF
     format (manual weight mapping, one-time cost), then run RL with
     the existing `scripts/_grpo_worker.py` + new general-purpose
     rewards.  Deploy back to JAX only if Phase 4 benchmarking shows
     PyTorch serving doesn't meet targets.
   - Decision is made at the top of Phase 3 based on current team
     capacity / JAX depth.  **Default recommendation: Option B**
     because the GRPO worker exists and RL is where bugs multiply.
3. **Sampling during RL**: the rollout sampler must be the Phase-1
   `PruningChatSampler` so the reward function sees the
   post-prune trajectory.  This is the non-negotiable coupling
   between phases.

**Deliverables.**  `lib/gemma_jax/rewards.py`, RL driver script
(JAX or HF depending on option chosen), RL-trained checkpoint, a
two-page report comparing pre-RL vs post-RL on the Phase-2 metrics
plus reward-curve plots.

**Gate to Phase 4.**  Post-RL model beats Phase-2 SFT on at least
one of {compression ratio at same accuracy, accuracy at same
length} without reward-hacking (verified by manual inspection of
rollouts).

---

### Phase 4 — Public benchmarks

**Goal.**  Measure end-to-end system performance on standard public
benchmarks.  Establish the headline numbers: accuracy, generated-token
count, KV-cache peak memory, end-to-end latency — for our
hierarchical-pruned model vs. comparable baselines.

**Work.**

1. **Benchmarks chosen** (with rationale):
   - *GSM8K* (math word problems) — standard, has verifiable answers,
     well-studied at E4B scale.
   - *MMLU (5-shot)* — broad multi-domain QA; good for "does
     hierarchical CoT hurt factual recall".
   - *AIME-24 / AIME-25* — harder math, longer reasoning, where
     pruning matters most for context efficiency.
   - *Polymath* — already wired into `lib/eval/` in the HF branch;
     port the benchmark driver only.
2. **Baselines**:
   - Gemma 4 E4B base (no SFT, no RL, no pruning).
   - Gemma 4 E4B SFT-only (Phase 2 checkpoint, pruner off — sanity
     check that SFT alone didn't regress base).
   - Gemma 4 E4B SFT + pruner on.
   - Gemma 4 E4B SFT + RL + pruner on (the target).
3. **Metrics per benchmark**:
   - Accuracy (task-standard metric).
   - Mean generated token count.
   - Mean *visible* token count (post-prune).
   - Mean number of prune events fired.
   - Peak KV-cache memory during generation (approximate via
     `end_index` sum × bytes-per-KV-entry).
   - Wall-clock latency per sample.
4. **`lib/gemma_jax/bench.py`**: benchmark runner; loops over
   (benchmark × baseline) and writes a CSV + a Markdown summary.

**Deliverables.**  `lib/gemma_jax/bench.py`,
`scripts/gemma_jax_benchmark.py`, a benchmark report with a single
summary table and per-benchmark detail sections.

**Gate.**  This is the final deliverable; no further phase.  Success
criterion is well-defined numbers, not a specific target — the
report stands regardless of which baseline wins.

---

### Critical files in the gemma repo to reference

| Concern              | File |
| -------------------- | ---- |
| Model class          | `/gemma/gm/nn/gemma4/_gemma4.py` (`Gemma4_E4B`) |
| Transformer forward  | `/gemma/gm/nn/gemma4/_transformer.py` |
| Cache definition     | `/gemma/gm/nn/_modules.py:34,214–234,326–335` |
| Attention + RoPE     | `/gemma/gm/nn/gemma4/_modules.py`, `/gemma/gm/math/_positional_embeddings.py` |
| Attention-type pattern | `/gemma/gm/nn/gemma4/_config.py::make_attention_layers_types` |
| Sampler (JIT)        | `/gemma/gm/text/_sampler_loop.py::SamplerLoop._sample_loop` |
| Sampler (streaming)  | `/gemma/gm/text/_sampler_loop.py::SamplerLoop._stream_sample_loop` — **the pruning hook** |
| Chat sampler         | `/gemma/gm/text/_chat_sampler.py::ChatSampler` |
| Prefill              | `/gemma/gm/text/_prefill.py::prefill` |
| Tokenizer + custom_tokens | `/gemma/gm/text/_tokenizer.py:136–200,316–334` |
| LoRA                 | `/gemma/gm/nn/_lora.py::LoRA` |
| SFT task shape       | `/gemma/gm/data/_tasks.py::Seq2SeqTask` |
| LoRA training example | `/examples/lora.py` (Kauldron `kd.Trainer` driver to mimic) |

### Code to reuse verbatim from the current HF branch

- `lib/gemma/markers.py` — constant names → `lib/gemma_jax/markers.py`.
- `lib/gemma/dataset.py::collapse_nested` — pure regex, no torch → `lib/gemma_jax/dataset.py`.
- `lib/gemma/dataprep/*` — stage-1 sampling, stage-2 Gemini
  segmentation, well-formedness checks.  No changes; its output is a
  plain HF dataset.
- `lib/trainer/rewards.py` (in the HF branch) as a starting point for
  `lib/gemma_jax/rewards.py` — math-specific rewards to be generalized.

---

## Risks, with mitigations

1. **JIT outer loop vs streaming latency.**  We must use
   `_stream_sample_loop` (Python outer loop) rather than the JIT
   `while_loop` to intervene on `<return|>`.  Per-step forward is
   still JIT-compiled; we lose only the outer-loop fusion.  In
   practice this should be a modest slowdown (maybe 10–20%).  **If
   it's worse**, explore `jax.pure_callback` to keep the outer JIT
   and escape to Python just on prune events — more complex but
   preserves most fusion.

2. **Tokens not single-piece in the SP vocab.**  `<|channel>` etc.
   may tokenize to 2+ subword pieces, breaking the "one special
   token per event" assumption.  **Mitigation**: test at setup time
   (`verify.py::Test 1`); if multi-piece, fall back to matching a
   short token sequence rather than a single ID in the stack
   bookkeeping.

3. **Gemma 4 E4B may not emit thinking-channel markers natively**
   in the DeepMind checkpoint.  HF's chat template opens
   `<|channel>thought` via a Jinja preamble when `enable_thinking=True`
   — the DeepMind chat sampler uses a simpler
   `<start_of_turn>user\n...\n<start_of_turn>model\n` template with
   no channel preamble.  **Mitigation**: add our own system preamble
   or forced-prefix `<|channel>thought\n` at the start of the
   generation.  Verify with a baseline generation.

4. **Orbax → HF export path is a manual weight mapping.**  If RL or
   production inference needs PyTorch/vLLM, we're on the hook for
   writing a converter.  Not impossible (Gemma 4 layer shapes are
   well-known) but real work.  **Mitigation**: if we need both, do
   SFT in JAX (cleaner LoRA) then convert for RL/inference, or accept
   dual-tracking.

5. **No GRPO in Kauldron.**  For the RL phase we either write a
   GRPO trainer in JAX (non-trivial) or convert checkpoints and do
   RL in PyTorch/TRL.  **Mitigation**: defer — SFT first, decide RL
   path once we have an SFT-trained checkpoint.

---

## Summary

We are building on the DeepMind gemma (JAX/Flax) repo, on branch
`claude/gemma4-context-compression-PoHkM`, under a new
`lib/gemma_jax/` package.  Work proceeds in four phases:

1. **Pruning correctness.**  Implement the `<|channel>` /
   `<channel|>` / `<return|>` stack-based pruner over Gemma 4 E4B
   using `end_index` rewind.  Prove the new sampler is a no-op
   vs. stock `Gemma4Sampler` when no `<return|>` fires, and that
   forced prune events produce logits within tolerance of a
   reference `[prompt, summary]` forward.
2. **SFT.**  Train Gemma 4 E4B + LoRA on the hierarchical dataset
   produced by `lib/gemma/dataprep/`.  Internal benchmarks
   (structure validity, final-answer accuracy on held-out, prune
   behaviour on sample generations) drive the bug-hunt.
3. **RL.**  Tune the SFT checkpoint to prefer deeper hierarchical
   reasoning with correct final answers, using correctness-gated
   compression + format rewards.  GRPO path decided at the top of
   the phase.
4. **Public benchmarks.**  Measure accuracy, token counts, KV-cache
   memory, and latency on GSM8K / MMLU / AIME / Polymath for base
   vs. SFT vs. SFT+RL with and without pruning.

The cheapest, highest-value gate is Phase 1's `sampler_noop` test:
it validates that the pruning hook does not perturb stock behaviour
and, if it fails, tells us concretely whether the `Gemma4` model
definition itself needs changes before any training cycles are spent.

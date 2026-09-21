# Llm — a CPU inference engine built on Traktor

A standalone example: a GGUF language-model runner and a ChatGPT-style chat
window, written entirely against Traktor's own `Core` and `Ui`. No third-party
code, no GPU. It is not part of the engine — it links against it the way any
external tool would, through `ExternalDependency` on the main solution.

`docs/index.html` walks the whole path — the container, the tokenizer, the
forward pass, the key/value cache and the sampler — one stage at a time, with
each step tied back to the source. Open it in a browser; it needs no server.

## What it supports

- **Container:** GGUF v2 and v3, memory mapped, weights decoded on the fly.
- **Quantization:** F32, F16, Q4_0, Q8_0, Q4_K, Q6_K — enough for the common
  `Q4_K_M`, `Q4_K_S` and `Q8_0` downloads. Anything else is refused by name.
- **Architectures:** `llama` and `mistral` (normal rotary) and `qwen2` (NeoX
  rotary). Others are refused rather than guessed at, because a wrong rotary
  type still reads fluently — it just stops tracking word order.
- **Tokenizers:** SentencePiece with byte fallback, and byte-level BPE with the
  GPT-2, Llama 3 and Qwen 2 pre-tokenizer patterns.
- **Chat markup:** ChatML, Llama 3 and Llama 2, recognized from the markers the
  model's Jinja template mentions rather than from its name.
- **Rotary scaling:** linear, and the per-channel `rope_freqs.weight` form that
  Llama 3.1 and later use. YaRN and LongRoPE are detected and warned about.
- **Sampling:** temperature, top-k, top-p and a repetition penalty, or greedy.

## Building

```
./build-projects-make-linux.sh          # generates tools/Llm/build/linux
cd build/linux/Llm     && make ReleaseShared
cd build/linux/Llm.App && make ReleaseShared
```

It links against the Traktor shared libraries aggregated in
`bin/latest/<platform>/releaseshared`, so the engine has to have been built
first. Its own products are aggregated next to them, which is why `run.sh`
needs no library path of its own.

## Running

```
./run.sh                             # open the chat window, then Open model...
./run.sh model.gguf                  # open the chat window with a model loaded
./run.sh -context=8192 model.gguf    # more context, at the cost of cache memory
./run.sh -selftest model.gguf        # bring-up report, no window
```

`-context=N` sets how many positions to allocate, defaulting to 4096. It is
clamped to what the model was trained for, quietly when that is just the
default landing above a small model and with a warning when you asked for more
than the model can do. The key/value cache is reported before it is allocated,
because it is the largest thing here and grows linearly with the context: a 3B
model with 36 layers wants 288 MiB at 4096 and 2.25 GiB at its full 32k.

`-selftest [-tokens=N]` is the first thing to reach for with a new model. It
prints the hyperparameters, the rotary scaling in force, every tensor's shape
and storage type with its first decoded weights, a tokenizer round trip, a
summary of the logits, and a short greedy generation with its rates.

The **logits line** is the one to diff — across a code change, or against a
reference implementation. It moves on any difference in the forward pass,
including differences far too small to change the sampled token. A greedy
argmax is not a sensitive test.

## The inspector

The panel to the right of the transcript shows the reply being made rather
than the reply. Its header keeps the latest forward pass in view as a bar
split by stage, with a legend giving each stage's share; the log below it
scrolls, and an entry expands or collapses when clicked.

- **Prompt.** Which chat template was used, how many bytes it produced and
  how many tokens they became. The tokens appear as chips that fill in one by
  one as they are read into the key/value cache, so the pre-tokenizer's cuts
  and the template markup are both visible. Tokens the cache already held
  from the previous turn are counted rather than re-read.
- **One entry per generated token.** The top of the distribution the token
  was drawn from, as bars: the chosen token in the accent color, the others
  that survived top-k and top-p filled, the ones that were cut outlined. The
  entropy in bits says how sure the model was. Under that, where this token's
  forward pass spent its time, and the length of the residual stream after
  every layer, which climbs because each layer only ever adds to it. The end
  of turn token gets an entry too, since choosing to stop is a choice.
- **Finished.** Why the reply ended, with the prompt and reply rates.

**Interactive** on the toolbar hands the sampler's job to you. At every step
the generator draws a token as usual, then stops and shows the distribution
in the inspector with the sampler's draw outlined; click any row to make
that the next token, including one the cuts would have excluded, or the end
of turn token to stop. The reply continues from your choice, and the token's
entry records both what you picked and what the sampler drew. **Stop** ends a
reply left waiting; switching the mode off mid reply lets the pending draw
stand. Turning it on shows the inspector if it was hidden, since that is
where the choosing happens.

The account comes from `Generator::flushTrace`, fed by optional trace
arguments to `Context::evaluate` and `Sampler::sample`; with the panel hidden
(the **Inspector** toolbar button) nothing is recorded. The timing adds a
timer read per stage and a dot product per layer to each pass, which is
below the noise of the matrix products around them.

## Testing without a model

`scripts/misc/make-test-gguf.py` writes synthetic GGUF files that need no
download, together with a `.expected` file giving the values every tensor must
decode to. The quantized blocks are generated field by field from the format
definition, so the expected values are an independent reading of the same
layout this engine implements.

```
python3 ../../scripts/misc/make-test-gguf.py out.gguf
python3 ../../scripts/misc/make-test-gguf.py --tokenizer bpe --pre qwen2 out.gguf
python3 ../../scripts/misc/make-test-gguf.py --rope-freqs varied out.gguf
```

## Known limits

- **Speed.** The kernels use `Vector4`, which is SSE2; Traktor's math has no
  AVX2 path and there is no int8 dot product. Expect several times less than
  llama.cpp on the same machine. An AVX2 kernel would be the single biggest
  win available.
- **No batched prefill.** A prompt is read one token at a time, so a long
  prompt costs one full forward pass per token.
- **Context defaults to 4096**, not to what the model was trained for, so long
  conversations are truncated from the front sooner than the model could
  manage. Raise it with `-context=N` if you have the memory for the cache.
- **The transcript wraps, the input does not.** `ui::Edit` is single line;
  Enter sends.

"""BORA-1.1B-A0.4B 텍스트 생성 테스트 (스트리밍)."""
import json
import os
import sys
import time

import mlx.core as mx
import mlx.nn as nn
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import Model, ModelArgs


def load_model(checkpoint_path: str):
    """체크포인트에서 모델 로드."""
    model_dir = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(model_dir, "config.json")) as f:
        config = json.load(f)

    args_dict = {k: v for k, v in config.items()
                 if hasattr(ModelArgs, k) or k in ModelArgs.__dataclass_fields__}
    model_args = ModelArgs(**args_dict)

    model = Model(model_args)
    weights = mx.load(checkpoint_path)
    model.load_weights(list(weights.items()), strict=False)
    mx.eval(model.parameters())
    model.eval()

    print(f"모델 로드 완료: {checkpoint_path}")
    from mlx.utils import tree_flatten
    total = sum(p.size for _, p in tree_flatten(model.parameters()))
    print(f"파라미터: {total:,} ({total/1e6:.0f}M)")

    return model, model_args


def _sample(logits, prev_ids, temperature, top_p, rep_penalty):
    """Sampling step (compiled)."""
    last_logits = logits[:, -1, :]

    # Vectorized repetition penalty
    if rep_penalty != 1.0:
        vals = last_logits[0, prev_ids]  # gather
        penalties = mx.where(vals > 0,
                             -vals * (1 - 1/rep_penalty),
                             -vals * (rep_penalty - 1))
        last_logits = last_logits.at[0, prev_ids].add(penalties)

    if temperature > 0:
        last_logits = last_logits / temperature
        probs = mx.softmax(last_logits, axis=-1)
        sorted_indices = mx.argsort(probs, axis=-1)[..., ::-1]
        sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1)
        cumsum = mx.cumsum(sorted_probs, axis=-1)
        mask = cumsum - sorted_probs > top_p
        sorted_probs = mx.where(mask, 0.0, sorted_probs)
        sorted_probs = sorted_probs / mx.sum(sorted_probs, axis=-1, keepdims=True)
        token = mx.random.categorical(mx.log(sorted_probs + 1e-10))
        token = mx.take_along_axis(sorted_indices, token[:, None], axis=-1).squeeze(-1)
    else:
        token = mx.argmax(last_logits, axis=-1)

    return token


def _eval_with_cache(logits, cache):
    """Evaluate logits and all cache states (KVCache + EMACache)."""
    arrays = [logits]
    for c in cache:
        if c is not None:
            s = c.state
            if isinstance(s, tuple):
                arrays.extend(s)
            else:
                arrays.append(s)
    mx.eval(*arrays)


def generate_stream(model, tokenizer, prompt: str, max_tokens: int = 256,
                    temperature: float = 0.8, top_p: float = 0.95,
                    repetition_penalty: float = 1.1):
    """Autoregressive 생성 (토큰 단위 스트리밍)."""
    tokens = tokenizer.encode(prompt, add_special_tokens=True)
    tokens = mx.array([tokens])

    cache = model.make_cache()
    generated = []
    eos_id = tokenizer.eos_token_id or 2

    # Prefill
    t0 = time.time()
    logits = model(tokens, cache=cache)
    _eval_with_cache(logits, cache)
    prefill_time = time.time() - t0
    prompt_len = tokens.shape[1]

    gen_start = time.time()

    for i in range(max_tokens):
        prev_ids = mx.array(generated[-50:]) if generated else mx.array([eos_id])
        token = _sample(logits, prev_ids, temperature, top_p, repetition_penalty)
        token_id = token.item()

        if token_id == eos_id:
            break

        generated.append(token_id)

        # 스트리밍: 토큰 즉시 디코드 출력
        text = tokenizer.decode([token_id])
        print(text, end="", flush=True)

        # Next step
        logits = model(token[:, None], cache=cache)
        _eval_with_cache(logits, cache)

    gen_time = time.time() - gen_start
    n_gen = len(generated)
    tok_s = n_gen / gen_time if gen_time > 0 else 0

    print()
    print(f"  [{prompt_len} prompt → {n_gen} tokens | "
          f"prefill {prefill_time:.2f}s | "
          f"gen {gen_time:.1f}s | "
          f"{tok_s:.1f} tok/s]")

    return tokenizer.decode(generated)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="./checkpoints/model_250.safetensors")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--interactive", action="store_true",
                        help="대화형 모드")
    args = parser.parse_args()

    model, model_args = load_model(args.checkpoint)

    tokenizer_dir = os.path.dirname(os.path.abspath(__file__))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    print(f"토크나이저: vocab_size={len(tokenizer)}")

    meta_path = args.checkpoint.replace(".safetensors", ".json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"Step: {meta.get('step')}, Loss: {meta.get('loss', 'N/A'):.4f}, "
              f"LR: {meta.get('lr', 'N/A')}")

    print(f"\n설정: temp={args.temperature}, top_p={args.top_p}, "
          f"rep_penalty={args.repetition_penalty}, max_tokens={args.max_tokens}")
    print("=" * 60)

    gen_kwargs = dict(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )

    if args.interactive:
        print("대화형 모드 (quit으로 종료)\n")
        while True:
            try:
                prompt = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if prompt.lower() in ("quit", "exit", "q"):
                break
            if not prompt:
                continue
            print()
            generate_stream(model, tokenizer, prompt, **gen_kwargs)
            print()
    else:
        prompts = [
            args.prompt or "한국의 수도는",
            "인공지능이란",
            "오늘 날씨가",
        ] if args.prompt is None else [args.prompt]

        for prompt in prompts:
            print(f"\n프롬프트: {prompt}")
            generate_stream(model, tokenizer, prompt, **gen_kwargs)
            print("-" * 40)


if __name__ == "__main__":
    main()

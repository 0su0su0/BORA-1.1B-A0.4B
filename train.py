"""
BORA-1.1B-A0.4B Pre-training Script
EMA + NSA + MoE (16 routed + 2 shared experts)

Usage:
    python3 train.py --data-dir /path/to/jsonl_files
"""

import argparse
import glob
import json
import math
import os
import random
import time
from functools import partial

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

# Local model import
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import Model, ModelArgs


def _sync_compile_state(target, source):
    """mx.compile 호환 state 동기화."""
    for key, val in source.items():
        if isinstance(val, dict):
            if key in target and isinstance(target[key], dict):
                _sync_compile_state(target[key], val)
            else:
                target[key] = val
        else:
            target[key] = val


def str_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected (true/false)')


mx.set_memory_limit(112 * 1024 * 1024 * 1024)  # 112GB

# ============================================================
# Config
# ============================================================

DEFAULT_CONFIG = {
    'batch_size': 4,
    'grad_accum': 64,
    'max_seq_len': 2048,
    'max_steps': 10000,
    'lr': 3e-4,
    'min_lr_ratio': 0.1,
    'warmup_steps': 500,
    'weight_decay': 0.1,
    'grad_clip': 1.0,
    'z_loss_weight': 1e-4,
    'seed': 888,
    'save_every': 500,
    'log_every': 10,
    'keep_checkpoints': 10,
}


# ============================================================
# JSONL Data Loader
# ============================================================

class JsonlDataLoader:
    """JSONL 파일에서 텍스트를 읽어 토큰화하는 데이터 로더."""

    def __init__(self, data_dir, tokenizer, batch_size=4, max_seq_len=2048, seed=888):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.seed = seed

        # JSONL 파일 찾기
        self.files = glob.glob(os.path.join(data_dir, "**/*.jsonl"), recursive=True)
        if not self.files:
            print(f"⚠️  No JSONL files found in {data_dir}")
            self.files = []
        else:
            print(f"📂 Found {len(self.files)} JSONL files")

        random.seed(seed)

    def __iter__(self):
        """배치를 생성하는 generator."""
        if not self.files:
            print("⚠️  No data files available. Using dummy data for testing.")
            # Dummy data for testing
            while True:
                input_ids = mx.array(np.random.randint(0, 32128, (self.batch_size, self.max_seq_len)))
                labels = mx.array(np.random.randint(0, 32128, (self.batch_size, self.max_seq_len)))
                yield input_ids, labels

        # 파일 셔플
        random.shuffle(self.files)

        for file_path in self.files:
            try:
                buffer = []
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            obj = json.loads(line)
                            text = obj.get("text", "")
                            if not text:
                                continue

                            # 토큰화
                            tokens = self.tokenizer.encode(text, add_special_tokens=True)
                            buffer.extend(tokens)

                            # 배치 생성
                            while len(buffer) >= (self.max_seq_len + 1) * self.batch_size:
                                batch_inputs = []
                                batch_labels = []

                                for _ in range(self.batch_size):
                                    chunk = buffer[:self.max_seq_len + 1]
                                    batch_inputs.append(chunk[:-1])
                                    batch_labels.append(chunk[1:])
                                    buffer = buffer[self.max_seq_len + 1:]

                                yield mx.array(batch_inputs), mx.array(batch_labels)
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                print(f"⚠️  Error reading {file_path}: {e}")
                continue


# ============================================================
# Loss
# ============================================================

LOSS_CONFIG = {
    'z_loss_weight': 1e-4,
    'pad_id': 0,
}


def loss_fn(model, input_ids, labels):
    """Cross-entropy loss with z-loss regularization."""
    logits = model(input_ids)

    vocab_size = logits.shape[-1]
    logits_flat = logits.reshape(-1, vocab_size)
    targets = labels.reshape(-1)

    # 패딩 마스크
    pad_id = LOSS_CONFIG['pad_id']
    mask = (targets != pad_id).astype(logits_flat.dtype)
    n_valid = mx.maximum(mx.sum(mask), mx.array(1.0))

    ce_per_token = nn.losses.cross_entropy(logits_flat, targets, reduction='none')
    ce_loss = mx.sum(ce_per_token * mask) / n_valid

    z_weight = LOSS_CONFIG['z_loss_weight']
    if z_weight > 0:
        log_z = mx.logsumexp(logits_flat, axis=-1)
        z_loss = z_weight * mx.sum(log_z ** 2 * mask) / n_valid
        return ce_loss + z_loss

    return ce_loss


# ============================================================
# LR Schedule
# ============================================================

def get_lr(step: int, warmup: int, max_steps: int, lr: float, min_lr_ratio: float) -> float:
    if step < warmup:
        return lr * (step + 1) / warmup
    if step >= max_steps:
        return lr * min_lr_ratio

    progress = (step - warmup) / (max_steps - warmup)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)


# ============================================================
# Gradient Clipping
# ============================================================

def clip_grad_norm(grads, max_norm: float):
    flat = tree_flatten(grads)
    total_norm_sq = mx.array(0.0)
    for _, g in flat:
        total_norm_sq = total_norm_sq + mx.sum(g * g)
    total_norm = mx.sqrt(total_norm_sq)

    scale = mx.minimum(mx.array(max_norm) / (total_norm + 1e-6), mx.array(1.0))
    grads = tree_unflatten([(k, v * scale) for k, v in flat])

    return grads, total_norm


# ============================================================
# Checkpoint
# ============================================================

def save_checkpoint(model, step: int, loss: float, lr: float,
                    save_dir: str, tag: str = None, optimizer=None):
    os.makedirs(save_dir, exist_ok=True)

    name = f"model_{tag}" if tag else f"model_{step}"

    weights = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(os.path.join(save_dir, f'{name}.safetensors'), weights)

    if optimizer is not None:
        opt_flat = dict(tree_flatten(optimizer.state))
        if opt_flat:
            opt_name = f"opt_{tag}" if tag else f"opt_{step}"
            mx.save_safetensors(os.path.join(save_dir, f'{opt_name}.safetensors'), opt_flat)

    metadata = {
        'step': step,
        'loss': float(loss),
        'lr': float(lr),
    }

    with open(os.path.join(save_dir, f'{name}.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Checkpoint saved: {name} (step={step}, loss={loss:.4f})")


def cleanup_checkpoints(save_dir: str, keep: int):
    pattern = os.path.join(save_dir, 'model_*.safetensors')
    files = sorted(glob.glob(pattern), key=os.path.getmtime)

    step_files = []
    for f in files:
        base = os.path.basename(f).replace('model_', '').replace('.safetensors', '')
        if base.isdigit():
            step_files.append(f)

    while len(step_files) > keep:
        old = step_files.pop(0)
        old_json = old.replace('.safetensors', '.json')
        old_opt = old.replace('model_', 'opt_')
        os.remove(old)
        if os.path.exists(old_json):
            os.remove(old_json)
        if os.path.exists(old_opt):
            os.remove(old_opt)


def find_latest_checkpoint(save_dir: str):
    import re
    pattern = re.compile(r'^model_(\d+)\.safetensors$')
    latest = None
    if not os.path.isdir(save_dir):
        return None
    for fname in os.listdir(save_dir):
        m = pattern.match(fname)
        if m:
            step = int(m.group(1))
            if latest is None or step > latest:
                latest = step
    return latest


def load_checkpoint(model, save_dir: str, step: int, optimizer=None):
    weights_path = os.path.join(save_dir, f'model_{step}.safetensors')
    if not os.path.exists(weights_path):
        print(f"⚠️  Checkpoint not found: {weights_path}")
        return model, {}

    weights = mx.load(weights_path)
    model.load_weights(list(weights.items()), strict=False)
    mx.eval(model.parameters())

    opt_path = os.path.join(save_dir, f'opt_{step}.safetensors')
    if optimizer is not None and os.path.exists(opt_path):
        opt_flat = mx.load(opt_path)
        optimizer.state = tree_unflatten(list(opt_flat.items()))
        mx.eval(optimizer.state)
        print(f"✅ Optimizer state loaded: opt_{step}.safetensors")
    elif optimizer is not None:
        print(f"⚠️  Optimizer state not found (starting fresh)")

    meta_path = os.path.join(save_dir, f'model_{step}.json')
    metadata = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            metadata = json.load(f)

    print(f"✅ Checkpoint loaded: step={step}")
    return model, metadata


# ============================================================
# Training
# ============================================================

def train(args):
    mx.random.seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Loss config
    LOSS_CONFIG['z_loss_weight'] = args.z_loss_weight

    # Load model
    model_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"📦 Loading model from {model_dir}")

    config_path = os.path.join(model_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    args_dict = {k: v for k, v in config.items()
                 if k in ModelArgs.__dataclass_fields__}
    model_args = ModelArgs(**args_dict)

    model = Model(model_args)
    weights_path = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(weights_path):
        weights = mx.load(weights_path)
        model.load_weights(list(weights.items()), strict=False)
        print(f"✅ Weights loaded: {weights_path}")
    else:
        print("⚠️  Using initial weights (no safetensors found)")

    model.unfreeze()

    flat_params = tree_flatten(model.parameters())
    total_params = sum(p.size for _, p in flat_params)
    print(f"📊 Parameters: {total_params:,} ({total_params/1e9:.2f}B)")

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer_path = model_dir
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        print(f"✅ Tokenizer loaded (vocab={tokenizer.vocab_size})")
    except:
        print("⚠️  Tokenizer not found in model dir, using fallback")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

    # Data
    print(f"\n📂 Loading data from {args.data_dir}")
    dataloader = JsonlDataLoader(
        args.data_dir,
        tokenizer,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        seed=args.seed,
    )

    print(f"\n⚙️  Training Configuration")
    print(f"  Batch size: {args.batch_size} × {args.grad_accum} = {args.batch_size * args.grad_accum} (effective)")
    print(f"  Max steps: {args.max_steps}")
    print(f"  LR: {args.lr} → {args.lr * args.min_lr_ratio} (cosine)")
    print(f"  Warmup: {args.warmup_steps} steps")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Grad clip: {args.grad_clip}")
    print(f"  Z-loss: {args.z_loss_weight}")

    # Optimizer
    optimizer = optim.AdamW(
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        betas=[0.9, 0.95],
        eps=1e-8
    )

    # Compile
    param_keys = [k for k, _ in tree_flatten(model.parameters())]
    state = [model.state, optimizer.state]

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    @partial(mx.compile, inputs=state, outputs=state)
    def compiled_step(input_ids, labels, acc_list):
        loss, grads = loss_and_grad_fn(model, input_ids, labels)
        flat_grads = tree_flatten(grads)
        new_acc = [acc_list[i] + flat_grads[i][1] for i in range(len(acc_list))]
        return loss, new_acc

    @partial(mx.compile, inputs=state, outputs=state)
    def do_update(acc_list, grad_accum):
        avg_grads = tree_unflatten([
            (param_keys[i], acc_list[i] / grad_accum) for i in range(len(acc_list))
        ])
        avg_grads, grad_norm = clip_grad_norm(avg_grads, args.grad_clip)
        optimizer.update(model, avg_grads)
        new_acc = [mx.zeros_like(g) for g in acc_list]
        return grad_norm, new_acc

    def zero_grads():
        return [mx.zeros_like(v) for _, v in tree_flatten(model.parameters())]

    # Resume
    start_step = 0
    latest_step = find_latest_checkpoint(args.save_dir)
    if latest_step is not None:
        print(f"🔄 Resuming from checkpoint: step {latest_step}")
        model, metadata = load_checkpoint(model, args.save_dir, latest_step, optimizer=optimizer)
        start_step = metadata.get('step', latest_step)
        state[1] = optimizer.state
    else:
        print("🚀 Starting training from scratch")

    # Log file
    os.makedirs(args.save_dir, exist_ok=True)
    log_path = os.path.join(args.save_dir, 'train.log')
    log_mode = 'a' if latest_step is not None else 'w'
    log_file = open(log_path, log_mode)

    def log(msg):
        print(msg)
        log_file.write(msg + '\n')
        log_file.flush()

    # Per-layer gamma for MoE
    gamma_min = 0.0001
    gamma_max = 0.0005
    moe_layer_indices = [
        i for i, layer in enumerate(model.layers)
        if hasattr(layer.ffn, "adaptive_routing") and layer.ffn.adaptive_routing
    ]
    n_moe = len(moe_layer_indices)
    per_layer_gamma = {}
    for rank, idx in enumerate(moe_layer_indices):
        t = rank / max(n_moe - 1, 1)
        per_layer_gamma[idx] = gamma_min + (gamma_max - gamma_min) * (t ** 0.5)
    log(f"🔧 Per-layer gamma: {gamma_min} ~ {gamma_max} (sqrt, {n_moe} MoE layers)")

    log(f"\n{'='*60}")
    log(f"🚀 Training Started (step={start_step})")
    log(f"{'='*60}")

    # Training loop
    global_step = start_step
    accum_count = 0
    accum_loss = 0.0
    log_loss = 0.0
    log_tokens = 0
    log_start = time.time()
    acc_grads = zero_grads()

    data_iter = iter(dataloader)

    try:
        while global_step < args.max_steps:
            try:
                input_ids, labels = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                input_ids, labels = next(data_iter)

            # Forward + Backward
            loss, acc_grads = compiled_step(input_ids, labels, acc_grads)
            mx.eval(loss, acc_grads)
            accum_loss += loss.item()
            accum_count += 1

            # Update
            if accum_count >= args.grad_accum:
                current_lr = get_lr(global_step, args.warmup_steps, args.max_steps, args.lr, args.min_lr_ratio)
                optimizer.learning_rate = current_lr

                grad_norm, acc_grads = do_update(acc_grads, mx.array(args.grad_accum, dtype=mx.float32))
                mx.eval(acc_grads, model.parameters(), optimizer.state, grad_norm)

                step_loss = accum_loss / args.grad_accum
                grad_norm_val = grad_norm.item()

                # Expert stats & bias update
                expert_stats = model.get_expert_stats()
                model.update_expert_biases(gamma=per_layer_gamma)
                _sync_compile_state(state[0], model.state)

                global_step += 1
                tokens_this_step = args.max_seq_len * args.batch_size * args.grad_accum
                log_loss += step_loss
                log_tokens += tokens_this_step

                accum_loss = 0.0
                accum_count = 0

                # Log
                if global_step % args.log_every == 0:
                    elapsed = time.time() - log_start
                    avg_loss = log_loss / args.log_every
                    tok_per_sec = log_tokens / elapsed if elapsed > 0 else 0

                    log(f"Step {global_step:>6d} | Loss: {avg_loss:.4f} | "
                        f"LR: {current_lr:.2e} | GradNorm: {grad_norm_val:.2f} | "
                        f"Tok/s: {tok_per_sec:,.0f}")

                    # Expert stats (샘플링)
                    if expert_stats:
                        all_keys = list(expert_stats.keys())
                        n = len(all_keys)
                        pick = sorted(set([0, n//4, n//2, 3*n//4, n-1]))
                        show_keys = [all_keys[i] for i in pick]
                        parts = []
                        for k in show_keys:
                            s = expert_stats[k]
                            parts.append(f"{k}[{s['min'].item():.1f}-{s['max'].item():.1f}%]")
                        log(f"  Expert: {' | '.join(parts)}")

                    log_loss = 0.0
                    log_tokens = 0
                    log_start = time.time()

                # Save
                if global_step % args.save_every == 0:
                    save_checkpoint(model, global_step, step_loss, current_lr,
                                    args.save_dir, optimizer=optimizer)
                    cleanup_checkpoints(args.save_dir, args.keep_checkpoints)

    except KeyboardInterrupt:
        log("\n⚠️  Training interrupted by user")

    log(f"\n✅ Training completed! Total steps: {global_step:,}")
    log_file.close()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='BORA-1.1B-A0.4B Pre-training (JSONL)')

    # Data
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Directory containing JSONL files')

    # Training
    parser.add_argument('--batch-size', type=int, default=DEFAULT_CONFIG['batch_size'])
    parser.add_argument('--grad-accum', type=int, default=DEFAULT_CONFIG['grad_accum'])
    parser.add_argument('--max-seq-len', type=int, default=DEFAULT_CONFIG['max_seq_len'])
    parser.add_argument('--max-steps', type=int, default=DEFAULT_CONFIG['max_steps'])
    parser.add_argument('--lr', type=float, default=DEFAULT_CONFIG['lr'])
    parser.add_argument('--min-lr-ratio', type=float, default=DEFAULT_CONFIG['min_lr_ratio'])
    parser.add_argument('--warmup-steps', type=int, default=DEFAULT_CONFIG['warmup_steps'])
    parser.add_argument('--weight-decay', type=float, default=DEFAULT_CONFIG['weight_decay'])
    parser.add_argument('--grad-clip', type=float, default=DEFAULT_CONFIG['grad_clip'])
    parser.add_argument('--z-loss-weight', type=float, default=DEFAULT_CONFIG['z_loss_weight'])
    parser.add_argument('--seed', type=int, default=DEFAULT_CONFIG['seed'])

    # Checkpoint
    parser.add_argument('--save-dir', type=str, default='./checkpoints')
    parser.add_argument('--save-every', type=int, default=DEFAULT_CONFIG['save_every'])
    parser.add_argument('--log-every', type=int, default=DEFAULT_CONFIG['log_every'])
    parser.add_argument('--keep-checkpoints', type=int, default=DEFAULT_CONFIG['keep_checkpoints'])

    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()

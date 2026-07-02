import json
import math
import os
import time
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import TextDataset
from models.gpt2 import GPT2, GPT2Config

# I/O
run_name = os.environ.get('RUN_NAME', datetime.now().strftime('%Y%m%d_%H%M%S'))
out_dir = os.path.join('checkpoints', 'gpt2', run_name)
tensorboard_dir = os.path.join('runs', 'gpt2', run_name)
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False  # if True, script exits right after the first eval
always_save_checkpoint = True  # if True, always save a checkpoint after each eval
init_from = 'gpt2'  # 'scratch' or 'resume' or 'gpt2*'

# data
dataset = 'wikitext-103'
gradient_accumulation_steps = 8
batch_size = 16  # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024

# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.1  # for pretraining 0 is good, for finetuning try 0.1+
bias = False  # do we use bias inside LayerNorm and Linear layers?

# adamw optimizer
learning_rate = 6e-5  # max learning rate
max_iters = 50000  # total number of training iterations
weight_decay = 1e-2
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0

# learning rate decay settings
decay_lr = True  # whether to decay the learning rate
warmup_iters = 2000  # how many steps to warm up for
lr_decay_iters = 50000
min_lr = 5e-6

# system
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True

# ---------------------------------------------------------------------------
# setup
os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337)
writer = SummaryWriter(log_dir=tensorboard_dir)

hyperparams = {
    'dataset': dataset,
    'batch_size': batch_size,
    'block_size': block_size,
    'gradient_accumulation_steps': gradient_accumulation_steps,
    'n_layer': n_layer,
    'n_head': n_head,
    'n_embd': n_embd,
    'dropout': dropout,
    'learning_rate': learning_rate,
    'max_iters': max_iters,
    'weight_decay': weight_decay,
    'beta1': beta1,
    'beta2': beta2,
    'grad_clip': grad_clip,
    'decay_lr': decay_lr,
    'warmup_iters': warmup_iters,
    'lr_decay_iters': lr_decay_iters,
    'min_lr': min_lr,
    'device': device,
    'dtype': dtype,
    'compile': compile,
    'init_from': init_from,
}
writer.add_text('config/hparams', json.dumps(hyperparams, indent=2, sort_keys=True))

device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == 'cuda' else torch.nullcontext()

# ---------------------------------------------------------------------------
# data
data_dir = os.path.join('data', dataset)

def get_dataloader(split):
    bin_path = os.path.join(data_dir, f'{split}.bin')
    ds = TextDataset(bin_path, block_size)
    return DataLoader(ds, batch_size=batch_size, shuffle=(split == 'train'),
                      pin_memory=(device_type == 'cuda'), num_workers=4)

train_loader = get_dataloader('train')
val_loader = get_dataloader('val')
train_iter = iter(train_loader)

def get_batch(loader_iter, loader):
    global train_iter
    try:
        x, y = next(loader_iter)
    except StopIteration:
        loader_iter = iter(loader)
        x, y = next(loader_iter)
        if loader is train_loader:
            train_iter = loader_iter
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)

# ---------------------------------------------------------------------------
# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=50257, dropout=dropout)
iter_num = 0
best_val_loss = 1e9

if init_from == 'scratch':
    print("Initializing a new model from scratch")
    config = GPT2Config(**model_args)
    model = GPT2(config)

elif init_from == 'resume':
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    print(f"Resuming training from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint['model_args'][k]
    config = GPT2Config(**model_args)
    model = GPT2(config)
    model.load_state_dict(checkpoint['model'])
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    model = GPT2.from_pretrained(init_from, override_args=dict(dropout=dropout))
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)

model.to(device)

# ---------------------------------------------------------------------------
# optimizer — apply weight decay only to 2-D params (weights, not biases/norms)
def configure_optimizer(model, weight_decay, learning_rate, betas):
    decay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0},
    ]
    return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)

optimizer = configure_optimizer(model, weight_decay, learning_rate, (beta1, beta2))
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])

scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

if compile:
    print("Compiling the model (takes ~1 min)...")
    model = torch.compile(model)

# ---------------------------------------------------------------------------
# helpers

def get_lr(it):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split, loader in [('train', train_loader), ('val', val_loader)]:
        losses = torch.zeros(eval_iters)
        it = iter(loader)
        for k in range(eval_iters):
            try:
                x, y = next(it)
            except StopIteration:
                it = iter(loader)
                x, y = next(it)
            x, y = x.to(device), y.to(device)
            with ctx:
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

def save_checkpoint(iter_num, best_val_loss, path):
    raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    checkpoint = {
        'model': raw_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_args': model_args,
        'iter_num': iter_num,
        'best_val_loss': best_val_loss,
    }
    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")

# ---------------------------------------------------------------------------
# training loop
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
best_path = os.path.join(out_dir, 'best.pt')

print(f"Training on {device} | dtype={dtype} | steps={max_iters}")
t0 = time.time()

while iter_num < max_iters:
    # update learning rate
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate and checkpoint
    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        writer.add_scalars('loss', {'train': losses['train'], 'val': losses['val']}, iter_num)
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                save_checkpoint(iter_num, best_val_loss, best_path)
        
        # Always save latest checkpoint
        save_checkpoint(iter_num, best_val_loss, ckpt_path)

    if eval_only:
        break

    # forward + backward with gradient accumulation
    optimizer.zero_grad(set_to_none=True)
    for micro_step in range(gradient_accumulation_steps):
        x, y = get_batch(train_iter, train_loader)
        with ctx:
            _, loss = model(x, y)
            loss = loss / gradient_accumulation_steps
        scaler.scale(loss).backward()

    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    scaler.step(optimizer)
    scaler.update()

    t1 = time.time()
    if iter_num % log_interval == 0:
        lossf = loss.item() * gradient_accumulation_steps
        print(f"iter {iter_num}: loss {lossf:.4f}, lr {lr:.2e}, time {(t1-t0)*1000:.1f}ms")
        writer.add_scalar('train/loss', lossf, iter_num)
        writer.add_scalar('train/lr', lr, iter_num)
    t0 = t1
    iter_num += 1

writer.close()

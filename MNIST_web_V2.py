from __future__ import annotations

from collections import OrderedDict
from enum import Enum
from time import monotonic
from torch.nn import (Dropout, Linear, Module, Sequential, SiLU) 
from torch.optim import (AdamW, Optimizer)
from torch.optim.lr_scheduler import (OneCycleLR, _LRScheduler)
from torch.utils.data import (DataLoader, random_split)
from torch.cuda.amp import (autocast, GradScaler)
from torchsummary import summary
from torchvision.datasets import MNIST
from torchvision.transforms import (Compose, Lambda, RandomRotation, ToTensor)
from tqdm import tqdm

import torch
import torch.nn.functional as F
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.autograd.profiler.profile(enabled=False)
torch.autograd.profiler.emit_nvtx(enabled=False)
torch.autograd.set_detect_anomaly(mode=False)


class Split(Enum):
    TRAIN: str = "Train"
    VALID: str = "Valid"
    TEST : str = "Test"


def fit_step(model: Module, loader: DataLoader, optim: Optimizer, scheduler: _LRScheduler, scaler: GradScaler, split: Split, device : str) -> tuple[float, float]:
    train = split == Split.TRAIN
    total_loss, total_acc = 0.0, 0.0
    
    model.train(mode=train)
    with torch.inference_mode(mode=not train), tqdm(loader, desc=split.value) as pbar:
        for x, l in pbar:
            x, l = x.to(device), l.cuda(device)

            with autocast():
                logits = model(x)
                loss = F.cross_entropy(logits, l)
                acc = (logits.argmax(dim=1) == l).sum() / x.size(0)

            if split == Split.TRAIN:
                optim.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optim)
                scheduler.step()
                scaler.update()

            total_loss += loss.item() / len(loader)
            total_acc += acc.item() / len(loader)
            pbar.set_postfix(loss=f"{total_loss:.2e}", acc=f"{total_acc * 100:.2f}%", lr="None" if scheduler is None else f"{scheduler.get_last_lr()[0]:.2e}")
    
    return total_loss, total_acc


def fit(model: Module, epochs: int, loaders: dict[Split, DataLoader], device : str) -> None:
    summary(model, (28 * 28, ))
    
    optim = AdamW(model.parameters())
    scheduler = OneCycleLR(optim, max_lr=1e-3, total_steps=len(loaders[Split.TRAIN]) * epochs)
    scaler = GradScaler()

    for _ in tqdm(range(epochs), desc="Epoch"):
        train_loss, train_acc = fit_step(model, loaders[Split.TRAIN], optim, scheduler, scaler, Split.TRAIN,device=device)
        valid_loss, valid_acc = fit_step(model, loaders[Split.VALID], None, None, scaler, Split.VALID,device=device)
    test_loss, test_acc = fit_step(model, loaders[Split.TEST], None, None, scaler, Split.TEST,device=device)
    print("----------------------------------")
    print(f"[TRAIN] loss: {train_loss:.2e} acc: {train_acc*100:.2f}%")
    print(f"[VALID] loss: {valid_loss:.2e} acc: {valid_acc*100:.2f}%")
    print(f"[TEST]  loss: {test_loss :.2e} acc: {test_acc *100:.2f}%")
    print("----------------------------------")


@torch.inference_mode()
def benchmark_step(model: Module, input_size: tuple, n: int, batch_size: int, device: torch.device, dtype: torch.dtype) -> None:
    model = model.to(device=device, dtype=dtype)
    x = torch.rand(*input_size).to(device=device, dtype=dtype)
    dts = []
    for _ in tqdm(range(n), desc=f"Benchmark batch_size: {batch_size} device: {device} dtype: {dtype}"):
        t_start = monotonic()
        model(x)
        dts.append(monotonic() - t_start)
    min_dt, avg_dt, max_dt = min(dts), sum(dts) / len(dts), max(dts)
    print("----------------------------------")
    print(f"Parameters batch_size: {batch_size} device: {device} dtype: {dtype}")
    print(f"min: {min_dt * 1_000:.2f} ms  avg: {avg_dt * 1_000:.2f} ms  max: {max_dt * 1_000:.2f} ms")
    print("----------------------------------")
    

@torch.inference_mode()
def benchmark(model: Module, input_size: tuple, n: int) -> None:
    model = model.eval()
    for device in [torch.device("cpu"), torch.device("cuda:0")]:
        for dtype in [torch.float32, torch.float16] if device == torch.device("cuda:0") else [torch.float32]:
            for batch_size in [2 ** i for i in range(8)]:
                benchmark_step(model, input_size, n, batch_size, device, dtype)


Module.fit = fit
Module.benchmark = benchmark


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    sets = {
        Split.TRAIN: MNIST("/tmp/mnist", download=True, train=True, transform=Compose([ToTensor(), RandomRotation(45), Lambda(lambda x: x.reshape(-1))])),
        **{s: d for s, d in zip([Split.VALID, Split.TEST], random_split(MNIST("/tmp/mnist", download=True, train=False, transform=Compose([ToTensor(), Lambda(lambda x: x.reshape(-1))])), [0.5, 0.5]))},
    }

    loaders = {
        Split.TRAIN: DataLoader(sets[Split.TRAIN], batch_size=256, shuffle=True,  pin_memory=True, drop_last=True,  num_workers=22),
        Split.VALID: DataLoader(sets[Split.VALID], batch_size=256, shuffle=False, pin_memory=True, drop_last=False, num_workers=22),
        Split.TEST : DataLoader(sets[Split.TEST],  batch_size=256, shuffle=False, pin_memory=True, drop_last=False, num_workers=22),
    }

    model = Sequential(OrderedDict({
        "i": Sequential(OrderedDict({"l1": Linear(28 * 28, 256), "d1": Dropout(0.2), "a1": SiLU()})),
        "h": Sequential(OrderedDict({"l2": Linear(    256, 256), "d2": Dropout(0.2), "a2": SiLU()})),
        "o": Linear(256, 10),
    })).to(device)
    model.o.weight.data.normal_(0, 0.02)
    model.o.bias.data.zero_()

    model.fit(5, loaders,device = device)
    model.benchmark((28 * 28, ), 100)
    torch.save(model.state_dict(), "mnist.pt")
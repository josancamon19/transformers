import argparse
import enum
import random
from einops import rearrange
import torch
import os
import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import (
    ParaphraseDetectionDataset,
    SonnetsDataset,
    load_paraphrase_data,
)
from evaluation import model_eval_paraphrase
from optimizer import AdamW
from peft import LoraConfig, TaskType, get_peft_model
import torch.distributed as dist
from torch.utils.data import DistributedSampler
import time
import functools
from transformers import GPT2Tokenizer
from torch import autocast
import wandb
from sacrebleu.metrics import CHRF
from torch.optim.lr_scheduler import OneCycleLR

TQDM_DISABLE = False

hf_cache_dir = "./.cache/huggingface"
os.makedirs(hf_cache_dir, exist_ok=True)


def timeit(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.2f} seconds to execute")
        return result

    return wrapper


# Fix the random seed
@timeit
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


@timeit
def cache_model():
    print("Downloading and caching GPT2 tokenizer...")
    _ = GPT2Tokenizer.from_pretrained("gpt2", cache_dir=hf_cache_dir)
    # OpenAIGPT2Model.from_pretrained()
    print("Tokenizer cached successfully!")


def get_lora_config(inference: bool, r=32):
    return LoraConfig(
        target_modules=[
            "query",
            "key",
            "value",
            "attention_dense",
            # if only interm_dense, has issue with interm*4, instead of custom *3 of ours, why?
            "gpt.gpt_layers.*.interm_dense",
            "gpt.gpt_layers.*.out_dense",
        ],
        task_type=TaskType.CAUSAL_LM,  # semantic label, not functional
        inference_mode=inference,
        r=r,
        lora_alpha=r * 2,
        lora_dropout=0.1,
    )


@timeit
def get_train_datasets(
    args,
    is_distributed: bool = False,
    rank: int = None,
):
    if args.model == "paraphrase":
        train_data = load_paraphrase_data(args.para_train)
        dev_data = load_paraphrase_data(args.para_dev)
        train_data = ParaphraseDetectionDataset(train_data, args)
        dev_data = ParaphraseDetectionDataset(dev_data, args)
    else:
        train_data = SonnetsDataset(args.sonnet_path)
        dev_data = SonnetsDataset(args.held_out_sonnet_dev_true_path)

    if is_distributed:
        if rank == 0:  # print only once
            world_size = torch.cuda.device_count()
            print(f"Distributed: {len(train_data)} total samples")
            print(f"Distributed: ~{len(train_data) // world_size} samples per GPU")
            print(
                f"Distributed: ~{(len(train_data) // world_size) // args.batch_size} batches per GPU"
            )
            # print(f"Distributed: Global effective batch size: {args.batch_size * world_size}")
    else:
        print(f"Single GPU: {len(train_data)} samples")
        print(f"Single GPU: {len(train_data) // args.batch_size} batches total")

    # handle multiple processes
    dev_sampler = (
        DistributedSampler(dev_data, shuffle=False) if is_distributed else None
    )
    train_sampler = (
        DistributedSampler(train_data, shuffle=True) if is_distributed else None
    )

    train_dataloader = DataLoader(
        train_data,  # [:10000],
        shuffle=(train_sampler is None),
        batch_size=args.batch_size,
        collate_fn=train_data.collate_fn,
        sampler=train_sampler,
    )
    dev_dataloader = DataLoader(
        dev_data,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=dev_data.collate_fn,
        sampler=dev_sampler,
    )
    return train_dataloader, dev_dataloader


def get_model_and_optimizer(args, device, model_class):
    model = model_class(args)
    if args.peft:
        model = get_peft_model(model, get_lora_config(False))
        model.print_trainable_parameters()

    if args.continue_prev_run:
        saved = torch.load(args.filepath, weights_only=False)
        model.load_state_dict(saved["model"])
        print("get_model_and_optimizer, pre-loaded:", args.filepath)

    model = model.to(device, dtype=torch.bfloat16 if args.use_bf16 else None)
    if args.model == ModelTarget.paraphrase:
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    else:
        optimizer = AdamW(
            [
                {"params": model.gpt.parameters(), "lr": args.lr},
                {
                    "params": model.classifier.parameters(),
                    "lr": args.lr * 50,
                },
            ],
            lr=args.lr,
        )
    return model, optimizer


def save_model(model, optimizer, args):
    save_info = {
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
        "args": args,
        "system_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.random.get_rng_state(),
    }

    torch.save(save_info, args.filepath)
    if args.peft:
        model.save_pretrained(f"./.models/{args.model}")
    print(f"saved model to {args.filepath}")


def get_wandb_run(args):
    return wandb.init(
        entity="josancamon19-cifrato",
        project="cs224n",
        config={
            "target": f"{args.model}",
            "tags": [args.model],
            "learning_rate": args.lr,
            "peft": args.peft,
            "distributed": False,
            "base_model": args.model_size,
            "gradient_accumulation": args.gradient_accumulation,
            "epochs": args.epochs,
        },
    )


def compute_validation_perplexity(model, dev_dataloader, device):
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for sample in dev_dataloader:
            b_ids = sample["token_ids"].to(device)
            b_mask = sample["attention_mask"].to(device)
            labels = b_ids[:, 1:].contiguous().flatten()
            logits = model(b_ids, b_mask)
            logits = logits[:, :-1].contiguous()
            logits = rearrange(logits, "b t d -> (b t) d")

            loss = F.cross_entropy(logits, labels, reduction="sum")
            valid_tokens = (labels != model.tokenizer.pad_token_id).sum().item()

            total_loss += loss.item()
            total_tokens += valid_tokens

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return perplexity


def validation_chrf_score(
    args,
    model,
    device,
    dev_input_dataset: SonnetsDataset,
    dev_true_dataset: SonnetsDataset,
):
    predictions = []
    with torch.inference_mode():
        for sample in dev_input_dataset:
            encoding = model.tokenizer(
                sample[1],
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            output = model.generate(
                encoding.to(device)["input_ids"],
                temperature=args.temperature,
                top_p=args.top_p,
            )
            predictions.append(output[1])
    chrf = CHRF()
    true_sonnets = [x[1] for x in dev_true_dataset]
    print("validation_chrf_score")
    print("true_sonnets[0]\n-\n", true_sonnets[0])
    print("generated_sonnets[0]\n-\n", predictions[0])
    print("\n")
    max_len = min(len(true_sonnets), len(predictions))
    true_sonnets = true_sonnets[:max_len]
    generated_sonnets = predictions[:max_len]

    # compute chrf
    chrf_score = chrf.corpus_score(generated_sonnets, [true_sonnets])
    return float(chrf_score.score)


def train_eval(
    rank,
    args,
    model,
    optimizer,
    dev_dataloader,
    device,
    epoch,
    train_loss,
    best_dev_acc,
    wandb_run,
):
    if args.model == "paraphrase":
        if rank is None or rank == 0:
            dev_acc, dev_f1, y_pred, y_true, _ = model_eval_paraphrase(
                dev_dataloader, model, device, args.use_bf16
            )
            print("model_eval_paraphrase")
            print("preds:", y_pred[:20])
            print("true: ", [int(x) for x in y_true[:20]])
            if dev_acc > best_dev_acc:
                best_dev_acc = dev_acc
                model_to_save = model.module if hasattr(model, "module") else model
                save_model(model_to_save, optimizer, args)
            if wandb_run:
                wandb_run.log({"best_dev_acc": best_dev_acc, "train_loss": train_loss})
            print(
                f"Epoch {epoch}: train loss :: {train_loss:.3f}, dev acc :: {dev_acc:.3f}"
            )
    else:
        # if epoch % 2 == 0:  # too few data, evaluate every 2 training runs
        # return best_dev_acc
        model.eval()

        print(f"Epoch {epoch}: train loss :: {train_loss:.3f}")
        print("evaluating sonnets:")

        # TODO: should compute excluding first 3 input lines?
        dev_input_dataset = SonnetsDataset(args.held_out_sonnet_dev_path)
        chrf_score = validation_chrf_score(
            args, model, device, dev_input_dataset, dev_dataloader.dataset
        )
        val_perplexity = compute_validation_perplexity(model, dev_dataloader, device)
        print("sonnets chrf_score, val_perplexity:", chrf_score, val_perplexity)

        if val_perplexity < best_dev_acc:
            model_to_save = model.module if hasattr(model, "module") else model
            save_model(model_to_save, optimizer, args)
            best_dev_acc = val_perplexity

        if wandb_run:
            wandb_run.log(
                {
                    "train_loss": train_loss,
                    "chrf_score": chrf_score,
                    "val_perplexity": val_perplexity,
                }
            )
    return best_dev_acc


def train_epoch(
    args,
    model,
    epoch,
    device,
    optimizer,
    train_dataloader,
    dev_dataloader,
    best_dev_acc,
    rank=None,
    train_sampler=None,
    gradient_accumulation_steps=1,
    use_bf16=False,
    wandb_run=None,
):
    # is necessary to make shuffling work properly across multiple epochs.
    # # Otherwise, the same ordering will be used in each epoch.
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)

    model.train()
    train_loss = 0
    num_batches = 0
    optimizer.zero_grad()

    for batch_idx, sample in enumerate(
        tqdm(train_dataloader, desc=f"train-{epoch}", disable=TQDM_DISABLE)
    ):
        # Get the input and move it to the gpu (I do not recommend training this model on CPU).
        b_ids, b_mask, labels = (
            sample["token_ids"].to(device),
            sample["attention_mask"].to(device),
            # .torch empty for sonnets
            sample.get("labels", torch.empty((0, 0))).flatten().to(device),
        )
        if args.model == "sonnet":
            # Ignore the first token to compose the labels.
            labels = b_ids[:, 1:].contiguous().flatten()

        with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
            logits = model(b_ids, b_mask)
            if args.model == "sonnet":
                # Ignore the last prediction in the sequence.
                logits = rearrange(logits[:, :-1].contiguous(), "b t d -> (b t) d")

            loss = F.cross_entropy(logits, labels, reduction="mean")
            loss = loss / gradient_accumulation_steps

        loss.backward()

        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        train_loss += loss.item() * gradient_accumulation_steps
        num_batches += 1

        if num_batches % gradient_accumulation_steps != 0:
            torch.cuda.empty_cache()

    if num_batches % gradient_accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    train_loss = train_loss / num_batches

    # TODO: docs
    if rank is not None:
        train_loss_tensor = torch.tensor(train_loss).cuda()
        dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
        train_loss = train_loss_tensor.item() / dist.get_world_size()

    best_dev_acc = train_eval(
        rank,
        args,
        model,
        optimizer,
        dev_dataloader,
        device,
        epoch,
        train_loss,
        best_dev_acc,
        wandb_run,
    )

    if rank is not None:
        best_dev_acc_tensor = torch.tensor(best_dev_acc).cuda()
        dist.broadcast(best_dev_acc_tensor, src=0)
        best_dev_acc = best_dev_acc_tensor.item()

    return best_dev_acc


def train(args, model_class):
    """Train GPT-2 for paraphrase detection on the Quora dataset."""

    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
    train_dataloader, dev_dataloader = get_train_datasets(args)
    model, optimizer = get_model_and_optimizer(args, device, model_class)
    best_dev_acc = 0 if args.model == "paraphrase" else float("inf")

    wandb_run = get_wandb_run(args)
    # scheduler = OneCycleLR(
    #     optimizer,
    #     max_lr=[args.lr * 3, args.lr * 50],  # Different max for each param group
    #     epochs=args.epochs,
    #     steps_per_epoch=len(train_dataloader),
    #     pct_start=0.2,  # 20% warmup
    #     anneal_strategy="cos",
    # )

    for epoch in range(args.epochs):
        best_dev_acc = train_epoch(
            args,
            model,
            epoch,
            device,
            optimizer,
            train_dataloader,
            dev_dataloader,
            best_dev_acc,
            None,
            None,
            args.gradient_accumulation,
            args.use_bf16,
            wandb_run,
            # None,
        )


def dpo():
    # - before collect train loss to wandb and visualize loss curves
    # - anything you can infer related to it?
    # -- sonnet has no dev set, how to, how to estimate how really good/bad is?
    # -- need a separate dev dataset?

    # - collect a bunch of input prompts (first few lines as input, or instruction tunned) (generate with GPT)
    # - sample outputs from the model (), multiple temperature
    # - label those sample pairs with GPT (pref, not pref)
    # - with (x, yw), (x, yl) train the model
    # - - save0. model to hf, do inference with runpod vllm, or try serving yourself directly with vllm

    # - train loop
    # - loss computed: -log(sigmoid(beta * (log*model(yw | x) - log*model(yl | x))))
    # - backprop.
    pass


def train_dist(rank, args, model_class):
    try:
        world_size = torch.cuda.device_count()
        dist.init_process_group(
            "nccl",  # gloo, never used, test things on cpu
            init_method="tcp://localhost:12355",
            # rendevouz, server python spins up, distribute works to other processes
            rank=rank,
            world_size=world_size,
        )
        torch.cuda.set_device(rank)
        train_dataloader, dev_dataloader = get_train_datasets(args, True, rank)
        device = torch.device(f"cuda:{rank}")

        model, optimizer = get_model_and_optimizer(args, device, model_class)
        # tries to schedule things ahead of time, mark operations ready to go, but it some vars never receive gradients
        # it will keep waiting for that. (find_unused_parameters = True)
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[rank], find_unused_parameters=True
        )

        best_dev_acc = 0

        for epoch in range(args.epochs):
            best_dev_acc = train_epoch(
                args,
                model,
                epoch,
                device,
                optimizer,
                train_dataloader,
                dev_dataloader,
                best_dev_acc,
                rank,
                train_dataloader.sampler,
                args.gradient_accumulation,
                args.use_bf16,
            )
            dist.barrier()
    except Exception as e:
        print(f"Error in rank {rank}: {e}")
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


class ModelTarget(enum.Enum):
    paraphrase = "paraphrase"
    sonnet = "sonnet"


def get_args(model: ModelTarget):
    parser = argparse.ArgumentParser()

    # paratrain
    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")
    parser.add_argument(
        "--para_dev_out", type=str, default="predictions/para-dev-output.csv"
    )
    parser.add_argument(
        "--para_test_out", type=str, default="predictions/para-test-output.csv"
    )
    # sonnet
    parser.add_argument("--sonnet_path", type=str, default="data/sonnets.txt")
    # test inputs
    parser.add_argument(
        "--held_out_sonnet_path", type=str, default="data/sonnets_held_out.txt"
    )
    # dev inputs
    parser.add_argument(
        "--held_out_sonnet_dev_path", type=str, default="data/sonnets_held_out_dev.txt"
    )
    # dev labels
    parser.add_argument(
        "--held_out_sonnet_dev_true_path",
        type=str,
        default="data/TRUE_sonnets_held_out_dev.txt",
    )
    # dev output sonnet gen from test inputs
    parser.add_argument(
        "--sonnet_out", type=str, default="predictions/generated_sonnets.txt"
    )

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--use_gpu", action="store_true", default=False)
    parser.add_argument("--peft", action="store_true", default=False)
    parser.add_argument("--distributed", action="store_true", default=False)
    parser.add_argument("--gradient_accumulation", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
    parser.add_argument("--temperature", type=float, default=1.2)
    parser.add_argument("--top_p", type=float, default=0.8)

    parser.add_argument("--continue_prev_run", action="store_true", default=False)
    parser.add_argument(
        "--model_size",
        type=str,
        help="The model size as specified on hugging face. DO NOT use the xl model.",
        choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"],
        default="gpt2",
    )

    args = parser.parse_args()
    args.cache_dir = hf_cache_dir
    seed_everything(args.seed)
    args.use_bf16 = check_bf16_support()
    args.model = model.value
    args.filepath = f"./.models/{args.model}/{args.model_size}-{args.lr}.pt"
    os.makedirs(f"./.models/{args.model}", exist_ok=True)
    add_arguments(args)
    return args


def add_arguments(args):
    """Add arguments that are deterministic on model size."""
    if args.model_size == "gpt2":
        args.d = 768
        args.l = 12
        args.num_heads = 12
    elif args.model_size == "gpt2-medium":
        args.d = 1024
        args.l = 24
        args.num_heads = 16
    elif args.model_size == "gpt2-large":
        args.d = 1280
        args.l = 36
        args.num_heads = 20
    elif args.model_size == "gpt2-xl":
        args.d = 1600
        args.l = 48
        args.num_heads = 25
    else:
        raise Exception(f"{args.model_size} is not supported.")
    return args


def check_bf16_support():
    if torch.cuda.is_available():
        # Check if GPU supports BF16
        major, minor = torch.cuda.get_device_capability()
        if major >= 8:  # Ampere (RTX 30xx) and newer
            print("✅ BF16 supported on this GPU")
            return True
        else:
            print("❌ BF16 not supported on this GPU (need Ampere or newer)")
            return False
    return False

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import GPT2Tokenizer
from src.transformer import Transformer
from torch.optim import AdamW
import torch.nn.functional as F
import os

os.makedirs("./models", exist_ok=True)


class PretrainDataset(Dataset):
    def __init__(self, tokenizer: GPT2Tokenizer, dataset_path: str, max_sequence_length: int):
        self.samples = []
        self.dataset_path = dataset_path
        # with open(dataset_path, "rb") as f:
        #     self.samples = f.read().decode("utf-8", errors="ignore").split("<|endoftext|>")
        # TODO: missing a lot of tokens when truncating, many > max sequence length
        with open(dataset_path, "r", encoding="utf-8", errors="ignore") as f:
            current_pos = 0
            for line in f:
                if "<|endoftext|>" in line:
                    parts = line.split("<|endoftext|>")
                    for i, part in enumerate(parts[:-1]):  # Skip last empty part
                        if part.strip():
                            self.samples.append((current_pos, current_pos + len(part.encode("utf-8"))))
                        current_pos += len(part.encode("utf-8")) + len(b"<|endoftext|>")
                else:
                    if line.strip():
                        self.samples.append((current_pos, current_pos + len(line.encode("utf-8"))))
                    current_pos += len(line.encode("utf-8"))
        # print(f"found: {len(self.samples)}")
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        start_pos, end_pos = self.samples[idx]
        with open(self.dataset_path, "r", encoding="utf-8", errors="ignore") as f:
            f.seek(start_pos)
            text = f.read(end_pos - start_pos)
        return text.strip()
        # return self.samples[idx]

    def collate_fn(self, batch: list[str]):
        tokenized = self.tokenizer(
            batch,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_sequence_length,
        )
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
        }


def train():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    max_sequence_length = 1024
    batch_size = 16
    embedding_dim = 128
    num_layers = 2
    num_heads = 8

    train_dataset = PretrainDataset(tokenizer, "data/owt_train.txt", max_sequence_length)
    valid_dataset = PretrainDataset(tokenizer, "data/owt_valid.txt", max_sequence_length)
    train_dataloader = DataLoader(
        train_dataset, batch_size, shuffle=True, collate_fn=train_dataset.collate_fn, pin_memory=True
    )
    valid_dataloader = DataLoader(
        valid_dataset, batch_size, shuffle=False, collate_fn=valid_dataset.collate_fn, pin_memory=True
    )
    model = Transformer(tokenizer.vocab_size, max_sequence_length, embedding_dim, num_layers, num_heads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    epochs = 10
    optim = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    def compute_inputs_loss(batch):
        input_ids = batch["input_ids"][:, :-1].to(device)
        labels = batch["input_ids"][:, 1:].to(device)  # better way to slice
        # print("input_ids.shape, labels.shape:", input_ids.shape, labels.shape)
        attention_mask = batch["attention_mask"][:, :-1].to(device)
        output = model(input_ids, attention_mask)
        output_flatten = output.view(-1, output.shape[-1])
        labels = labels.contiguous().view(-1)
        # print("output, output_flatten, labels:", output.shape, output_flatten.shape, labels.shape)
        return F.cross_entropy(output_flatten, labels)

    best_valid_loss = float("inf")

    for i in range(epochs):
        train_loss = 0
        model.train()

        for batch in tqdm(train_dataloader, desc=f"train-epoch {i + 1}"):
            optim.zero_grad()
            loss = compute_inputs_loss(batch)
            print("loss:", loss.item())
            train_loss += loss.item()
            loss.backward()
            optim.step()

        train_loss = train_loss / len(train_dataloader)
        print(f"epoch {i + 1} train_loss: {train_loss}")

        valid_loss = 0
        model.eval()
        with torch.inference_mode():
            for batch in tqdm(valid_dataloader, desc=f"valid-epoch {i + 1}"):
                valid_loss += compute_inputs_loss(batch).item()

        valid_loss = valid_loss / len(valid_dataloader)
        print(f"epoch {i + 1} valid_loss: {valid_loss}")
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            data = {"model": model.state_dict(), "optimizer": optim.state_dict()}
            torch.save(data, f"./models/gpt2-epoch-{i + 1}.pt")


train()


import os
from collections import Counter

import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class Tokenizer:
    def __init__(self):
        self.vocab = {
            "<PAD>": 0,
            "<UNK>": 1,
            "<SOS>": 2,
        }
        self.inv_vocab = {}

    def build_vocab(self, captions):
        counter = Counter()

        for caption in captions:
            counter.update(caption.lower().split())

        for word in sorted(counter.keys()):
            self.vocab[word] = len(self.vocab)

        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def encode(self, text):
        tokens = [self.vocab["<SOS>"]]

        for word in text.lower().split():
            tokens.append(self.vocab.get(word, self.vocab["<UNK>"]))

        return tokens

    def decode(self, tokens):
        words = []

        for token in tokens:
            words.append(self.inv_vocab.get(token, "<UNK>"))

        return " ".join(words)

    def __len__(self):
        return len(self.vocab)

class Flickr8kDataset(Dataset):
    def __init__(self, image_dir, captions_df, tokenizer, transform=None):
        self.image_dir = image_dir
        self.captions_df = captions_df
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.captions_df)

    def __getitem__(self, idx):
        sample = self.captions_df.iloc[idx]

        image_name = sample["image"]
        caption = sample["caption"]

        image_path = os.path.join(self.image_dir, image_name)

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        caption = self.tokenizer.encode(caption)

        return image, caption

def collate_fn(batch):
    images, captions = zip(*batch)

    images = torch.stack(images)

    max_len = max(len(caption) for caption in captions)

    padded_captions = []
    attention_masks = []

    for caption in captions:
        mask = [1] * len(caption) + [0] * (max_len - len(caption))
        padded = caption + [0] * (max_len - len(caption))

        padded_captions.append(padded)
        attention_masks.append(mask)

    captions = torch.tensor(padded_captions)
    attention_masks = torch.tensor(attention_masks)

    return images, captions, attention_masks

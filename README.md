# SOC-multimodal-transformer-24B2725

by Vaibhav Kumar Singh

Building a multimodal transformer from scratch as part of the Seasons of Code, IIT Bombay project. This project focuses on understanding transformers mathematically and implementing every core component manually using PyTorch, from attention mechanisms to full vision-language models.

## Tasks Completed

### Task 0 : Foundations & Tensor Warmup

Set up the PyTorch environment and built intuition for core tensor operations, masking, softmax, LayerNorm, causal attention, and manual gradients while studying the fundamentals of transformers through _Attention Is All You Need_.

### Task 1 : First Attention Model

Implemented a character-level bigram language model and upgraded it with a single self-attention head from scratch, including causal masking, token embeddings, and autoregressive text generation.

### Task 2 : Full Transformer Decoder

Built a decoder-only transformer with multi-head attention, residual connections, LayerNorm, and feed-forward networks, while also deriving attention gradients mathematically and performing ablation experiments.

### Task 3 : Vision Transformer

Implemented a Vision Transformer (ViT) for image classification on CIFAR-10 by converting images into patch embeddings, adding positional encodings, and training transformer encoder blocks, followed by a comparison with a CNN baseline.

### Task 4 : Cross-Attention & Multimodal Fusion

Implemented cross-attention from scratch to enable information exchange between image and text embeddings, visualized attention maps, and explored how multimodal transformers align features across different modalities.

### Task 5 : CLIP-Style Contrastive Learning

Built a CLIP-style dual encoder consisting of a Vision Transformer and Transformer text encoder, implemented InfoNCE contrastive loss with a learnable temperature parameter, and trained the model to align image and text representations.

### Task 6 : Training on Flickr8k

Constructed an end-to-end vision-language pipeline using the Flickr8k dataset, including dataset preprocessing, tokenization, dataloaders, and a complete training loop to learn aligned image-text embeddings through contrastive learning.

## Repository Structure

```text
Vision-Language-Model/
│
├── TASK 0/                         # Tensor fundamentals & transformer basics
├── TASK 1/                         # Bigram language model + single-head attention
├── TASK 2/                         # Decoder-only Transformer
├── TASK 3/                         # Vision Transformer (ViT) for CIFAR-10
├── TASK 4/                         # Cross-Attention implementation
├── TASK 5/                         # CLIP-style contrastive learning
├── TASK 6/                         # Flickr8k training pipeline
│
├── README.md                       # Project overview
├── VLM_Mentee_Handbook_Tasks_0_1_2.pdf
├── VLM_Mentee_Handbook_Tasks_3_4.pdf
└── VLM_Mentee_Handbook_Tasks_5_6.pdf
```

---

## Applications

This project demonstrates the building blocks behind modern Vision-Language Models and their real-world applications:

- Image captioning and visual understanding
- Cross-modal image-text retrieval and semantic search
- Multimodal AI assistants and Visual Question Answering (VQA)
- Foundation for large Vision-Language Models such as CLIP, BLIP, Flamingo, and GPT-4V

---

## Tech Stack

- Python
- PyTorch
- Torchvision
- NumPy
- Pandas
- Matplotlib
- Pillow (PIL)

### Datasets

- Tiny Shakespeare
- Flickr8k
- CIFAR-10

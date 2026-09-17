# LLM From Scratch

A decoder-only Transformer language model implemented and pretrained from scratch using PyTorch.

The goal of this project is to build the complete language-model training pipeline, including:

- Dataset preparation and cleaning
- BPE tokenizer training
- Tokenization
- Decoder-only Transformer implementation
- Multi-head causal self-attention
- Pretraining with next-token prediction
- Validation and perplexity evaluation
- Checkpointing
- Text generation
- Instruction fine-tuning

## Current Progress

- [x] Environment setup
- [x] CUDA verification
- [x] Basic dataset preparation
- [x] BPE tokenizer
- [x] Tokenized dataset
- [x] Transformer architecture
- [ ] Pretraining
- [ ] Evaluation
- [ ] Text generation
- [ ] Instruction fine-tuning

## Hardware

Development environment:

- PyTorch
- CUDA
- NVIDIA GPU

The training configuration can be adjusted depending on available GPU memory.

## Project Structure

```text
.
├── data/
│   ├── raw/
│   └── cleaned/
├── tokenizer/
├── checkpoints/
├── prepare_data.py
├── requirements.txt
└── README.md
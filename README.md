# AI Infrastructure Knowledge Base

A structured collection of notes on AI infrastructure, covering GPU programming, distributed systems, inference optimization, reinforcement learning, and AI agents.

## Folder Structure

```
AI_Infra/
├── 00_AI_Infra.md          # Root index file
├── basic/                   # GPU & CUDA fundamentals
│   ├── 00_Basic.md
│   ├── 01_GPU_Fundamentals.md
│   ├── 02_CPU_GPU_Execution.md
│   ├── 03_CUDA_Advanced.md
│   ├── 04_Triton_Programming.md
│   ├── 05_Matrix_Multiplication.md
│   └── 06_FlashAttention_CS336.md
├── distributed/             # Distributed training
│   ├── 00_Distributed.md
│   ├── 01_Communication.md
│   ├── 02_Parallelism.md
│   └── 03_Training_Systems.md
├── inference/               # Inference optimization
│   ├── 00_Inference.md
│   ├── 01_KV_Cache.md
│   ├── 02_Batching.md
│   ├── 03_Speculative_Decoding.md
│   ├── 04_Quantization.md
│   ├── 05_Frameworks.md
│   └── 06_Serving.md
├── rl/                      # Reinforcement learning
│   ├── 00_RL.md
│   ├── 01_Algorithms.md
│   ├── 02_Infrastructure.md
│   ├── 03_Frameworks.md
│   └── 04_Note.md
└── agent/                   # AI agents
    ├── 00_Agent.md
    ├── 01_Fundamentals.md
    ├── 02_Execution.md
    ├── 03_Multi_Agent.md
    ├── 04_Serving.md
    └── 05_Production.md
```

## Naming Conventions

### Folders
- Use lowercase with underscores: `folder_name/`
- Each folder represents a topic area

### Files
- Format: `XX_Topic_Name.md`
- `XX` is a two-digit number for ordering (00, 01, 02, ...)
- `00_*.md` is the index/overview file for each folder
- Use underscores to separate words: `01_GPU_Fundamentals.md`
- Use PascalCase for acronyms and proper nouns: `KV_Cache`, `GPU`, `CUDA`

### Index Files
- Each folder has a `00_FolderName.md` as the entry point
- Contains overview and links to subtopics
- Links to parent folder for navigation

## How to Use

This knowledge base is designed to work with [build-your-knowledge](https://github.com/ShanningZhuang/build-your-knowledge), a VitePress-based template that:

1. Auto-generates sidebar from folder structure
2. Supports math equations (KaTeX)
3. Deploys easily to Vercel

### Quick Start

1. Clone the template: `git clone https://github.com/ShanningZhuang/build-your-knowledge.git`
2. Copy this AI_Infra folder into the template
3. Run `npm install && npm run docs:dev`
4. Deploy to Vercel

## Topics Covered

| Topic | Description |
|-------|-------------|
| **Basic** | GPU architecture, CUDA programming, Triton, Flash Attention |
| **Distributed** | Collective communication, parallelism strategies, training systems |
| **Inference** | KV cache, batching, speculative decoding, quantization, serving |
| **RL** | Algorithms, infrastructure, frameworks for reinforcement learning |
| **Agent** | LLM agents, tool use, multi-agent systems, production deployment |

## Generating Content with AI

See [PROMPT.md](PROMPT.md) for a ready-to-use prompt when asking LLMs (Claude, GPT, etc.) to help generate or expand your knowledge base.

## Contributing

When adding new content:

1. Follow the naming conventions above
2. Update the index file (`00_*.md`) in the folder
3. Add appropriate links to parent/child documents

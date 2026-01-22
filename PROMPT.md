# Knowledge Base Generation Prompt

Use this prompt when asking LLMs to help generate or expand your knowledge base.

---

## System Prompt

```
You are helping me build a structured knowledge base. Follow these conventions strictly:

### Folder Structure
- Each topic area is a folder with lowercase and underscores: `topic_name/`
- Folders are NOT numbered (e.g., `agent/`, not `01_agent/`)
- Each folder contains markdown files numbered for reading order

### File Naming Convention
- Format: `XX_Topic_Name.md` where XX is a two-digit number (00, 01, 02, ...)
- `00_*.md` is always the index/overview file for the folder
- Use underscores between words: `01_GPU_Fundamentals.md`
- Keep acronyms uppercase: `KV_Cache`, `GPU`, `CUDA`, `RL`
- Examples:
  - `00_Agent.md` (index file)
  - `01_Fundamentals.md`
  - `02_Tool_Use.md`
  - `03_Multi_Agent.md`

### Index File Format (00_*.md)
Each folder must have an index file with this structure:

```markdown
# Topic Title

> Parent: [Parent Topic](../00_Parent.md)

## Overview

Brief description of what this topic covers.

## Topics

1. **Subtopic 1** - Brief description
2. **Subtopic 2** - Brief description
3. **Subtopic 3** - Brief description
```

### Content File Format (01_*.md, 02_*.md, etc.)
```markdown
# Topic Title

> Parent: [Parent Index](00_Index.md)

## Overview

Introduction to the topic.

## Section 1

Content...

## Section 2

Content...

## Related

- [Related Topic 1](01_Related.md) - Description
- [Related Topic 2](02_Another.md) - Description
```

### Linking Rules
- Always use relative paths: `./`, `../`
- Link to the exact filename with number prefix
- Include the .md extension
- Examples:
  - Same folder: `[Topic](01_Topic.md)`
  - Parent folder: `[Parent](../00_Parent.md)`
  - Sibling folder: `[Other](../other_folder/00_Other.md)`

### Content Guidelines
- Use clear, concise language
- Include code examples where relevant
- Use ASCII diagrams for architecture/flow visualization
- Add practical examples and use cases
- Structure content from fundamentals to advanced
```

---

## Example User Prompts

### Creating a New Topic Folder

```
Create a knowledge base folder for "model_serving" with the following structure:
- 00_Model_Serving.md (index)
- 01_Deployment_Basics.md
- 02_Load_Balancing.md
- 03_Scaling_Strategies.md
- 04_Monitoring.md

Follow the knowledge base conventions. The parent folder is AI_Infra (link to ../00_AI_Infra.md).
```

### Expanding an Existing Topic

```
Expand the file `inference/01_KV_Cache.md` with:
- More detailed explanation of multi-query attention (MQA)
- Grouped-query attention (GQA)
- Code examples in Python
- ASCII diagram showing memory layout

Follow the knowledge base conventions and maintain links to related files.
```

### Adding a New Subtopic

```
Add a new file `05_Caching_Strategies.md` to the `inference/` folder.
Cover: prompt caching, prefix caching, semantic caching.
Link it from 00_Inference.md and add cross-references to 01_KV_Cache.md.
Follow the knowledge base conventions.
```

### Restructuring Content

```
The file `basic/03_CUDA_Advanced.md` is too long. Split it into:
- 03_CUDA_Memory.md (memory management)
- 04_CUDA_Synchronization.md (sync primitives)
- 05_CUDA_Profiling.md (Nsight, profiling tools)

Update all internal links and renumber subsequent files.
Follow the knowledge base conventions.
```

---

## Quick Reference

| Element | Convention | Example |
|---------|------------|---------|
| Folder | lowercase_underscores | `model_serving/` |
| Index file | 00_FolderName.md | `00_Model_Serving.md` |
| Content file | XX_Topic_Name.md | `01_Deployment.md` |
| Acronyms | UPPERCASE | `KV_Cache`, `GPU` |
| Links | relative with .md | `[Link](../00_Parent.md)` |

---

## Full Example Structure

```
knowledge_base/
├── 00_Knowledge_Base.md        # Root index
├── topic_a/
│   ├── 00_Topic_A.md           # Index: links to ../00_Knowledge_Base.md
│   ├── 01_Fundamentals.md      # Links to 00_Topic_A.md
│   ├── 02_Core_Concepts.md
│   └── 03_Advanced.md
├── topic_b/
│   ├── 00_Topic_B.md
│   ├── 01_Basics.md
│   └── 02_Implementation.md
└── README.md                   # GitHub readme (not part of docs)
```

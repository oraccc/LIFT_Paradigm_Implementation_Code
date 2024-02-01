<div align="center">

# LIFT Paradigm Implementation

![Static Badge](https://img.shields.io/badge/Benchmark-HellaSwag-e07a5f)
![Static Badge](https://img.shields.io/badge/Benchmark-ARC-e07a5f) 
![Static Badge](https://img.shields.io/badge/Benchmark-TruthfulQA-e07a5f) 
![Static Badge](https://img.shields.io/badge/Benchmark-MMLU-e07a5f) 
![Static Badge](https://img.shields.io/badge/Benchmark-HumanEval-e07a5f) 
![Static Badge](https://img.shields.io/badge/Benchmark-MBPP-e07a5f)
![Static Badge](https://img.shields.io/badge/Model-StarCoder-f4d35e) 
![Static Badge](https://img.shields.io/badge/Model-Mistral-f4d35e)

:page_with_curl: [Paper](https://arxiv.org/abs/2312.11508)

</div>

**LIFT** (LLM Instruction Fusion Transfer) is novel and versatile paradigm designed to elevate the instruction quality to new heights, proposed in the paper "[Rethinking the Instruction Quality: LIFT is What You Need](https://arxiv.org/abs/2312.11508)"



### :key: LIFT's Key Concept: Data Distribution Transfer

LIFT is designed to combine the advantages of **data expansion** and **curation**, mitigating their shortcomings to generate a diverse and high-quality dataset while significantly reducing quantity. 

Our approach comprises two phases. Initially, we implement **"Dataset Distribution Expansion"** to broaden the data distribution and include more high-quality subspaces. Subsequently, we employ **"Dataset Variety and Quality Curation"** to eliminate redundancy, concentrating on enhancing the high-quality segments across overall data subspaces.

<p align="center">
  <img src="document\pics\data_transfer_patterns.png" width="550px">
</p>



### :hammer_and_wrench: Paradigm Workflow

As described in the next figure, our paradigm LIFT follows a two-phase structure.

<p align="center">
  <img src="document\pics\workflow.png" width="400px">
</p>

#### 1. Dataset Enhancement & Expansion

We guide GPT-4 to act as a prompt re-writer, generating challenging instructions based on specified generation rules. We iterate this process for $k$ rounds, merging the expanded datasets with the original dataset to create the final expanded dataset.

Considering the variation in content for NLU and code generation tasks within the instruction dataset, we configure distinct settings for GPT prompts to enhance complexity. For details of the prompt template please refer to `document/prompt_template`.

<p align="center">
  <img src="document\pics\expansion.png" width="700px">
</p>



#### 2. Dataset Variety and Quality Curation

### Usage
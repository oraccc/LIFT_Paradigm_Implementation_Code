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

:package: [Dataset](https://drive.google.com/drive/folders/1V_YMuFUZeLSyPeJU-F7r9mIXezm33ddo?usp=drive_link)

</div>

**LIFT** (LLM Instruction Fusion Transfer) is novel and versatile paradigm designed to elevate the instruction quality to new heights, proposed in the paper "[Rethinking the Instruction Quality: LIFT is What You Need](https://arxiv.org/abs/2312.11508)"



### :key: 1. LIFT's Key Concept: Data Distribution Transfer

LIFT is designed to combine the advantages of **data expansion** and **curation**, mitigating their shortcomings to generate a diverse and high-quality dataset while significantly reducing quantity. 

Our approach comprises two phases. Initially, we implement **"Dataset Distribution Expansion"** to broaden the data distribution and include more high-quality subspaces. Subsequently, we employ **"Dataset Variety and Quality Curation"** to eliminate redundancy, concentrating on enhancing the high-quality segments across overall data subspaces.

<p align="center">
  <img src="document\pics\data_transfer_patterns.png" width="550px">
</p>



### :hammer_and_wrench: 2. Paradigm Workflow

As described in the next figure, our paradigm LIFT follows a two-phase structure.

<p align="center">
  <img src="document\pics\workflow.png" width="400px">
</p>

#### Dataset Enhancement & Expansion

We guide GPT-4 to act as a prompt re-writer, generating challenging instructions based on specified generation rules. We iterate this process for $k$ rounds, merging the expanded datasets with the original dataset to create the final expanded dataset.

> Considering the variation in content for NLU and code generation tasks within the instruction dataset, we configure distinct settings for GPT prompts to enhance complexity. For details of the prompt template please refer to `document/prompt_template`.

<p align="center">
  <img src="document\pics\expansion.png" width="675px">
</p>




#### Dataset Variety and Quality Curation

**Variety Curation Process**

1. GPT generates embeddings with **1536** dimensions for the whole dataset.
2. Employing covariance matrix calculations and eigenvalue decomposition, we identify and retain the top eigenvectors that preserve nearly **95%** of the original embeddings' variance.
3. Analyze row variance and identify items with significant differences in the reduced space.
4. Select items with the highest **20%** row variances.

<p align="center">
  <img src="document\pics\variety_curation.png" width="700px">
</p>


**Quality Curation Process**

1. Use GPT-4 as an instruction scorer, generating GPT quality scores across four dimensions: **accuracy**, **explanation**, **clarity**, and **difficulty**. 

   > For details of the scoring prompt template please refer to `document/prompt_template`.

2. Apply a positively correlated mapping function to derive a lengthwise semantic score.

3. Combine GPT quality score and lengthwise semantic score to produce the final quality score.

4. Select items with highest quality scores for the final curated dataset.


<p align="center">
  <img src="document\pics\quality_curation.png" width="650px">
</p>


### :rocket: 3. Usage

#### Run Paradigm 

**Requirements**

- [x] You need an **OpenAI API key** to access the GPT3.5, GPT4 and the embedding models.
- [x] Prepare the original dataset (in `.json` format).

**Reproduce Steps**

1. Go to the `paradigm/expansion_phase` folder, and run the `expand_instruction_XXX_task.py` for the corresponding instruction dataset.

#### Finetune LLMs
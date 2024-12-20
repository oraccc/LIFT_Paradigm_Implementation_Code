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
![Static Badge](https://img.shields.io/badge/Model-WizardCoder-f4d35e)
![Static Badge](https://img.shields.io/badge/Model-LLaMA-f4d35e)

:page_with_curl: [Paper](https://arxiv.org/abs/2312.11508) :package: [Dataset](https://drive.google.com/drive/folders/1V_YMuFUZeLSyPeJU-F7r9mIXezm33ddo?usp=drive_link)

</div>

**LIFT** (LLM Instruction Fusion Transfer) is novel and versatile paradigm designed to elevate the instruction quality to new heights, proposed in the paper "[Rethinking the Instruction Quality: LIFT is What You Need](https://arxiv.org/abs/2312.11508)"



## :key: 1. LIFT's Key Concept: Data Distribution Transfer

LIFT is designed to combine the advantages of **data expansion** and **curation**, mitigating their shortcomings to generate a diverse and high-quality dataset while significantly reducing quantity. 

Our approach comprises two phases. Initially, we implement **"Dataset Distribution Expansion"** to broaden the data distribution and include more high-quality subspaces. Subsequently, we employ **"Dataset Variety and Quality Curation"** to eliminate redundancy, concentrating on enhancing the high-quality segments across overall data subspaces.

<p align="center">
  <img src="document\pics\data_transfer_patterns.png" width="550px">
</p>




## :hammer_and_wrench: 2. Paradigm Workflow

As described in the next figure, our paradigm LIFT follows a two-phase structure.

<p align="center">
  <img src="document\pics\workflow.png" width="400px">
</p>

### Dataset Enhancement & Expansion

We guide GPT-4 to act as a prompt re-writer, generating challenging instructions based on specified generation rules. We iterate this process for $k$ rounds, merging the expanded datasets with the original dataset to create the final expanded dataset.

> Considering the variation in content for NLU and code generation tasks within the instruction dataset, we configure distinct settings for GPT prompts to enhance complexity. For details of the prompt template please refer to `document/prompt_template`.

<p align="center">
  <img src="document\pics\expansion.png" width="675px">
</p>




### Dataset Variety and Quality Curation

#### **Variety Curation Process**

1. GPT generates embeddings with **1536** dimensions for the whole dataset.
2. Employing covariance matrix calculations and eigenvalue decomposition, we identify and retain the top eigenvectors that preserve nearly **95%** of the original embeddings' variance.
3. Analyze row variance and identify items with significant differences in the reduced space.
4. Select items with the highest **20%** row variances.

<p align="center">
  <img src="document\pics\variety_curation.png" width="700px">
  <p align="center">
      <b>Variety Curation Process</b>
  </p>
</p>



#### **Quality Curation Process**

1. Use GPT-4 as an instruction scorer, generating GPT quality scores across four dimensions: **accuracy**, **explanation**, **clarity**, and **difficulty**. 

   > For details of the scoring prompt template please refer to `document/prompt_template`.

2. Apply a positively correlated mapping function to derive a lengthwise semantic score.

3. Combine GPT quality score and lengthwise semantic score to produce the final quality score.

4. Select items with highest quality scores for the final curated dataset.

<p align="center">
  <img src="document\pics\quality_curation.png" width="650px">
  <p align="center">
      <b>Quality Curation Process</b>
  </p>
</p>



## :rocket: 3. Usage

### Run Paradigm 

#### **Requirements**

Ensure you have the following prerequisites before running Paradigm:

-  **OpenAI API key**: Obtain an OpenAI API key to access GPT3.5, GPT4, and the embedding models.
-  **Original Dataset**: Prepare the original dataset in `.json` format. 

#### **Reproduce Steps**

Follow these steps to reproduce the Paradigm workflow:

1. Navigate to the `paradigm/expansion_phase` folder and execute the `expand_instruction_XXX_task.py` script for the relevant instruction dataset.
2. Merge the expanded datasets (produced in step 1) with the original dataset.
3. Move to the `paradigm/curation_phase/variety_curation` folder and run the `generate_embedding.py` script.
4. Execute `embedding_variety_curation.py` to obtain the variety-curated dataset.
5. In the `paradigm/curation_phase/quality_curation` folder, run `generate_gpt_score.py` and `generate_length_score.py`.
6. Finally, run `quality_curation.py` to generate the final curated dataset.

### Finetune LLMs

#### Requirements

* **An active Azure Machine Learning Studio's cluster subscription.**
* **A WandB API Key**: You can generate an API key by visiting [this link](https://wandb.ai/settings).

#### Finetuning Steps

1. Open the file `cluster_finetuning/<task_name>/src/train_xxx.py/` and make the following adjustments:
   * Change the `wandb_api_key` to your own key to track the training process.
   * Change the `blob_base_model_path` to the relative path within the **azure workspace blob store** where the foundation models are stored.
2. Open the file `cluster_finetuning/<task_name>/finetune-job.yml` and make adjustments to the following parameters based on your configuration:
   * **blob_data_path**: The relative path within the **azure workspace blob store** where curated instruction datasets are stored.
   * **checkpoint_dir**: The relative path within the **azure workspace blob store** where checkpoint data will be stored during finetuning.
   * **environment**: The training environment in studio. For environment yaml, refer to `env_req`.
   * **compute**: The cluster compute name in studio.

3. Submit and launch the finetuning job by executing the following commands.

   ```bash
   # az login --use-device-code
   az ml job create --file finetune-job.yml
   ```

   You will be able to track your job's progress in the Azure ML Studio's Jobs tab.  You can also view the training progress on the WandB website. A link towards WandB job can be found in the user logs.

4. Merge LoRA adaptor (Only for NLU tasks). Launch the merge job by executing the following commands in the terminal. Please change the `peft_model_path` and `merged_model_output_dir` in the yaml file according to your settings.

   ```bash
   # az login --use-device-code
   az ml job create --file merge-job-single-instance.yml
   ```

#### Evaluate the Performance

Please refer to the following LLMs evaluation frameworks.

* [bigcode-evaluation-harness](https://github.com/bigcode-project/bigcode-evaluation-harness) (HumanEval, MBPP)

* [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) (HellaSwag, ARC, TruthfulQA, MMLU)



## :page_with_curl: 4. Citation

If you find our research helpful, please cite our paper in the following format.

```
@article{xu-2023-variety,
  title={Variety and Quality over Quantity: Towards Versatile Instruction Curation},
  author={Xu, Yang and Yao, Yongqiang and Huang, Yufan and Qi, Mengnan and Wang, Maoquan and Gu, Bin and Sundaresan, Neel},
  journal={arXiv preprint arXiv:2312.11508},
  year={2023}
}
```


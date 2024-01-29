import os
import json
import openai
from tqdm import tqdm
from multiprocessing import Pool

SOURCE_CODE_ALPACA_20K_PATH = "path/to/original/dataset"
EVOLVED_CODE_DATASET_PATH = "path/to/save/evolved/round/k/dataset"

openai.api_type = "azure"
openai.api_base = "https://inferenceendpointeastus.openai.azure.com/"
openai.api_version = "2023-03-15-preview"
openai.api_key = "xxxxx"


def generate_evolved_instructions(original_code_dataset: list, idx: int, base_prompt: str = None):
    count = 0
    new_dataset = []
    for code in tqdm(original_code_dataset):
        # # process original datasets to generate better answers to original prompts
        # question = ""
        # question += code["instruction"]
        # if code["input"]:
        #     question += "\ninput:\n" + code["input"]

        # try:
        #     response = openai.ChatCompletion.create(
        #         engine="athena-gpt-35-turbo",
        #         messages=[{"role": "user", "content": question}],
        #         temperature=0.5,
        #     )
        #     new_dataset.append(
        #         {
        #             "instruction": code["instruction"],
        #             "input": code["input"],
        #             "output": response["choices"][0]["message"]["content"].strip()
        #         }
        #     )
        # except Exception as e:
        #     print(f"Error processing code on {idx}th cpu with index {count}")
        #     print(f"Error message: {e}")
        # count += 1

        # generate new datasets
        question = ""
        question += code["instruction"]
        if code["input"]:
            question += "\ninput:\n" + code["input"]
        prompt = base_prompt + question
        try:
            response = openai.ChatCompletion.create(
                engine="athena-gpt-35-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
            )
            new_prompt = response["choices"][0]["message"]["content"]
            new_response = openai.ChatCompletion.create(
                engine="athena-gpt-35-turbo",
                messages=[{"role": "user", "content": new_prompt}],
                temperature=0.5,
            )
            new_answer = new_response["choices"][0]["message"]["content"]
            if "input:" in new_prompt:
                new_instruction, new_input = new_prompt.split("input:")
            else:
                new_instruction = new_prompt
                new_input = ""
            new_dataset.append(
                {
                    "instruction": new_instruction.strip(),
                    "input": new_input.strip(),
                    "output": new_answer.strip()
                }
            )
        except Exception as e:
            print(f"Error processing code on {idx}th cpu with index {count}")
            print(f"Error message: {e}")
        count += 1
    
    print(f"total length on {idx}th cpu: {len(new_dataset)}")
    return new_dataset


if __name__ == "__main__":
    base_prompt = "Please increase the difficulty of the given programming test question a bit. You can increase the difficulty using, but not limited to, the following methods: \n\n"
    evolve_methods = [
        "Add new constraints and requirements to the original problem, adding approximately 10 additional words.",
        "Replace a commonly used requirement in the programming task with a less common and more specific one.",
        "If the original problem can be solved with only a few logical steps, please add more reasoning steps.",
        "Provide a piece of erroneous code as a reference to increase misdirection.",
        "Propose higher time or space complexity requirements, but please refrain from doing so frequently.",
    ]
    for i, method in enumerate(evolve_methods):
        base_prompt += f"{i+1}. {method} \n"
    base_prompt += "\n"
    code_alpaca = json.load(open(SOURCE_CODE_ALPACA_20K_PATH, "r"))

    cpu_count = os.cpu_count()
    per_cpu_count = len(code_alpaca) // cpu_count

    # Split the source code dataset into subdatasets for parallel processing
    code_subdatasets = []
    for i in range(cpu_count):
        start_idx = i * per_cpu_count
        end_idx = (i + 1) * per_cpu_count if i < cpu_count-1 else len(code_alpaca)
        code_subdatasets.append(code_alpaca[start_idx:end_idx])
    
    # Use multiprocessing to parallelize the generation of evolved instructions
    with Pool(processes=cpu_count) as pool:
        processed_dataset_list = pool.starmap(
            generate_evolved_instructions,
            [(subdataset, idx, base_prompt) for idx, subdataset in enumerate(code_subdatasets)]
        )
    pool.close()
    pool.join()

    final_dataset = []
    for each_processed_dataset in processed_dataset_list:
        final_dataset.extend(each_processed_dataset)
    
    print(f"Total length of final dataset: {len(final_dataset)}")
    with open(EVOLVED_CODE_DATASET_PATH, "w") as f:
        json.dump(final_dataset, f, indent=4)
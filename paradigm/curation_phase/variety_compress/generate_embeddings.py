import json
import numpy as np
from tqdm import tqdm
import backoff
import openai
from openai.error import OpenAIError

openai.api_type = "azure"
openai.api_base = "https://inferenceendpointeastus.openai.azure.com/"
openai.api_version = "2023-03-15-preview"
openai.api_key = "xxxxx"


COMPLETE_TEXT_DICT = {
    "text_input": (
        "<INST> {instruction} Input: {input} </INST> {output}"
    ),
    "text_no_input": (
        "<INST> {instruction} </INST> {output}"
    ),
}


@backoff.on_exception(backoff.expo, OpenAIError, max_time=3)
def openai_embedding(text, engine="athena-text-embedding-ada-002"):
    response = openai.Embedding.create(input=text, engine=engine)
    hyp = response["data"][0]["embedding"] if response else None
    return hyp


if __name__ == "__main__":
    with open("path/to/dataset.json") as file:
        data = json.load(file)

    text_input, text_no_input = COMPLETE_TEXT_DICT["text_input"], COMPLETE_TEXT_DICT["text_no_input"]
    processed_data_dict = {}
    embedding_len = 1536

    for idx, item in tqdm(enumerate(data), total=len(data)):
        complete_text = (
            text_input.format_map(dict(instruction=item["instruction"], input=item["input"], output=item["output"]))
            if item["input"] != ""
            else text_no_input.format_map(dict(instruction=item["instruction"], output=item["output"]))
        )
        try:
            embedding = openai_embedding(complete_text)
            processed_data_dict[idx] = embedding
            assert len(embedding) == embedding_len
        except Exception as e:
            print(f"Error processing data with idx {idx}\nError Message: {e}")

    data_array = np.array(list(processed_data_dict.items()), dtype=[("idx", int), ("embedding", float, (embedding_len,))])
    np.save("embedding_data_ada_002.npy", data_array)




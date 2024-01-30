import os
import json
import openai
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from openai.error import OpenAIError
import backoff
import re

openai.api_type = "azure"
openai.api_base = "https://inferenceendpointeastus.openai.azure.com/"
openai.api_version = "2023-03-15-preview"
openai.api_key = "xxxxx"


SYSTEM_MESSAGE = (
    "We would like to request your feedback on the performance of an AI assistant. "
    "The assistant provides outputs for instruction and input (if any)."
)

SCORE_CRITERIA_AND_SHOTS = (
    "Please score the response to the instruction and input according to the following criteria."
    "The maximum score is 100 points, and it consists of 4 parts: \n"
    "1. Clarity (15 points): Assign a score based on how effectively the instruction conveys the problem. High-quality, clear questions score higher.\n"
    "2. Difficulty (25 points): Rate the complexity of the instruction's problem. Higher difficulty should receive a higher score.\n"
    "3. Explanations (25 points): Assess if the response includes detailed explanations alongside any code provided. The more comprehensive the explanation, the higher the score.\n"
    "4. Accuracy (35 points): Score the response based on the accuracy and correctness of the solution to the instruction's problem. Higher accuracy should receive a higher score.\n"
    "Here's some examples and socres you can follow:\n"
    "### Example 1: \n### Instruction:\nFind all substrings from a given string.\n\n### Response:\nTo find all substrings from a given string, you can use nested loops to iterate through each possible combination of characters in the string.\n\nHere's an example code in Python:\n\n```python\ndef find_substrings(string):\n    substrings = []\n    n = len(string)\n\n    # Generate all possible substrings\n    for i in range(n):\n        for j in range(i + 1, n + 1):\n            substrings.append(string[i:j])\n\n    return substrings\n\n\n# Test the function\ninput_string = \"abc\"\nresult = find_substrings(input_string)\nprint(result)\n```\n\nOutput:\n```\n['a', 'ab', 'abc', 'b', 'bc', 'c']\n```\n\nThe function `find_substrings` takes a string as input and initializes an empty list `substrings` to store the generated substrings. It then uses nested loops to iterate through each possible combination of indices `i` and `j`, where `i` represents the starting index and `j` represents the ending index of the substring. The substring is obtained using string slicing `string[i:j]` and is appended to the `substrings` list.\n\nFinally, the function returns the `substrings` list containing all the generated substrings.\n\nIn the example, the input string is \"abc\", and the output is `['a', 'ab', 'abc', 'b', 'bc', 'c']`, which represents all possible substrings of the input string."
    "\n### Score for Example 1: Total Score: 67/100; Clarity: 2/15; Difficulty: 10/25; Explanations: 20/25; Accuracy: 35/35;\n\n"
    "### Example 2: \n### Instruction:\nSuggest a solution to the following problem: given an array of numbers, find all pairs of numbers whose product is equal to the given target number.\n\n### Input:\narr = [1,5,3,2], target = 10\n\n### Response:\nOne possible solution to this problem is to use a nested loop to iterate through each pair of numbers in the array and check if their product is equal to the target number. Here is an example implementation in Python:\n\n```python\ndef find_pairs(arr, target):\n    pairs = []\n    for i in range(len(arr)):\n        for j in range(i+1, len(arr)):\n            if arr[i] * arr[j] == target:\n                pairs.append((arr[i], arr[j]))\n    return pairs\n\narr = [1, 5, 3, 2]\ntarget = 10\nresult = find_pairs(arr, target)\nprint(result)\n```\n\nOutput:\n```\n[(2, 5)]\n```\n\nIn this solution, we iterate through each element in the array using the outer loop. Then, for each element, we iterate through the remaining elements using the inner loop. By starting the inner loop from `i+1`, we avoid checking the same pair of numbers twice (e.g., (1, 5) and (5, 1)). If we find a pair whose product is equal to the target, we add it to the `pairs` list. Finally, we return the list of pairs.\n\nNote that this solution has a time complexity of O(n^2), where n is the length of the array. If the array is very large, this solution may not be efficient. In such cases, alternative approaches like using a hash table or sorting the array may provide better performance."
    "\n### Score for Example 2: Total Score: 91/100; Clarity: 15/15; Difficulty: 20/25; Explanation: 21/25; Accuracy: 35/35;\n\n"
    "### Example 3: \n### Instruction:\nImplement a function to delete a node in a linked list\n\n### Response:\nTo delete a node in a linked list, we need to update the pointers of the previous node and the next node to bypass the node we want to delete.\n\nHere is an example implementation in Python:\n\n```python\nclass Node:\n    def __init__(self, data=None):\n        self.data = data\n        self.next = None\n\nclass LinkedList:\n    def __init__(self):\n        self.head = None\n\n    def insert(self, data):\n        new_node = Node(data)\n        if self.head is None:\n            self.head = new_node\n        else:\n            current = self.head\n            while current.next:\n                current = current.next\n            current.next = new_node\n\n    def delete(self, data):\n        if self.head is None:\n            return\n\n        # If the node to be deleted is the head node\n        if self.head.data == data:\n            self.head = self.head.next\n            return\n\n        current = self.head\n        previous = None\n        while current:\n            if current.data == data:\n                break\n            previous = current\n            current = current.next\n\n        # If the node is not found\n        if current is None:\n            return\n\n        previous.next = current.next\n\n    def display(self):\n        current = self.head\n        while current:\n            print(current.data, end=\" \")\n            current = current.next\n        print()\n\n# Example usage:\nlinked_list = LinkedList()\nlinked_list.insert(1)\nlinked_list.insert(2)\nlinked_list.insert(3)\nlinked_list.insert(4)\nlinked_list.insert(5)\n\nprint(\"Before deletion:\")\nlinked_list.display()\n\nlinked_list.delete(3)\n\nprint(\"After deletion:\")\nlinked_list.display()\n```\n\nOutput:\n```\nBefore deletion:\n1 2 3 4 5 \nAfter deletion:\n1 2 4 5 \n```\n\nIn this example, we create a linked list and insert some nodes. Then, we delete a node with the value of 3 using the `delete` method. Finally, we display the linked list before and after the deletion."
    "\n### Score for Example 3: Total Score: 81/100; Clarity: 10/15; Difficulty: 13/25; Explanation: 23/25; Accuracy: 35/35;\n\n"
)

USER_MESSAGE = {
    "text_input": SCORE_CRITERIA_AND_SHOTS + (
        "Please score the upcoming Instruction, Input and Response based on these examples across four dimensions, and then add the four scores together to get the total score. Try to avoid getting a full score as much as possible. \n"
        "Please first output a single line containing the total score number only. " 
        "In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias. \n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}"
    ),
    "text_no_input": SCORE_CRITERIA_AND_SHOTS + (
        "Please score the upcoming Instruction and Response based on these examples across four dimensions, and then add the four scores together to get the total score. Try to avoid getting a full score as much as possible. \n"
        "Please first output a single line containing the total score number only. " 
        "In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias. \n\n"
        "### Instruction:\n{instruction}\n\n### Response:\n{output}"
    )
}


@backoff.on_exception(backoff.expo, openai.error.RateLimitError, max_time=5)
def get_prompt_result(prompt):
    response = openai.ChatCompletion.create(
                # engine="athena-gpt-35-turbo",
                engine="athena-gpt-4",
                messages=[
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
            )
    return str(response['choices'][0]['message'].get('content') or '').strip()


def process_single_item(idx, item):
    text_input, text_no_input = USER_MESSAGE["text_input"], USER_MESSAGE["text_no_input"]
    complete_user_prompt = (
        text_input.format_map(dict(instruction=item["instruction"], input=item["input"], output=item["output"]))
        if item["input"] != ""
        else text_no_input.format_map(dict(instruction=item["instruction"], output=item["output"]))
    )
    try:
        response = get_prompt_result(complete_user_prompt)
        score_text = response.split("\n")[0]
        number_match = re.search(r"\d+", score_text)
        if number_match:
            score = int(number_match.group())
        else:
            score = 0
        print(f"{idx}th item processed, score: {score}")
    except Exception as e:
        score = 0
        print(f"Error when processing the {idx}th item")
        print(f"Error message: {e}")

    return (idx, score)

if __name__ == "__main__":
    with open("path/to/variety/curated/dataset.json") as file:
        data = json.load(file)

    # num_processes = multiprocsessing.cpu_count()
    num_processes = 8

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        results = list(executor.map(process_single_item, range(len(data)), data))
    results.sort(key=lambda x: x[0])

    with open("gpt_scores.txt", "w") as file:
        for item in results:
            file.write(f"{item[0]}\t{item[1]}\n")

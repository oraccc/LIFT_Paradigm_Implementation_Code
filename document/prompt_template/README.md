## GPT Prompt Templates in LIFT



### :clipboard: Dataset Expansion Prompt Template

#### 1. Natural Language Understanding Tasks

```
SYSTEM MESSAGE:
I want you act as a professional prompt refinement specialist. 
USER PROMPTS:
Your task is to transform a provided prompt into a more intricate version utilizing a structured data format, introducing complexity to challenge well-known AI systems. 
However, ensure that the revised prompt remains reasonable, comprehensible, and capable of being understood and addressed by humans. 
You can enhance the complexity through various methods, including but not limited to:
(1) The depth and breadth of the inquiry can be increased.
(2) Replace general concepts with more specific concepts.
(3) If original problem can be solved with just a few simple thinking processes, you can rewrite it to explicitly request multiple-step reasoning.

#Instruction#
{Instruction}
#Input#
{Input}
```

#### 2. Code Generation Tasks

```
SYSTEM MESSAGE:
I want you act as a professional prompt refinement specialist.
USER PROMPTS:
Please increase the difficulty of the given programming test question a bit. 
You can increase the difficulty using, but not limited to, the following methods:
(1) Add new constraints and requirements to the original problem, adding approximately 10 additional words.
(2) Replace a commonly used requirement in the programming task with a less common and more specific one.
(3) If the original problem can be solved with only a few logical steps, please add more reasoning steps.
(4) Provide a piece of erroneous code as a reference to increase misdirection.
(5) Propose higher time or space complexity requirements, but please refrain from doing so frequently.

#Instruction#
{Instruction}
#Input#
{Input}
```



### :clipboard: Quality Evaluation Prompt Template

```
SYSTEM MESSAGE:
We would like to request your feedback on the performance of an AI assistant. The assistant provides outputs for instruction and input (if any).
USER PROMPTS:
Please score the response to the instruction and input according to the following criteria.
The maximum score is 100 points, and it consists of 4 parts:
1. Clarity (15 points): Assign a score based on how effectively the instruction conveys the problem. High-quality, clear questions score higher.
2. Difficulty (25 points): Rate the complexity of the instruction's problem. Higher difficulty should receive a higher score.
3. Explanations (25 points): Assess if the response includes detailed explanations alongside any code provided. The more comprehensive the explanation, the higher the score.
4. Accuracy (35 points): Score the response based on the accuracy and correctness of the solution to the instruction's problem. Higher accuracy should receive a higher score.
Here's some examples and socres you can follow:
### Example 1: 
### Instruction: {EXAMPLE INSTRUCTION 1}
### Response: {EXAMPLE OUTPUT 1}
### Score for Example 1: {SCORE 1}
### Example 2:
### Instruction: {EXAMPLE INSTRUCTION 2}
### Input: {EXAMPLE INPUT 2}
### Response: {EXAMPLE OUTPUT 2}
### Score for Example 2: {SCORE 2}
### Example 3:
### Instruction: {EXAMPLE INSTRUCTION 3}
### Response: {EXAMPLE OUTPUT 3}
### Score for Example 3: {SCORE 3}

Please score the upcoming Instruction, Input and Response based on these examples across four dimensions, and then add the four scores together to get the total score. Try to avoid getting a full score as much as possible.
Please first output a single line containing the total score number only. 
In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias.
### Instruction:
{INSTRUCTION}
### Input:
{INPUT}
### Response:
{OUTPUT}
```


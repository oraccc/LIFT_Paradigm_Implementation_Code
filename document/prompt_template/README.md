## GPT Prompt Templates in LIFT



### :clipboard: Dataset Expansion Prompt Template

#### 1. Natural Language Understanding Tasks

```bash
SYSTEM MESSAGE:
I want you act as a professional prompt refinement specialist. 
USER PROMPTS:
Your task is to transform a provided prompt into a more intricate version utilizing a structured data format, introducing complexity to challenge well-known AI systems. However, ensure that the revised prompt remains reasonable, comprehensible, and capable of being understood and addressed by humans. 
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

```bash
SYSTEM MESSAGE:
I want you act as a professional prompt refinement specialist.
USER PROMPTS:
Please increase the difficulty of the given programming test question a bit. You can increase the difficulty using, but not limited to, the following methods:
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


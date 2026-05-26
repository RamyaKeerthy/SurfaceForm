# Beyond Surface Forms: Symbolic Edits as a Test for Logical Reasoning with LLMs

This repository contains the code, data, prompts, and wrappers used for the experiments in **Beyond Surface Forms: Symbolic Edits as a Test for Logical Reasoning with LLMs**.

## Setup

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

## Repository Structure

- `data/`  
  Contains the input datasets and generated outputs.

- `inference/prompts/`  
  Contains all prompts used for inference and evaluation.

- `wrapper/`  
  Contains wrapper implementations for interfacing with different models and tools.

- `logic-program_edits.py`  
  Main script for generating symbolic edits over logic-program examples.

## Running Logic-Program Edits

### Cumulative Logic-Program Edits

Generates a sequence of cumulative edits applied to each logic program.

```bash
python wrapper/too_wrapper_{tool}.py \
  --input data/input.jsonl \
  --output data/{data}_cumulative_mutations.jsonl \
  --edit-type cumulative \
  --save-edits 0
```

### Single Logic-Program Edits (Preserve Final Answer)

Generates a single edit that modifies the program while preserving the original final answer.

```bash
python wrapper/too_wrapper_{tool}.py \
  --input data/input.jsonl \
  --output data/{data}_operator-replacements-individual.jsonl \
  --edit-type single \
  --final-answer-preservation True \
  --save-edits 1
```

### Single Logic-Program Edits (Change Prediction)

Generates a single edit that alters the final prediction of the program.

```bash
python wrapper/too_wrapper_{tool}.py \
  --input data/input.jsonl \
  --output data/{data}_operator-replacements-individual-with-alteration.jsonl \
  --edit-type single \
  --final-answer-preservation False \
  --save-edits 1
```

## Data

Input data files are located in the `data/` directory. Generated outputs from edit generation scripts will also be saved in this directory unless otherwise specified.

## Prompts

All prompts used for edit generation, inference, and evaluation are available in:

```text
inference/prompts/
```

## Model Wrappers

Wrapper implementations for supported models and tools are available in:

```text
wrapper/
```

Use the appropriate wrapper and execution script depending on the model, tool, and edit type being evaluated.

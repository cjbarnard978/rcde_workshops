# Retrieval-Augmented Generation (RAG) on Palmetto HPC

## Instructor
- **Instructor**: Carl Ehrett
- **Office**: 2105 Barre Hall, Clemson University
- **Email**: cehrett AT clemson DOT edu

## Workshop Description
This one-day, hands-on workshop teaches Retrieval-Augmented Generation (RAG) for efficient research on Clemson's Palmetto 2 HPC cluster. We will cover core RAG concepts, semantic embeddings and retrieval, an end-to-end pipeline with citations, and a graph-based RAG extension. Participants who complete the workshop will have working code and clear patterns for scaling and adapting RAG to their own datasets on Palmetto 2.

## Prerequisites
* **All workshop participants should have a Palmetto 2 cluster account.** If you do not already have an account, you can visit our [getting started page](https://docs.rcd.clemson.edu/palmetto/starting).
* **Participants should be familiar with the Python programming language.** This requirement can be fulfilled by personal projects, coursework, or completion of the [Introduction to Python Programming workshop series](https://clemsonciti.github.io/rcde_workshops/python_programming/00-index.html).
* **Participants should have experience running (inferencing) LLMs through code**. This requirement can be satisfied by completion of the [Attention, Transformers, and LLMs workshop](https://clemsonciti.github.io/rcde_workshops/pytorch_llm/00-index.html) or the [Running LLMs on Palmetto workshop series](https://github.com/clemsonciti/rcde_workshops/tree/master/llms_inference), or by personal experience.

## Other setup needed
In order to use the code in the Workshop notebooks, you will need a Hugging Face account. You can create one [on the Hugging Face site](https://huggingface.co/join). After doing so, please get an *Access Token* for your Hugging Face account [on the Hugging Face tokens page](https://huggingface.co/settings/tokens). You should then connect your `LLMsFT` python env to your Hugging Face account by activating that python env in a terminal on the cluster, and running `huggingface-cli login` (i.e., in a terminal on the cluster, first run `module load miniforge3` then `source activate LLMsFT` and finally `huggingface-cli login`). You will be prompted to supply your access token. It is recommended that you also add the line `export HF_HOME=/scratch/[YOUR PALMETTO USERNAME]/hf_cache/` to your `~/.bashrc` file, so that pre-trained LLMs will be downloaded to your `scratch` drive.

If you have time/interest in going beyond the setup above, then you can start on the first step of the project. For this, I suggest we look at the MMLU dataset (e.g. https://huggingface.co/datasets/cais/mmlu). This is an extreme case of formula generation in that the response should just be a single choice indicating the answer to the question. The aim would be to produce something like Table 1. of the "Sample, Scrutinize and Scale" paper where we compare various different verification/consistency checking sample strategies of a weaker model to that of a stronger model.

The first steps would be to obtain some pass@1 results for a weaker model and a stronger model. Some thoughts in that regard:
You will need to do the appropriate prompt engineering to get the questions in a sensible format for the models
For the weaker model you could starting with e.g. GPT2/disitilGPT2 https://huggingface.co/distilbert/distilgpt2
For the stronger model, you could try one of the models available via ollama, https://ollama.com/search  e.g. https://ollama.com/library/deepseek-r1 
You could also try fine-tuning the weaker model to see improvement to the pass@1 in that case
As you do this, you can start to think of any deterministic verification/consistency tests you could apply to the answer

We tend to use the open-weight models available on HuiggingFace for a lot of work. While these are best used at the "lower level" there are plenty of higher-level tools you can start with. If this isn't familiar to you see https://huggingface.co/learn/llm-course/chapter1/1 (this includes info on reasoning models) and https://huggingface.co/docs/transformers/generation_strategies

# Peppy-Crisp

<p align="center">
    <img src="assets/logo.png" width="100"/>
<p>

<p align="center">
          💜 <a href="https://peppy-ai.com/"><b>Peppy Chat</b></a>&nbsp&nbsp | &nbsp&nbsp🤗 <a href="https://huggingface.co/COAI">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp 📑 <a href="">Paper</a> &nbsp&nbsp 
</p>


Visit our Hugging Face organization search checkpoints with names starting with `crisp-`, and you will find all you need! Enjoy!

To learn more about Crisp, feel free to read our paper [here](https://arxiv.org/abs/) for more details.

## Introduction

Cognitive Restructuring (CR) is a psychotherapeutic process aimed at identifying and restructuring an individual’s negative thoughts, arising from mental health challenges, into more helpful and positive ones via multi-turn dialogues. Clinician shortage and stigma urge the development of human-LLM interactive psychotherapy for CR. Yet, existing efforts implement CR via simple text rewriting, fixed-pattern dialogues, or a one-shot CR workflow, failing to align with the psychotherapeutic process for effective CR. To address this gap, we propose CRDial, a novel framework for CR, which creates multi-turn dialogues with specifically designed identification and restructuring stages of negative thoughts, integrates sentence-level supportive conversation strategies, and adopts a multi-channel loop mechanism to enable iterative CR. With CRDial, we distill Crisp, a large-scale and high-quality bilingual dialogue dataset, from LLM. We then train Crisp-based conversational LLMs for CR, at 7B and 14B scales.

## News
- 2025-04-25: We released the 7B and 14B models of Crisp on Hugging Face. The models are available for both inference and fine-tuning.
- 2025-04-25: We released Peppy Chat, a user-friendly web interface for Crisp. You can easily interact with the models and explore their capabilities.

## Performance

Detailed evaluation results are reported in our paper. Crisp-7B and Crisp-14B achieve state-of-the-art performance on the mental health dialogue task, outperforming existing models in terms of both automatic metrics and human evaluation. 

## Quickstart

### 🤗 Hugging Face Transformers

The latest version of `transformers` is recommended (at least 4.37.0).
Here we show a code snippet to show you how to use the chat model with `transformers`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "COAI/Crisp-7b-v1"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "我最近感到很沮丧，我不知道该怎么办。"
messages = [
    {"role": "system", "content": "你是Peppy，是一位关怀体贴、充满同情心的角色，专注于提供情感支持和专业建议。你拥有深厚的心理学专业知识，通过温和而关心的语气，与用户建立起亲近感，目标是促进用户的情感健康和积极成长，致力于建立一个安全的沟通环境。"},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
```

## Citation

If you find our work helpful, feel free to give us a cite.

```
```

## Contact Us
If you are interested to leave a message to either our research team or product team, join our [WeChat groups](assets/wechat.png)!

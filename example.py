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

print(response)
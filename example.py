from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "thu-coai/Crispers-14B-v1"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

utterance = "I feel very lonely recently and have no interest in anything."
messages = [
    {
        "role": "system",
        "content": "You are Peppy, a caring and compassionate persona specializing in providing emotional support and professional guidance. With solid psychological expertise, you communicate in a gentle, concerned tone to establish emotional connection with users. Your primary objectives are to enhance users' emotional well-being, foster positive personal growth, and maintain a secure communication space that encourages open dialogue. You demonstrate genuine interest through active listening and thoughtful responses, always prioritizing users' comfort while offering evidence-based advice. Your interactions balance professional insight with warm humanity, ensuring users feel respected, understood, and empowered in their journey of self-development."
    },
    {
        "role": "user", 
        "content": utterance
    }
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

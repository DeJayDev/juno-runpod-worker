import os

import torch
from vllm import LLM, SamplingParams

from juno.util import generate_conversation

if os.environ.get("PWD") == "/workspace":
    print("Loading from secretloader...")
    from juno.secretloader import load

    load()
else:
    print("Loading from dotenv...")
    from dotenv import load_dotenv

    load_dotenv()

model = LLM(
    model="mistralai/Ministral-8B-Instruct-2410",
    tokenizer_mode="mistral",
    config_format="mistral",
    load_format="mistral",
    tensor_parallel_size=int(os.environ.get("RUNPOD_GPU_COUNT", "1")),
)
sampler = SamplingParams(temperature=0.15, max_tokens=8192)

with open("prompt.txt", "r") as f:
    sysprompt = f.read().strip()

logic_question = "Can you tell me about Quebec?"
persona_question = "If you could go anywhere on Earth, where would you go?"
dev_question = "Hey Juno, introduce yourself! Can you tell us your favorite color and if you prefer summer or winter?"

while True:
    user_input = str(input("Enter your question (type 'exit' to quit): "))
    if user_input.lower() == "exit":
        print("Exiting the conversation loop.")
        break

    conversation = generate_conversation(sysprompt, user_input)
    outputs = model.chat(
        conversation,
        sampling_params=sampler,
        use_tqdm=False,
        chat_template_content_format="string",
    )

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0]
        print(f"Generated text: {generated_text.text}")


torch.distributed.destroy_process_group()

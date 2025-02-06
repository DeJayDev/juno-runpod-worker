import os

import runpod
from dotenv import load_dotenv
from runpod.serverless import log
from runpod.serverless.utils.rp_validator import validate
from vllm import LLM, SamplingParams

from juno.schema import VALIDATIONS
from juno.util import generate_conversation

load_dotenv()

with open("prompt.txt", "r") as f:
    SYSTEM_PROMPT = f.read().strip()

MODEL = "mistralai/Ministral-8B-Instruct-2410"

log.info("Loading {}...".format(MODEL))

model = LLM(
    model=MODEL,
    tokenizer_mode="mistral",
    config_format="mistral",
    load_format="mistral",
    tensor_parallel_size=int(os.environ.get("RUNPOD_GPU_COUNT", "1")),
)

sampler = SamplingParams(temperature=0.15, max_tokens=1024)


def handler(job):
    input_validation = validate(job["input"], VALIDATIONS)

    if "errors" in input_validation:
        return {"error": input_validation["errors"]}
    job_input = input_validation["validated_input"]

    conversation = generate_conversation(SYSTEM_PROMPT, job_input["prompt"])
    modeloutput = model.chat(
        conversation,
        sampling_params=sampler,
        use_tqdm=False,
        chat_template_content_format="string",
    )

    generated_text = ""

    for chunk in modeloutput:
        for output in chunk.outputs:
            generated_text += output.text

    return generated_text


runpod.serverless.start({"handler": handler, "rp_args": {"rp_api_port": "12"}})

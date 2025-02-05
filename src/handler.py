""" Example handler file. """
import os

import llama_cpp
import runpod
import engine

# TODO: CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python on docker container install

# If your handler runs inference on a model, load the model here.
# You will want models to be loaded into memory before starting serverless.

gguf_engine = engine.GGUFEngine()


async def handler(job):
    """ Handler function that will be used to process jobs. """
    job_input = job["input"]

    response = await gguf_engine.async_chat_completion(job_input)
    return response


runpod.serverless.start({
    "handler": handler,
    "concurrency_modifier": 5
})

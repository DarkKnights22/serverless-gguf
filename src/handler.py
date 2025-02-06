""" Example handler file. """
import os

import llama_cpp
import runpod
import engine

# TODO: CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python on docker container install

# If your handler runs inference on a model, load the model here.
# You will want models to be loaded into memory before starting serverless.

try:
    gguf_engine = engine.GGUFEngine()
except Exception as e:
    print({"errorrrr456": str(e)})
    import traceback
    traceback.print_exc()


async def handler(job):
    """ Handler function that will be used to process jobs. """
    try:
        job_input = job["input"]

        response = await gguf_engine.async_chat_completion(job_input)
        return response
    except Exception as e:
        print({"errorrrr112": str(e)})
        import traceback
        traceback.print_exc()


runpod.serverless.start({
    "handler": handler,
    "concurrency_modifier": 5
})

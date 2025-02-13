""" Example handler file. """

import logging
import sys

import runpod
from runpod.serverless.utils.rp_validator import validate

import engine


# Restore default exception handling
def custom_excepthook(exc_type, exc_value, exc_traceback):
    logging.error("⚠️ Uncaught Exception!")
    logging.error(f"Exception Type: {exc_type.__name__}")
    logging.error(f"Exception Message: {exc_value}")
    logging.error("🔍 Full Tracebackk:")
    print("🔍 Full Tracebackk:")
    traceback.print_exception(exc_type, exc_value, exc_traceback)  # Print full error


# Override RunPod's excepthook
sys.excepthook = custom_excepthook

# Configure logging to make sure it appears in RunPod logs
logging.basicConfig(level=logging.DEBUG)

schema = {
    "prompt": {
        "type": str,
        "required": True,
    },
    "temperature": {
        "type": float,
        "required": False,
        "default": 0.7,
        "constraints": lambda x: 0.0 <= x <= 1.0,
    },
    "top_p": {
        "type": float,
        "required": False,
        "default": 0.9,
        "constraints": lambda x: 0.0 <= x <= 1.0,
    },
    "top_k": {
        "type": int,
        "required": False,
        "default": 60,
        "constraints": lambda x: x > 0,
    },
    "presence_penalty": {
        "type": float,
        "required": False,
        "default": 0.6,
        "constraints": lambda x: 0.0 <= x <= 1.0,
    },
    "frequency_penalty": {
        "type": float,
        "required": False,
        "default": 0.5,
        "constraints": lambda x: 0.0 <= x <= 1.0,
    },
    "stream": {
        "type": bool,
        "required": False,
        "default": False,
    },
}

try:
    gguf_engine = engine.GGUFEngine()
except Exception as e:
    print({"errorrrr456": str(e)})
    import traceback

    traceback.print_exc()


def handler(job):
    """ Handler function that will be used to process jobs. """
    try:
        logging.info("jobIS2", job)
        validated_input = validate(job["input"], schema)

        if "errors" in validated_input:
            yield {"error": validated_input["errors"]}
            return

        prompt = validated_input["validated_input"]["prompt"]
        temperature = validated_input["validated_input"]["temperature"]
        top_p = validated_input["validated_input"]["top_p"]
        top_k = validated_input["validated_input"]["top_k"]
        presence_penalty = validated_input["validated_input"]["presence_penalty"]
        frequency_penalty = validated_input["validated_input"]["frequency_penalty"]
        stream = validated_input["validated_input"]["stream"]

        response_generator = gguf_engine.chat_completion(prompt,
                                                         temperature,
                                                         top_p,
                                                         top_k,
                                                         presence_penalty,
                                                         frequency_penalty,
                                                         stream)

        for chunk in response_generator:
            if isinstance(chunk, dict) and "choices" in chunk:
                # Extract the actual token from the response (modify if needed)
                text = chunk["choices"][0].get("text", "")
                yield {"response": text}

    except Exception as e:
        print({"errorrrr112": str(e)})
        import traceback
        traceback.print_exc()


runpod.serverless.start({
    "handler": handler,
    "return_aggregate_stream": True  # Enables real-time streaming in RunPod
})

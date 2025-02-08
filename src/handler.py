""" Example handler file. """

import logging
import sys
import traceback

import runpod

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
        import jsonpickle
        jsonpickle.set_encoder_options('simplejson', sort_keys=False, indent=4)

        print(jsonpickle.encode(job))
        job_input = job["input"]

        response = gguf_engine.chat_completion(job_input)
        return response
    except Exception as e:
        print({"errorrrr112": str(e)})
        import traceback
        traceback.print_exc()


runpod.serverless.start({
    "handler": handler
})

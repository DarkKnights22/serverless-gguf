import logging
import os

import llama_cpp


class GGUFEngine:
    """ GGUF Engine class. """

    def __init__(self):
        """ Initialize the GGUF Engine. """

        logging.info("Loading environment variables...")
        repo_id: str = os.getenv("REPO_ID", "unsloth/DeepSeek-R1-Distill-Llama-70B-GGUF")

        # DeepSeek-R1-UD-IQ1_S/DeepSeek-R1-UD-IQ1_S-00001-of-00003.gguf
        file_name: str = os.getenv("FILE_NAME", "DeepSeek-R1-Distill-Llama-70B-Q4_K_M.gguf")

        # DeepSeek-R1-UD-IQ1_S/DeepSeek-R1-UD-IQ1_S-00002-of-00003.gguf,DeepSeek-R1-UD-IQ1_S/DeepSeek-R1-UD-IQ1_S-00003-of-00003.gguf
        additional_files_str: str = os.getenv("ADDITIONAL_FILES")

        download_dir: str = os.getenv("DOWNLOAD_DIR", f"/workspace/models/{repo_id}")
        cache_dir: str = "/workspace/hfcache"

        additional_files: list[str] | None = additional_files_str.split(",") if additional_files_str else None

        n_gpu_layers: int = int(os.getenv("N_GPU_LAYERS", 30))

        self.max_tokens: int = int(os.getenv("MAX_TOKENS", 512))

        logging.info("Loaded environment variables. Initializing Llama...")
        self.llm: llama_cpp.Llama = llama_cpp.Llama.from_pretrained(
            repo_id=repo_id,
            filename=file_name,
            additional_files=additional_files,
            local_dir=download_dir,
            cache_dir=cache_dir,
            n_gpu_layers=n_gpu_layers,
            verbose=True,
        )

        logging.info("Llama initialized.")

    def chat_completion(self, prompt: str,
                        temperature: float,
                        top_p: float,
                        top_k: int,
                        presence_penalty: float,
                        frequency_penalty: float,
                        stream: bool):

        prompt: str = f"<｜User｜>{prompt}<｜Assistant｜>"

        logging.info(f"message_input: {prompt}")
        logging.info(f"temperature: {temperature}")
        logging.info(f"top_p: {top_p}")
        logging.info(f"top_k: {top_k}")
        logging.info(f"presence_penalty: {presence_penalty}")
        logging.info(f"frequency_penalty: {frequency_penalty}")
        logging.info(f"stream: {stream}")

        return self.llm.create_completion(prompt=prompt,
                                          max_tokens=self.max_tokens,
                                          top_p=top_p,
                                          top_k=top_k,
                                          presence_penalty=presence_penalty,
                                          frequency_penalty=frequency_penalty,
                                          temperature=temperature,
                                          stream=stream)

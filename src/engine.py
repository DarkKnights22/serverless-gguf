import os

import llama_cpp
import logging

class GGUFEngine:
    """ GGUF Engine class. """

    def __init__(self):
        repo_id: str = os.getenv("REPO_ID", "unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF")

        file_name: str = os.getenv("FILE_NAME", "DeepSeek-R1-Distill-Qwen-1.5B-Q2_K.gguf")

        additional_files_str: str = os.getenv("ADDITIONAL_FILES")

        download_dir: str = os.getenv("DOWNLOAD_DIR", f"/workspace/models/{repo_id}")
        cache_dir: str = "/workspace/hfcache"

        additional_files: list[str] | None = additional_files_str.split(",") if additional_files_str else None

        self.temperature: float = os.getenv("TEMPERATURE", 0.7)
        self.top_p: float = os.getenv("TOP_P", 0.9)
        self.top_k: int = os.getenv("TOP_K", 60)
        self.presence_penalty: float = os.getenv("PRESENCE_PENALTY", 0.6)
        self.frequency_penalty: float = os.getenv("FREQUENCY_PENALTY", 0.5)
        self.max_tokens: int = os.getenv("MAX_TOKENS", 512)

        # self.chat_format: str = os.getenv("CHAT_FORMAT", "chat_template.default")

        self.llm: llama_cpp.Llama = llama_cpp.Llama.from_pretrained(
            repo_id=repo_id,
            filename=file_name,
            additional_files=additional_files,
            local_dir=download_dir,
            cache_dir=cache_dir,
            # chat_format=self.chat_format,
            use_mlock=True,
        )

    def chat_completion(self, job):
        logging.info("jobIS", job)
        stream = job.get("stream", False)

        chat_history = "<｜User｜>x = 534251<｜Assistant｜>Got it. x = 534251"
        new_question = "<｜User｜>What's 2+x?<｜Assistant｜>"
        prompt = chat_history + new_question
        import jsonpickle
        jsonpickle.set_encoder_options('simplejson', sort_keys=False, indent=4)

        jsonpickle.encode(job)
        return self.llm.create_completion(prompt=prompt,
                                          max_tokens=self.max_tokens,
                                          top_p=self.top_p,
                                          top_k=self.top_k,
                                          presence_penalty=self.presence_penalty,
                                          frequency_penalty=self.frequency_penalty,
                                          temperature=self.temperature,
                                          stream=stream,
                                          # response_format={"type": "json_object"},
                                          )


import os
import llama_cpp
import asyncio


class GGUFEngine:
    """ GGUF Engine class. """

    def __init__(self):
        repo_id: str = os.getenv("REPO_ID", "unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF")

        file_name: str = os.getenv("FILE_NAME", "DeepSeek-R1-Distill-Qwen-1.5B-Q2_K.gguf")

        additional_files_str: str = os.getenv("ADDITIONAL_FILES")

        download_dir: str = os.getenv("DOWNLOAD_DIR", f"/workspace/models/{repo_id}")
        cache_dir: str = "/workspace/hfcache"

        additional_files: list[str] | None = additional_files_str.split(",") if additional_files_str else None

        self.llm: llama_cpp.Llama = llama_cpp.Llama.from_pretrained(
            repo_id=repo_id,
            filename=file_name,
            additional_files=additional_files,
            local_dir=download_dir,
            cache_dir=cache_dir,
            use_mlock=True,
        )

    async def async_chat_completion(self, job):
        """
        Runs text generation asynchronously using llm.generate().
        """

        messages = [
            {"role": "system", "content": "You are an assistant who helps with maths problems."},
            {
                "role": "user",
                "content": "What is the sum of 2 and 3?",
            }
        ]

        stream = job.get("stream", False)

        response = await asyncio.to_thread(
            self.llm.create_chat_completion,
            messages=messages,
            max_tokens=256,
            temperature=0.6,  # Adjust as needed
            stream=stream,
            response_format={
                "type": "json_object",
            },
        )

        return response

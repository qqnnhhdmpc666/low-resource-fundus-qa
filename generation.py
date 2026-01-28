import torch
import time
import threading
from transformers import TextIteratorStreamer

def stream_generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens=256,
    log_prefix="[GEN]"
):
    print(f"{log_prefix} tokenize...")
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    print(f"{log_prefix} input_ids shape:", inputs["input_ids"].shape)

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )

    gen_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=False,          # 🔥 关键：先关采样
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    def _run_generate():
        print(f"{log_prefix} generate() start")
        try:
            model.generate(**gen_kwargs)
            print(f"{log_prefix} generate() end")
        except Exception as e:
            print(f"{log_prefix} ❌ generate exception:", e)

    thread = threading.Thread(target=_run_generate)
    thread.start()

    output_text = ""
    start = time.time()

    for token in streamer:
        print(token, end="", flush=True)
        output_text += token

    print("\n" + f"{log_prefix} streaming done, time={time.time()-start:.2f}s")
    return output_text

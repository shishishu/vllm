from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


MODEL = (
    ".hf_cache/hub/models--Qwen--Qwen3-1.7B/"
    "snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
)


def build_prompt(tokenizer, messages):
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    llm = LLM(
        model=MODEL,
        dtype="float16",
        max_model_len=2048,
        max_num_seqs=1,
        gpu_memory_utilization=0.65,
        kv_cache_memory_bytes=512 * 1024 * 1024,
        enforce_eager=True,
        tensor_parallel_size=1,
    )
    sampling_params = SamplingParams(
        temperature=0.4,
        top_p=0.9,
        max_tokens=256,
    )

    messages = [
        {
            "role": "system",
            "content": "你是一个简洁、准确的中文助手。",
        }
    ]

    print("Qwen3-1.7B interactive chat. Type /exit to quit, /clear to reset.")
    while True:
        try:
            user_message = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_message:
            continue
        if user_message in {"/exit", "/quit"}:
            break
        if user_message == "/clear":
            messages = [messages[0]]
            print("上下文已清空。")
            continue

        messages.append({"role": "user", "content": user_message})
        prompt = build_prompt(tokenizer, messages)
        outputs = llm.generate([prompt], sampling_params)
        assistant_message = outputs[0].outputs[0].text.strip()
        messages.append({"role": "assistant", "content": assistant_message})
        print(f"Qwen: {assistant_message}")


if __name__ == "__main__":
    main()

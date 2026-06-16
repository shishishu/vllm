from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def build_prompt(tokenizer, messages):
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    llm = LLM(
        model=MODEL,
        dtype="float16",
        max_model_len=1024,
        gpu_memory_utilization=0.65,
        enforce_eager=True,
        tensor_parallel_size=1,
        attention_backend="TRITON_ATTN",
    )
    sampling_params = SamplingParams(
        temperature=0.4,
        top_p=0.9,
        max_tokens=160,
    )

    messages = [
        {
            "role": "system",
            "content": "你是一个简洁、准确的中文助手。",
        }
    ]

    print("Qwen interactive chat. Type /exit to quit, /clear to reset context.")
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

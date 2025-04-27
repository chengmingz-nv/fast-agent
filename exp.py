from openai import OpenAI

if __name__ == "__main__":
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=r"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",
    )
    _user_prompt = [
        "What is the capital of the moon?",
        "What is the capital of France?",
        "What is the difference between Python and Java?",
        "1 + 1 = ?",
        "Who is WuKong?",
    ]
    for i in range(len(_user_prompt)):
        print(f"?? Question: {_user_prompt[i]}")
        completion = client.chat.completions.create(
            model=r"nvdev/nvidia/llama-3.1-nemotron-ultra-253b-v1",
            messages=[{"role": "user", "content": _user_prompt[i]}],
            temperature=0.2,
            top_p=0.6,
            max_tokens=1024 * 4,
            stream=True,
        )
        for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                print(chunk.choices[0].delta.content, end="")

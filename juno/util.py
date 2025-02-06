def generate_conversation(prompt: str, question: str):
    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": question},
    ]


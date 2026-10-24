import os

from dotenv import load_dotenv
from groq import Groq


def main():
    load_dotenv()

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        print("ERROR: GROQ_API_KEY .env file mein nahi mili.")
        return

    client = Groq(api_key=api_key)

    # Llama 3.3 model ka replacement
    model = "openai/gpt-oss-20b"

    print("Groq API connection test shuru ho raha hai...")
    print(f"Model: {model}")

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": "Say hello in one sentence.",
                }
            ],
            temperature=0,
            max_completion_tokens=100,
        )

        print("\nSUCCESS: Groq API working hai.")
        print("Response:", response.choices[0].message.content)

    except Exception as error:
        print(f"\nERROR: {type(error).__name__}: {error}")


if __name__ == "__main__":
    main()
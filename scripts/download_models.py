import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification




def download_models():
    MODEL_IDENTIFIER = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

    # TODO: Initialize `model` and `tokenizer`
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_IDENTIFIER)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_IDENTIFIER)
    if not MODEL_IDENTIFIER:
        print(
            "❌ Error: Model download script is not configured.",
            file=sys.stderr,
            flush=True,
        )
        print("👉 /scripts/download_models.py")
        sys.exit(1)

    try:
        print(f"Downloading model `{MODEL_IDENTIFIER}`...")
        # TODO: Initialize `model` and `tokenizer`.
        # tokenizer = AutoTokenizer.from_pretrained(MODEL_IDENTIFIER)
        # model = AutoModelForSequenceClassification.from_pretrained(MODEL_IDENTIFIER)
        tokenizer.save_pretrained("./models")
        model.save_pretrained("./models")
        print("✅ Models downloaded successfully.")
    except Exception as error:
        print(
            f"❌ Error downloading models: {error}",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(1)


if __name__ == "__main__":
    download_models()

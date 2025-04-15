import torch
import torch.nn as nn
from tqdm import tqdm
from shared.data.imdb_dataset import IMDBDataModule
from shared.models.text_classifier import SimpleTextClassifier, SimpleAttentionTextClassifier
from transformers import AutoTokenizer


def evaluate_model(
        model_type='simple_attention',
        model_path='project_experiments/text/imdb/imdb_model.pt',
        tokenizer_name='bert-base-uncased',
        embed_dim=128,
        max_tokens=20000,
):
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Setup data
    data_module = IMDBDataModule(tokenizer_name=tokenizer_name, max_tokens=max_tokens)
    data_module.setup()

    test_loader = data_module.test_dataloader()

    # Setup model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    if model_type == 'simple_attention':
        model = SimpleAttentionTextClassifier(
            vocab_size=tokenizer.vocab_size,
            embed_dim=embed_dim
        ).to(device)
    else:
        model = SimpleTextClassifier(
            vocab_size=tokenizer.vocab_size,
            embed_dim=embed_dim
        ).to(device)

    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Setup metrics
    criterion = nn.CrossEntropyLoss()
    total_test_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_batches = 0

    # Evaluation loop
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            inputs = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_test_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct = (preds == labels).sum().item()
            total_correct += correct
            total_samples += len(labels)
            total_batches += 1

    # Calculate metrics
    avg_loss = total_test_loss / total_batches
    accuracy = total_correct / total_samples

    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    return {
        "loss": avg_loss,
        "accuracy": accuracy
    }
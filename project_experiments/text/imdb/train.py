import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from shared.data.imdb_dataset import IMDBDataModule
from shared.models.text_classifier import SimpleTextClassifier, SimpleAttentionTextClassifier
from transformers import AutoTokenizer


def train_model(
        model_type='attention', # 'simple_attention' or 'normal'
        tokenizer_name='bert-base-uncased',
        embed_dim=128,
        learning_rate=0.001,
        epochs=2,
        max_tokens=20000,
):
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Setup data
    data_module = IMDBDataModule(tokenizer_name=tokenizer_name, max_tokens=max_tokens)
    data_module.setup()

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    # Setup model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # model type
    if model_type == 'simple_attention':
        print('Training simple attention model')
        model = SimpleAttentionTextClassifier(
            vocab_size=tokenizer.vocab_size,
            embed_dim=embed_dim
        ).to(device)

    else:
        print('Training simple average model')
        model = SimpleTextClassifier(
            vocab_size=tokenizer.vocab_size,
            embed_dim=embed_dim
        ).to(device)

    # Setup training
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(epochs):
        # Train phase
        model.train()
        train_loss = 0.0

        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]")):
            inputs = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if i % 200 == 199:
                print(f'Epoch [{epoch + 1}, {i + 1:5d}] train loss: {train_loss / 200:.3f}')
                train_loss = 0.0

        # Validation phase
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for i, batch in enumerate(tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [Val]")):
                inputs = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()

                if i % 200 == 199:
                    print(f'Epoch [{epoch + 1}, {i + 1:5d}] validation loss: {val_loss / 200:.3f}')
                    val_loss = 0.0

    # Save the model
    model_path = 'project_experiments/text/imdb/imdb_model.pt'
    torch.save(model.state_dict(), model_path)
    print(f'Model saved to {model_path}')

    return model_path
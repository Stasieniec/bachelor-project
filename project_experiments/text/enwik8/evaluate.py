import os, sys, torch, math
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.data.enwiki8_dataset import WikipediaDataModule          # the Dataset class we wrote
from shared.models.gpt       import SimpleGPT           # the model we built



def evaluate_model(
        model_path   = "project_experiments/text/enwik8/gpt.pt",
        d_model      = 256,
        n_heads      = 4,
        n_layers     = 6,
        block_size   = 512,
        batch_size   = 16,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    ds_test = WikipediaDataModule(split="test", block_size=block_size)
    loader  = torch.utils.data.DataLoader(ds_test, batch_size=batch_size,
                                          shuffle=False, drop_last=True)

    model = SimpleGPT(
        vocab_size = 256,
        d_model    = d_model,
        n_heads    = n_heads,
        n_layers   = n_layers,
        block_size = block_size,
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    losses = []
    with torch.no_grad():
        for xb, yb in tqdm(loader, desc="Test"):
            xb, yb = xb.to(device), yb.to(device)
            _, loss = model(xb, yb)
            losses.append(loss.item())

    ppl = math.exp(sum(losses)/len(losses))
    print(f"Test loss {sum(losses)/len(losses):.4f}   |   perplexity {ppl:.1f}")
    return {"loss": sum(losses)/len(losses), "perplexity": ppl}

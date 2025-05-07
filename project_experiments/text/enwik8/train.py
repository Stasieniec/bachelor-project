import os, sys, torch, math, time
import torch.nn.functional as F
from tqdm import tqdm

# project root on path  ( = …/project_experiments/..)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.data.enwiki8_dataset import WikipediaDataModule          # the Dataset class we wrote
from shared.models.gpt       import SimpleGPT           # the model we built

# ---------- main callable ----------------------------------------------------
def train_model(
        model_path      = "project_experiments/text/enwik8/gpt.pt",
        d_model         = 256,
        n_heads         = 4,
        n_layers        = 6,
        block_size      = 512,
        batch_size      = 12,
        lr              = 3e-4,
        epochs          = 3,
        eval_interval   = 1000,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Dataset & loader --------------------------------------------------------
    ds_train = WikipediaDataModule(split="train",  block_size=block_size)
    ds_val   = WikipediaDataModule(split="val",    block_size=block_size)

    train_loader = torch.utils.data.DataLoader(
        ds_train, batch_size=batch_size, shuffle=True, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        ds_val,   batch_size=batch_size, shuffle=False, drop_last=True)

    # Model -------------------------------------------------------------------
    model = SimpleGPT(
        vocab_size = 256,
        d_model    = d_model,
        n_heads    = n_heads,
        n_layers   = n_layers,
        block_size = block_size,
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    # ------------- training loop --------------------------------------------
    step = 0
    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for xb, yb in pbar:
            xb, yb = xb.to(device), yb.to(device)
            _, loss = model(xb, yb)

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            step += 1
            pbar.set_postfix(loss=f"{loss.item():.3f}")

            # quick val loss every eval_interval steps
            if step % eval_interval == 0:
                val_loss = evaluate(model, val_loader, device)
                print(f"\nstep {step:>7}  |  val loss {val_loss:.3f}")

        # --- end epoch -------------------------------------------------------

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print("Saved model to", model_path)
    return model_path


# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, device="cpu"):
    model.eval()
    losses = []
    for xb, yb in loader:
      xb, yb = xb.to(device), yb.to(device)
      _, loss = model(xb, yb)
      losses.append(loss.item())
    model.train()
    return sum(losses)/len(losses)

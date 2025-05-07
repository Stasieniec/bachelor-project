import os, sys, torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.models.gpt     import SimpleGPT

def generate_text(
        prompt          = "The meaning of life is",
        model_path      = "project_experiments/text/enwik8/gpt.pt",
        max_new_tokens  = 400,
        temperature     = 1.0,
        top_k           = 100,
        d_model         = 256,
        n_heads         = 4,
        n_layers        = 6,
        block_size      = 512,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------- load model ----------------------------------------------
    model = SimpleGPT(
        vocab_size = 256,
        d_model    = d_model,
        n_heads    = n_heads,
        n_layers   = n_layers,
        block_size = block_size,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # ------------- encode prompt to byte‑ids -------------------------------
    # bytes -> tensor[int] of shape (1, T₀)
    prompt_ids = torch.tensor(list(prompt.encode("utf‑8")), dtype=torch.long,
                              device=device)[None, :]

    # ------------- generate -------------------------------------------------
    ids = model.generate(prompt_ids,
                         max_new_tokens=max_new_tokens,
                         temperature=temperature,
                         top_k=top_k)

    # ------------- decode to string ----------------------------------------
    # detach, drop batch dim, convert 0‑255 ints back to bytes
    generated = bytes(ids[0].tolist()).decode("utf‑8", errors="ignore")
    print(generated)
    return generated

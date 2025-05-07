# shared/data/enwiki8_dataset.py
import gzip, torch, pathlib
from torch.utils.data import Dataset

SPLIT_SIZES = dict(train=int(90e6), val=int(5e6), test=int(5e6))

def _load_enwik8(path: str | pathlib.Path) -> torch.Tensor:
    """
    Returns a 1‑D uint8 tensor with the first 100 MB of enwik8.
    """
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found ‑‑ put enwik8.gz there or change the path")
    with gzip.open(path, "rb") as f:             # bytes → uint8 numpy → tensor
        data = f.read(sum(SPLIT_SIZES.values()))
    return torch.tensor(list(data), dtype=torch.uint8)   # (100_000_000,)

class WikipediaDataModule(Dataset):
    """
    A simple window‑based dataset for enwik8.

    Each __getitem__(i) returns a pair (x, y) of length `block_size`
    such that y is x shifted 1 byte to the right.
    """
    def __init__(self,
                 split: str,                    # "train" | "val" | "test"
                 block_size: int = 512,
                 path: str | pathlib.Path = "shared/data/enwik8.gz"):
        assert split in SPLIT_SIZES, f"split must be one of {list(SPLIT_SIZES)}"
        self.block = block_size

        # --- load full 100 MB only once and cache on the class ----------------
        if not hasattr(WikipediaDataModule, "_full_data"):
            WikipediaDataModule._full_data = _load_enwik8(path)

        full = WikipediaDataModule._full_data                   # (100M,)
        n_train = SPLIT_SIZES["train"]
        n_val   = SPLIT_SIZES["val"]
        # slice without copying
        if split == "train":
            self.data = full[            : n_train]
        elif split == "val":
            self.data = full[n_train     : n_train + n_val]
        else:  # test
            self.data = full[n_train + n_val : ]

        # cast once to int64 for Embedding
        self.data = self.data.long()

    # ---------------------------------------------------------------------
    def __len__(self):
        # last window must have a target;  -1  because y is shifted by 1
        return len(self.data) - self.block - 1

    def __getitem__(self, idx: int):
        x = self.data[idx              : idx + self.block]       # (block,)
        y = self.data[idx + 1          : idx + self.block + 1]   # (block,)
        return x, y

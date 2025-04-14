from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding


class IMDBDataModule:
    def __init__(self, tokenizer_name='bert-base-uncased', max_tokens=20000):
        self.tokenizer_name = tokenizer_name
        self.max_tokens = max_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.collator = DataCollatorWithPadding(self.tokenizer)

    def setup(self):
        # Load dataset
        raw_dataset = load_dataset('imdb')

        # Create train/val split
        split = raw_dataset['train'].train_test_split(test_size=0.15)
        self.train_dataset = split['train']
        self.val_dataset = split['test']
        self.test_dataset = raw_dataset['test']

        # Tokenize datasets
        self.train_dataset = self.tokenize_and_prepare(self.train_dataset)
        self.val_dataset = self.tokenize_and_prepare(self.val_dataset)
        self.test_dataset = self.tokenize_and_prepare(self.test_dataset)

    def tokenize_and_prepare(self, dataset):
        # Tokenize
        dataset = dataset.map(
            lambda x: self.tokenizer(x['text'], truncation=True, padding=False),
            batched=True,
            remove_columns=['text']
        )

        # Add length field for variable batching
        dataset = dataset.map(lambda x: {'length': len(x['input_ids'])})

        # Sort by length
        dataset = dataset.sort('length')

        return dataset

    def get_dataloader(self, dataset, batch_size=None):
        from shared.utils.batch_sampler import VariableBatchSampler

        sampler = VariableBatchSampler(
            dataset=dataset,
            max_tokens=self.max_tokens
        )

        return DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=self.collator
        )

    def train_dataloader(self):
        return self.get_dataloader(self.train_dataset)

    def val_dataloader(self):
        return self.get_dataloader(self.val_dataset)

    def test_dataloader(self):
        return self.get_dataloader(self.test_dataset)
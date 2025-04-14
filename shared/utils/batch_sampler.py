from torch.utils.data import Sampler

class VariableBatchSampler(Sampler):
    def __init__(self, dataset, max_tokens=20000, shuffle=False):
        super().__init__(dataset)
        self.dataset = dataset
        self.max_tokens = max_tokens
        self.shuffle = shuffle

        self.indices = list(range(len(dataset)))
        self.lengths = [dataset[i]['length'] for i in self.indices]

    def __iter__(self):
        curr_batch = []
        curr_max_len = 0

        for i in self.indices:
            element_length = self.lengths[i]

            new_max_len = max(curr_max_len, element_length)
            new_batch_size = len(curr_batch) + 1
            new_padded_total_size = new_max_len * new_batch_size

            if new_padded_total_size > self.max_tokens:
                yield curr_batch
                curr_batch = [i]
                curr_max_len = element_length
            else:
                curr_batch.append(i)
                curr_max_len = new_max_len

        if curr_batch:
            yield curr_batch
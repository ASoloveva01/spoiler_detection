from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch
def create_dataloader(texts, targets, tokenizer, max_len=512):

    ds = ReviewsDataset(texts=texts.to_numpy(), targets=targets.to_numpy(), tokenizer=tokenizer, max_len=max_len)
    # Определяем sampler для баланса классов в каждом батче
    class_counts = targets.value_counts()
    weights = [1/class_counts[i] for i in targets.values]
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(ds), replacement=True)

    return DataLoader(ds, sampler=sampler, batch_size=16)

class ReviewsDataset(Dataset):

    def __init__(self, texts, targets, tokenizer, max_len=512):
        self.texts = texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        target = self.targets[item]

        encoding = self.tokenizer.encode_plus(
                    text,
                    add_special_tokens=True,
                    max_length=self.max_len,
                    return_token_type_ids=False,
                    pad_to_max_length=True,
                    return_attention_mask=True,
                    return_tensors='pt',
                    )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
            }

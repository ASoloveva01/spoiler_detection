from torch import nn
from transformers import AutoModel

class SpoilerClassifier(nn.Module):
    def __init__(self, pretrained_model_name, n_classes=2):
        super(SpoilerClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model_name, return_dict=False)
        self.drop = nn.Dropout(p=0.5)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
          input_ids=input_ids,
          attention_mask=attention_mask
        )
        output = self.drop(pooled_output)

        return self.out(output)
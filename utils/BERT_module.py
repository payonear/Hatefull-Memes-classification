import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

class BERT_module(nn.Module):
    def __init__(self, output_dim, device='cpu', learn=False):
        super(BERT_module, self).__init__()
        self.model = BertModel.from_pretrained('bert-base-cased')
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                                    do_lower_case=True)
        self.model.resize_token_embeddings(len(tokenizer))
        if not learn:
            for param in self.model.parameters():
                param.requires_grad = False
        self.fc = nn.Linear(self.model.config.hidden_size, output_dim)
        self.model.to(device)

    def forward(self, inputs):
        input_ids, attention_mask = inputs[:,0], inputs[:,1]
        h, _ = self.model(input_ids=input_ids,
                         attention_mask=attention_mask)
        h_cls = h[:, 0]
        outputs = self.fc(h_cls)
        return outputs

    @staticmethod
    def transform(max_len = 512):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                                    do_lower_case=True)
        return lambda sent: torch.stack(list(map(tokenizer.encode_plus(
                        sent,                  # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = max_len,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        truncation=True,
                        return_tensors = 'pt',     # Return pytorch tensors.
                   ).get, ['input_ids', 'attention_mask']))).squeeze()
        
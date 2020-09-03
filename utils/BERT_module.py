import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

class BERT_module(nn.Module):
    def __init__(self, device='cpu'):
        super(BERT_module, self).__init__()
        self.model = BertModel.from_pretrained('bert-base-cased')
        self.model.to(device)

    def forward(self, input_ids):   
        outputs = self.model(input_ids)[0].squeeze(dim=0)
        return outputs[0]

    @staticmethod
    def transform(max_len = 512, device='cpu'):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                                    do_lower_case=True)
        return lambda sent: tokenizer.encode_plus(
                        sent,                  # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = max_len,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        truncation=True,
                        return_tensors = 'pt',     # Return pytorch tensors.
                   )['input_ids'].to(device)
        
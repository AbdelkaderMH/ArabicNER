import torch
import torch.nn as nn
from transformers import BertModel
from arabiner.nn import BaseModel

class BertMTLTagger(BaseModel):
    def __init__(self, **kwargs):
        super(BertMTLTagger, self).__init__(**kwargs)

        #self.max_num_labels = max(self.num_labels)
        hidden_size = self.bert.config.hidden_size
        classifiers = [nn.Linear(hidden_size, 1) for num_labels in range(self.num_labels)]
        self.classifiers = torch.nn.Sequential(*classifiers)

    def forward(self, x):
        y = self.bert(x)
        y = self.dropout(y["last_hidden_state"])
        #y = y["last_hidden_state"]
        
        output = list()

        for i, classifier in enumerate(self.classifiers):
            logits = classifier(y)
            #print('CLS logits', logits.shape)

            # Pad logits to allow Multi-GPU/DataParallel training to work
            # We will truncate the padded dimensions when we compute the loss in the trainer
            output.append(logits)

        # Return tensor of the shape B x T x L x C
        # B: batch size
        # T: sequence length
        # L: number of tag types
        # C: number of classes per tag type
        output = torch.cat(output, dim=2)
        return output


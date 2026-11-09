"""
Phase 2: Legal Named Entity Recognition (L-NER)
Architecture: InLegalBERT Encoder + BiLSTM + Linear Projection / CRF
with Character Boundary Alignment Calibration.
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class LegalNERModel(nn.Module):
    def __init__(self, pretrained_model_name="law-ai/InLegalBERT", num_labels=14, lstm_hidden=256, dropout=0.2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(pretrained_model_name)
        hidden_size = self.encoder.config.hidden_size
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_hidden * 2, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        lstm_out, _ = self.lstm(sequence_output)
        logits = self.classifier(self.dropout(lstm_out))

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))
        return logits, loss

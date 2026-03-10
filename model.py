import torch
import torch.nn as nn

class CNN_BiLSTM_ECG(nn.Module):
    def __init__(self, num_classes=7):
        super(CNN_BiLSTM_ECG, self).__init__()
        
        # CNN avec BatchNorm
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # Réduction de séquence
        self.global_pool = nn.AdaptiveAvgPool1d(100)
        
        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=64, hidden_size=128, 
            batch_first=True, dropout=0.3, 
            bidirectional=True
        )
        
        # Fully connected
        self.fc = nn.Sequential(
            nn.Linear(128*2, 128),
            nn.ReLU(),
            nn.Dropout(0.5),   # dropout un peu plus fort
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.cnn(x)          # (B, 64, L/8)
        x = self.global_pool(x)  # (B, 64, 100)
        x = x.permute(0, 2, 1)   # (B, 100, 64)

        _, (h_n, _) = self.lstm(x)  
        h_n = h_n.permute(1, 0, 2).reshape(x.size(0), -1)  # (B, 256)

        return self.fc(h_n)

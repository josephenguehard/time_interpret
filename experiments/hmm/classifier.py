import torch as th
import torch.nn as nn


class StateClassifier(nn.Module):
    def __init__(
        self,
        feature_size: int,
        n_state: int,
        hidden_size: int,
        rnn: str = "GRU",
        regres: bool = True,
        bidirectional: bool = False,
        return_all: bool = False,
    ):
        super(StateClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.n_state = n_state
        self.rnn_type = rnn
        self.regres = regres
        self.return_all = return_all
        # Input to torch LSTM should be of size (seq_len, batch, input_size)
        if self.rnn_type == "GRU":
            self.rnn = nn.GRU(
                feature_size,
                self.hidden_size,
                bidirectional=bidirectional,
                batch_first=True,
            ).to(self.device)
        else:
            self.rnn = nn.LSTM(
                feature_size,
                self.hidden_size,
                bidirectional=bidirectional,
                batch_first=True,
            ).to(self.device)

        self.regressor = nn.Sequential(
            nn.BatchNorm1d(num_features=self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.hidden_size, self.n_state),
        )

    def forward(self, x):
        if self.rnn_type == "GRU":
            all_encodings, encoding = self.rnn(x)
        else:
            all_encodings, (encoding, state) = self.rnn(x)

        if self.regres:
            if not self.return_all:
                return self.regressor(encoding.view(encoding.shape[1], -1))
            else:
                reshaped_encodings = all_encodings.view(
                    all_encodings.shape[1] * all_encodings.shape[0], -1
                )
                return th.t(
                    self.regressor(reshaped_encodings).view(
                        all_encodings.shape[0], -1
                    )
                )
        else:
            return encoding.view(encoding.shape[1], -1)

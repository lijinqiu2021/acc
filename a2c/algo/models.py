import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def get_conv_net(n, m, hidden_size):
    if n <= 1 or m <= 1:
        image_embedding_size = n * m * 3
        image_conv = nn.Sequential(
            nn.Flatten(),
            layer_init(nn.Linear(image_embedding_size, hidden_size)),
            nn.ReLU(),
        )
    elif n <= 3 or m <= 3:
        image_embedding_size = (n + 2 - 2 + 1 - 2 + 1 - 2 + 1) * (m + 2 - 2 + 1 - 2 + 1 - 2 + 1) * 64
        image_conv = nn.Sequential(
            layer_init(nn.Conv2d(3, 32, (2, 2), stride=1, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, (2, 2), stride=1, padding=0)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, (2, 2), stride=1, padding=0)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(image_embedding_size, hidden_size)),
            nn.ReLU(),
        )
    elif n <= 15 or m <= 15:
        image_embedding_size = (n - 2 + 1 - 2 + 1 - 2 + 1) * (m - 2 + 1 - 2 + 1 - 2 + 1) * 64
        image_conv = nn.Sequential(
            layer_init(nn.Conv2d(3, 32, (2, 2), stride=1, padding=0)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, (2, 2), stride=1, padding=0)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, (2, 2), stride=1, padding=0)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(image_embedding_size, hidden_size)),
            nn.ReLU(),
        )
    else:
        image_embedding_size = ((n - 2 + 1) // 2 - 2 + 1 - 2 + 1) * ((m - 2 + 1) // 2 - 2 + 1 - 2 + 1) * 64
        image_conv = nn.Sequential(
            layer_init(nn.Conv2d(3, 32, (2, 2), stride=1, padding=0)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            layer_init(nn.Conv2d(32, 64, (2, 2), stride=1, padding=0)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, (2, 2), stride=1, padding=0)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(image_embedding_size, hidden_size)),
            nn.ReLU(),
        )

    return image_conv


class ActorModel(nn.Module):
    def __init__(self, obs_space, action_space, recurrent=True, hidden_size=64):
        super().__init__()

        # Decide which components are enabled
        self.recurrent = recurrent
        self.hidden_size = hidden_size
        n = obs_space["obs_image"].shape[0]
        m = obs_space["obs_image"].shape[1]

        # Define image embedding
        self.image_conv = get_conv_net(n, m, hidden_size)

        # Define memory
        if self.recurrent:
            self.memory_rnn = nn.LSTMCell(self.hidden_size, self.hidden_size)
            for name, param in self.memory_rnn.named_parameters():
                if "bias" in name:
                    nn.init.constant_(param, 0)
                elif "weight" in name:
                    nn.init.orthogonal_(param, 1.0)

        # Resize image embedding
        self.embedding_size = self.hidden_size

        # Define actor's model
        self.head = nn.Sequential(
            layer_init(nn.Linear(self.embedding_size, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, action_space.n), std=0.01),
        )

    @property
    def memory_size(self):
        return 2 * self.hidden_size

    def forward(self, obs, memory=None):
        x = obs["obs_image"].permute(0, 3, 1, 2).to(torch.float32)
        conv_x = self.image_conv(x)

        if self.recurrent:
            hidden = (memory[:, : self.hidden_size], memory[:, self.hidden_size :])
            hidden = self.memory_rnn(conv_x, hidden)
            embedding = hidden[0]
            memory = torch.cat(hidden, dim=1)
        else:
            embedding = conv_x

        head_x = self.head(embedding)
        dist = Categorical(logits=F.log_softmax(head_x, dim=1))

        return dist, memory


class ObsCriticModel(nn.Module):
    def __init__(self, obs_space, recurrent=True, hidden_size=64):
        super().__init__()

        # Decide which components are enabled
        self.recurrent = recurrent
        self.hidden_size = hidden_size
        n = obs_space["obs_image"].shape[0]
        m = obs_space["obs_image"].shape[1]

        # Define image embedding
        self.image_conv = get_conv_net(n, m, hidden_size)

        # Define memory
        if self.recurrent:
            self.memory_rnn = nn.LSTMCell(self.hidden_size, self.hidden_size)
            for name, param in self.memory_rnn.named_parameters():
                if "bias" in name:
                    nn.init.constant_(param, 0)
                elif "weight" in name:
                    nn.init.orthogonal_(param, 1.0)

        # Resize image embedding
        self.embedding_size = self.hidden_size

        # Define critic's model
        self.head = nn.Sequential(
            layer_init(nn.Linear(self.embedding_size, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 1), std=1.0),
        )

    @property
    def memory_size(self):
        return 2 * self.hidden_size

    def forward(self, obs, memory=None):
        x = obs["obs_image"].permute(0, 3, 1, 2).to(torch.float32)
        conv_x = self.image_conv(x)

        if self.recurrent:
            hidden = (memory[:, : self.hidden_size], memory[:, self.hidden_size :])
            hidden = self.memory_rnn(conv_x, hidden)
            embedding = hidden[0]
            memory = torch.cat(hidden, dim=1)
        else:
            embedding = conv_x

        head_x = self.head(embedding)
        value = head_x.squeeze(1)

        return value, memory


class StateCriticModel(nn.Module):
    def __init__(self, obs_space, recurrent=True, hidden_size=64):
        super().__init__()

        # Decide which components are enabled
        self.recurrent = recurrent
        self.hidden_size = hidden_size
        n = obs_space["state_image"].shape[0]
        m = obs_space["state_image"].shape[1]

        # Define image embedding
        self.image_conv = get_conv_net(n, m, hidden_size)

        # Resize image embedding
        self.embedding_size = self.hidden_size

        # Define critic's model
        self.head = nn.Sequential(
            layer_init(nn.Linear(self.embedding_size, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 1), std=1.0),
        )

    @property
    def memory_size(self):
        return 2 * self.hidden_size

    def forward(self, obs, memory=None):
        x = obs["state_image"].permute(0, 3, 1, 2).to(torch.float32)
        conv_x = self.image_conv(x)

        embedding = conv_x

        head_x = self.head(embedding)
        value = head_x.squeeze(1)

        return value, memory


class ObsStateCriticModel(nn.Module):
    def __init__(self, obs_space, recurrent=True, hidden_size=64):
        super().__init__()

        # Decide which components are enabled
        self.recurrent = recurrent
        self.hidden_size = hidden_size
        obs_n = obs_space["obs_image"].shape[0]
        obs_m = obs_space["obs_image"].shape[1]
        state_n = obs_space["state_image"].shape[0]
        state_m = obs_space["state_image"].shape[1]

        # Define image embedding
        self.obs_image_conv = get_conv_net(obs_n, obs_m, hidden_size)
        self.state_image_conv = get_conv_net(state_n, state_m, hidden_size)

        # Define memory
        if self.recurrent:
            self.memory_rnn = nn.LSTMCell(self.hidden_size, self.hidden_size)
            for name, param in self.memory_rnn.named_parameters():
                if "bias" in name:
                    nn.init.constant_(param, 0)
                elif "weight" in name:
                    nn.init.orthogonal_(param, 1.0)

        # Resize image embedding
        self.embedding_size = self.hidden_size + self.hidden_size

        # Define critic's model
        self.head = nn.Sequential(
            layer_init(nn.Linear(self.embedding_size, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 1), std=1.0),
        )

    @property
    def memory_size(self):
        return 2 * self.hidden_size

    def forward(self, obs, memory=None):
        obs_x = obs["obs_image"].permute(0, 3, 1, 2).to(torch.float32)
        state_x = obs["state_image"].permute(0, 3, 1, 2).to(torch.float32)
        conv_obs_x = self.obs_image_conv(obs_x)
        conv_state_x = self.state_image_conv(state_x)

        if self.recurrent:
            hidden = (memory[:, : self.hidden_size], memory[:, self.hidden_size :])
            hidden = self.memory_rnn(conv_obs_x, hidden)
            obs_embedding = hidden[0]
            memory = torch.cat(hidden, dim=1)
        else:
            obs_embedding = conv_obs_x
        embedding = torch.cat((obs_embedding, conv_state_x), dim=-1)

        head_x = self.head(embedding)
        value = head_x.squeeze(1)

        return value, memory

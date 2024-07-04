import torch


class DictList(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def __len__(self):
        return len(next(iter(dict.values(self))))

    def __getitem__(self, index):
        if isinstance(index, str):
            return dict.get(self, index)
        else:
            return DictList({key: value[index] for key, value in dict.items(self)})

    def __setitem__(self, index, d):
        for key, value in d.items():
            dict.__getitem__(self, key)[index] = value


def vec_obs_as_tensor(obs, device):
    # single_obs: shape=[D1, D2, D3 ...] ... (named s_o), s_o should be np.ndarray
    # obs: np.array([s_o, s_o, s_o ...]) shape=[E, D1, D2, D3 ...]
    # to_tensor: shape=[E, D1, D2, D3 ...]
    if isinstance(obs, dict):
        return DictList({key: torch.as_tensor(obs_).to(device) for (key, obs_) in obs.items()})
    else:
        return torch.as_tensor(obs).to(device)


def batch_tensor_obs_squeeze(obss):
    # vec_obs: shape=[E, D1, D2, D3 ...] ... (named v_o) , v_o should be torch.Tensor
    # obss: [v_o, v_o, v_o ...(xT)]
    # to_tensor: shape=[T, E, D1, D2, D3 ...] => [E, T, D1, D2, D3 ...] => [(E*T), D1, D2, D3 ...]
    assert isinstance(obss, list)

    if isinstance(obss[0], dict):
        return {
            key: torch.stack([obs[key] for obs in obss]).transpose(0, 1).reshape(-1, *obss[0][key].shape[1:])
            for key in obss[0].keys()
        }
    else:
        return torch.stack(obss).transpose(0, 1).reshape(-1, *obss[0].shape[1:])

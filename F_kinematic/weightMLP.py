import torch
import torch.nn as nn
import numpy as np

# code from https://github.com/mikeqzy/3dgs-avatar-release.git

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

def get_embedder(multires, input_dims=3):
    if multires == 0:
        return lambda x: x, input_dims
    assert multires > 0

    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dims,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)

    def embed(x, eo=embedder_obj): return eo.embed(x)

    return embed, embedder_obj.out_dim

class VanillaCondMLP(nn.Module):
    def __init__(self, dim_in, dim_cond, dim_out, n_neurons=256, n_hidden_layers=4, multires=0, last_layer_init=False,
                 skip_in=[], cond_in=[], dim_coord=3):
        super(VanillaCondMLP, self).__init__()

        self.n_input_dims = dim_in
        self.n_output_dims = dim_out

        self.n_neurons, self.n_hidden_layers = n_neurons, n_hidden_layers


        dims = [dim_in] + [self.n_neurons for _ in range(self.n_hidden_layers)] + [dim_out]

        self.embed_fn = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=dim_in)
            self.embed_fn = embed_fn
            dims[0] = input_ch

        self.last_layer_init = last_layer_init

        self.num_layers = len(dims)

        self.skip_in = skip_in
        self.cond_in = cond_in

        for l in range(0, self.num_layers - 1):
            if l + 1 in skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            if l in cond_in:
                lin = nn.Linear(dims[l] + dim_cond, out_dim)
            else:
                lin = nn.Linear(dims[l], out_dim)

            if self.last_layer_init and l == self.num_layers - 2:
                torch.nn.init.normal_(lin.weight, mean=0., std=1e-5)
                torch.nn.init.constant_(lin.bias, val=0.)


            setattr(self, "lin" + str(l), lin)

        self.activation = nn.LeakyReLU()

    def forward(self, coords, cond=None):
        if cond is not None:
            cond = cond.expand(coords.shape[0], -1)

        if self.embed_fn is not None:
            coords_embedded = self.embed_fn(coords)
        else:
            coords_embedded = coords

        x = coords_embedded
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.cond_in:
                x = torch.cat([x, cond], 1)

            if l in self.skip_in:
                x = torch.cat([x, coords_embedded], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)

        return x


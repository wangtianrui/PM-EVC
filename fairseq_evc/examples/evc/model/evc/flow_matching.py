from abc import ABC
import torch
import torch.nn as nn
import torch.nn.functional as F
from .fmdecoder import Decoder
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import random

class CFM(torch.nn.Module, ABC):
    def __init__(
        self,
        n_feats,
        cfm_params,
        decoder_params,
        content_emb_dim=0,
        spk_emb_dim=0,
        emo_emb_dim=0,
        conv_emb_dim=0,
        out_channel=-1,
        t_scheduler="",
    ):
        super().__init__()
        self.n_feats = n_feats
        self.spk_emb_dim = spk_emb_dim
        self.solver = cfm_params["solver"]
        self.sigma_min = cfm_params["sigma_min"]
        self.estimator = Decoder(in_channels=2*n_feats+spk_emb_dim+emo_emb_dim+conv_emb_dim+content_emb_dim, out_channels=out_channel, **decoder_params)
        self.t_scheduler = t_scheduler

    @torch.inference_mode()
    def forward(self, mu, mask, n_timesteps, temperature=1.0, spks=None, emo=None, conv_f=None, bool_masks=None):
        """Forward diffusion

        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """
        z = torch.randn_like(mu) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device)
        if self.t_scheduler == 'cosine':
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, emo=emo, conv_f=conv_f, bool_masks=bool_masks)

    def solve_euler(self, x, t_span, mu, mask, spks, emo, conv_f=None, bool_masks=None):
        """
        Fixed euler solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes
        """
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]

        # I am storing this because I can later plot it by putting a debugger here and saving it to a file
        # Or in future might add like a return_all_steps flag
        sol = []

        for step in range(1, len(t_span)):
            dphi_dt = self.estimator(x, mask, mu, t, spks, emo, conv_f=conv_f, bool_masks=bool_masks)

            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        return sol[-1]

    def compute_loss(self, x1, mask, mu, spks=None, emo=None, conv_f=None, bool_maks=None):
        """Computes diffusion loss

        Args:
            x1 (torch.Tensor): Target
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): target mask
                shape: (batch_size, 1, mel_timesteps)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            spks (torch.Tensor, optional): speaker embedding. Defaults to None.
                shape: (batch_size, spk_emb_dim)

        Returns:
            loss: conditional flow matching loss
            y: conditional flow
                shape: (batch_size, n_feats, mel_timesteps)
        """
        b, _, t = mu.shape
        # random timestep
        t = torch.rand([b, 1, 1], device=mu.device, dtype=mu.dtype)
        if self.t_scheduler == 'cosine':
            t = 1 - torch.cos(t * 0.5 * torch.pi)
            
        z = torch.randn_like(x1)
        y = (1 - (1 - self.sigma_min) * t) * z + t * x1
        u = x1 - (1 - self.sigma_min) * z
        
        est = self.estimator(y, mask, mu, t.squeeze(), spks, emo, conv_f=conv_f, bool_masks=bool_maks)
        # print(est.mean(), est.sum())
        loss = F.mse_loss(est, u, reduction="sum") / (torch.sum(mask) * u.shape[1])
        return loss, y

class FlowMatchingModule(nn.Module):
    def __init__(
        self,
        spk_emb_dim,
        emo_emb_dim,
        conv_emb_dim=None,
        n_feats=80,
        decoder_config=None
    ):
        super().__init__()
        self.spk_emb_dim = spk_emb_dim
        self.emo_emb_dim = emo_emb_dim
        self.n_feats = n_feats
        self.decoder = CFM(
            n_feats=n_feats,
            out_channel=n_feats,
            cfm_params={'name': 'CFM', 'solver': 'euler', 'sigma_min': 1e-06},
            decoder_params=decoder_config,
            spk_emb_dim=n_feats//2,
            emo_emb_dim=n_feats//2,
            conv_emb_dim=n_feats//2 if conv_emb_dim is not None else 0
        )
        self.spk_fc = nn.Linear(spk_emb_dim, n_feats//2)
        self.emo_fc = nn.Linear(emo_emb_dim, n_feats//2)
        self.conv_emb_dim = conv_emb_dim
        if conv_emb_dim is not None:
            self.conv_fc = nn.Linear(conv_emb_dim, n_feats//2)

    def forward(self, coarse_est, spk, emo, conv_out, tgt_fbank, x_mask, feat_len):
        
        conv_conds = None
        # if self.conv_emb_dim is not None:
        #     conv_out = self.conv_fc(conv_out)
        #     conv_conds = torch.zeros(conv_out.shape, device=conv_out.device)
        #     for i, j in enumerate(feat_len):
        #         if random.random() < 0.5:
        #             continue
        #         index = random.randint(0, int(0.25 * j))
        #         conv_conds[i, :index] = conv_out[i, :index]
        #     conv_conds = conv_conds.permute(0, 2, 1)
        
        spk = self.spk_fc(spk)
        emo = self.emo_fc(emo)
        coarse_est = coarse_est.permute(0, 2, 1)
        spk = spk.permute(0, 2, 1)
        tgt_fbank = tgt_fbank.permute(0, 2, 1)
        emo = emo.permute(0, 2, 1)
        # Compute loss of the decoder
        diff_loss, _ = self.decoder.compute_loss(x1=tgt_fbank, mask=x_mask, mu=coarse_est, spks=spk, emo=emo, conv_f=conv_conds)
        return diff_loss

    def synthesise(self, coarse_est, spk, emo, conv_f, x_mask, n_timesteps, feat_len, temperature=1.0):
        spk = self.spk_fc(spk)
        emo = self.emo_fc(emo)
        
        conv_f = None
        if self.conv_emb_dim is not None:
            conv_f = self.conv_fc(conv_f)
            conv_conds = torch.zeros(emo.shape, device=conv_f.device)
            for i, j in enumerate(feat_len):
                conv_conds[i, :len(conv_f[i])] = conv_f[i, :]
            conv_f = conv_conds.permute(0, 2, 1)
            
        coarse_est = coarse_est.permute(0, 2, 1)
        spk = spk.permute(0, 2, 1)
        emo = emo.permute(0, 2, 1)
        # Generate sample tracing the probability flow
        decoder_outputs = self.decoder(coarse_est, x_mask, n_timesteps, temperature, spk, emo, conv_f=conv_f)
        return decoder_outputs
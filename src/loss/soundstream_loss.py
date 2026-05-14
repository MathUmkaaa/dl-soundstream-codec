import torch
from torch import nn
import torch.nn.functional as F
import torchaudio


class MelLoss(nn.Module):
    def __init__(self, sample_rate=16000, n_mels=64):
        super().__init__()
        self.mel_transforms = nn.ModuleList()
        self.alphas = []
        for s in range(6, 12):
            w = 2 ** s
            mel = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate, n_fft=w, win_length=w,
                hop_length=w // 4, n_mels=n_mels, power=1.0
            )
            self.mel_transforms.append(mel)
            self.alphas.append((w / 2) ** 0.5)

    def forward(self, x, x_hat):
        min_len = min(x.shape[-1], x_hat.shape[-1])
        x = x[..., :min_len]
        x_hat = x_hat[..., :min_len]
        loss = 0.0
        for mel, alpha in zip(self.mel_transforms, self.alphas):
            s = mel(x)
            s_hat = mel(x_hat)
            min_t = min(s.shape[-1], s_hat.shape[-1])
            s = s[..., :min_t]
            s_hat = s_hat[..., :min_t]
            l1 = F.l1_loss(s_hat, s)
            l2 = F.mse_loss(torch.log(s_hat + 1e-5), torch.log(s + 1e-5)).sqrt()
            loss = loss + l1 + alpha * l2
        return loss


class AdvLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def d_loss(self, real_logits, fake_logits):
        loss = 0.0
        for r, f in zip(real_logits, fake_logits):
            loss = loss + F.relu(1 - r).mean() + F.relu(1 + f).mean()
        return loss / len(real_logits)

    def g_loss(self, fake_logits):
        loss = 0.0
        for f in fake_logits:
            loss = loss - f.mean()
        return loss / len(fake_logits)


class FeatLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, fmaps_real, fmaps_fake):
        loss = 0.0
        n = 0
        for fmap_r, fmap_f in zip(fmaps_real, fmaps_fake):
            for r, f in zip(fmap_r[:-1], fmap_f[:-1]):
                loss = loss + (r - f).abs().mean()
                n += 1
        return loss / n


class SoundStreamLoss(nn.Module):
    def __init__(self, lambda_adv=1.0, lambda_feat=100.0, lambda_rec=1.0, lambda_commit=1.0, sample_rate=16000, n_mels=64):
        super().__init__()
        self.lambda_adv = lambda_adv
        self.lambda_feat = lambda_feat
        self.lambda_rec = lambda_rec
        self.lambda_commit = lambda_commit
        self.mel_loss = MelLoss(sample_rate=sample_rate, n_mels=n_mels)
        self.adv_loss = AdvLoss()
        self.feat_loss = FeatLoss()

    def forward(self, x, x_hat, fmaps_real, fmaps_fake, commit_loss):
        fake_logits = [fm[-1] for fm in fmaps_fake]
        l_rec = self.mel_loss(x, x_hat)
        l_adv = self.adv_loss.g_loss(fake_logits)
        l_feat = self.feat_loss(fmaps_real, fmaps_fake)
        l_w = commit_loss
        total = self.lambda_rec * l_rec + self.lambda_adv * l_adv + self.lambda_feat * l_feat + self.lambda_commit * l_w
        return total, {
            "loss_rec": l_rec.detach(),
            "loss_adv": l_adv.detach(),
            "loss_feat": l_feat.detach(),
            "loss_commit": l_w.detach(),
        }

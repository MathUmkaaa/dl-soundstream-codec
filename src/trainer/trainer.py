import torch
from torchmetrics.audio import ShortTimeObjectiveIntelligibility
from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer
from src.model.discriminator import MultiDiscriminator

class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def __init__(self, *args, discriminator=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss = self.criterion
        if discriminator is None:
            discriminator = MultiDiscriminator()
        self.discriminator = discriminator.to(self.device)
        self.stoi = ShortTimeObjectiveIntelligibility(fs=16000, extended=False).to(self.device)
        self.opt_g, self.opt_d = self._get_opts()

    def _get_opts(self):
        if isinstance(self.optimizer, dict):
            opt_g = self.optimizer["model"]
            opt_d = self.optimizer["discriminator"]
            return opt_g, opt_d
        opt_d = getattr(self, "optimizer_d", None)
        if opt_d is None:
            opt = self.optimizer.__class__
            vals = self.optimizer.defaults.copy()
            opt_d = opt(self.discriminator.parameters(), **vals)
        return self.optimizer, opt_d

    def _get_x(self, batch):
        if "audio" in batch:
            x = batch["audio"]
        elif "waveform" in batch:
            x = batch["waveform"]
        else:
            raise KeyError("audio")
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return x

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)

        x = self._get_x(batch)
        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.discriminator.train()

            x_hat, commit_loss, indices = self.model(x)
            fmaps_real = self.discriminator(x)
            fmaps_fake = self.discriminator(x_hat.detach())
            d_real = [fm[-1] for fm in fmaps_real]
            d_fake = [fm[-1] for fm in fmaps_fake]
            loss_d = self.loss.adv_loss.d_loss(d_real, d_fake)

            self.opt_d.zero_grad()
            loss_d.backward()
            self.opt_d.step()

            for p in self.discriminator.parameters():
                p.requires_grad_(False)

            fmaps_real = self.discriminator(x)
            fmaps_fake = self.discriminator(x_hat)
            fmaps_real = [[f.detach() for f in fm] for fm in fmaps_real]
            loss_g, losses = self.loss(x, x_hat, fmaps_real, fmaps_fake, commit_loss)

            self.opt_g.zero_grad()
            loss_g.backward()
            self._clip_grad_norm()
            self.opt_g.step()

            for p in self.discriminator.parameters():
                p.requires_grad_(True)

        else:
            self.discriminator.eval()
            x_hat, commit_loss, indices = self.model(x)
            fmaps_real = self.discriminator(x)
            fmaps_fake = self.discriminator(x_hat)
            fmaps_real = [[f.detach() for f in fm] for fm in fmaps_real]
            loss_g, losses = self.loss(x, x_hat, fmaps_real, fmaps_fake, commit_loss)
            loss_d = self.loss.adv_loss.d_loss(
                [fm[-1] for fm in fmaps_real], [fm[-1] for fm in fmaps_fake]
            )
            batch["STOI"] = self.stoi(x_hat.squeeze(1), x.squeeze(1))

        batch["x"] = x
        batch["x_hat"] = x_hat
        batch["loss"] = loss_g
        batch["loss_g"] = loss_g.detach()
        batch["loss_d"] = loss_d.detach()
        batch.update(losses)
        probs = torch.zeros(self.model.rvq.quantizers[0].codebook_size, device=indices.device)
        for i in range(indices.shape[1]):
            probs = probs + torch.bincount(indices[:, i, :].flatten(), minlength=probs.shape[0])
        probs = probs / probs.sum()
        perp = torch.exp(-(probs * torch.log(probs + 1e-10)).sum())
        batch["codebook_perplexity"] = perp.detach()

        for loss_name in ["loss", "loss_d", "loss_g", "loss_rec", "loss_adv", "loss_feat", "loss_commit", "codebook_perplexity", "STOI"]:
            if loss_name in batch:
                metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        if mode == "train":
            return
        if batch_idx != 0:
            return
        x = batch["x"][0].detach().cpu()
        x_hat = batch["x_hat"][0].detach().cpu().clamp(-1, 1)
        self.writer.add_audio("audio/real", x, 16000)
        self.writer.add_audio("audio/recon", x_hat, 16000)

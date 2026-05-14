import argparse
from pathlib import Path

import soundfile
import torch
import torchaudio
from tqdm import tqdm
from torchmetrics.audio import ShortTimeObjectiveIntelligibility
from torchmetrics.audio.nisqa import NonIntrusiveSpeechQualityAssessment

from src.model.soundstream import SoundStream

SR = 16000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    device = args.device
    model = SoundStream().to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    model.load_state_dict(ckpt)
    model.eval()

    stoi = ShortTimeObjectiveIntelligibility(SR).to(device)
    nisqa = NonIntrusiveSpeechQualityAssessment(SR).to(device)
    files = sorted(Path(args.data_dir).rglob("*.flac"))
    if args.limit is not None:
        files = files[: args.limit]

    stoi_vals = []
    nisqa_vals = []
    with torch.no_grad():
        for path in tqdm(files):
            wav, sr = soundfile.read(str(path))
            x = torch.tensor(wav, dtype=torch.float32)
            if sr != SR:
                x = torchaudio.functional.resample(x, sr, SR)
            x = x.unsqueeze(0).unsqueeze(0).to(device)

            x_hat, commit_loss, indices = model(x)
            stoi_val = stoi(x_hat.squeeze(1), x.squeeze(1)).item()
            stoi_vals.append(stoi_val)

            nisqa_out = nisqa(x_hat.squeeze(1))
            if nisqa_out.dim() == 2:
                nisqa_out = nisqa_out[0]
            nisqa_vals.append(nisqa_out[0].item())

    mean_stoi = sum(stoi_vals) / len(stoi_vals)
    mean_nisqa = sum(nisqa_vals) / len(nisqa_vals)
    print("STOI", mean_stoi)
    print("NISQA", mean_nisqa)


if __name__ == "__main__":
    main()

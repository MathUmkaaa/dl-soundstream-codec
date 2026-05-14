import logging
from pathlib import Path
import torch, torchaudio
import torch.nn.functional as F
import soundfile as sf

from src.datasets.base_dataset import BaseDataset

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000 # частота дискретизации
CROP_SAMPLES = 8000 # количесвто

class LibriSpeechDataset(BaseDataset):
    def __init__(self, data_dir, name="train", limit=None, *args, **kwargs):
        self.data_dir = Path(data_dir)
        self.name = name
        
        flac_audio_files = self.data_dir.rglob('*.flac')
        index = []
        for path in flac_audio_files:
            index.append({"path": str(path), "label": 0})

        if limit is not None:
            index = index[:limit]
        super().__init__(index, *args, **kwargs)

    def __getitem__(self, ind):
        path = self._index[ind]["path"]
        waveform_np, sampling_rate = sf.read(path, dtype="float32", always_2d=True)
        waveform = torch.from_numpy(waveform_np).transpose(0, 1)
        waveform = torchaudio.functional.resample(waveform, sampling_rate, SAMPLE_RATE)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if self.name != "train":
            return {"audio": waveform, "label": 0}
        if waveform.shape[-1] < CROP_SAMPLES:
            waveform = F.pad(waveform, (0, CROP_SAMPLES - waveform.shape[-1]), mode="replicate")
    
        start = torch.randint(0, waveform.shape[-1] - CROP_SAMPLES + 1, (1,)).item()
        waveform = waveform[:, start : start + CROP_SAMPLES]
        
        return {"audio": waveform, "label": 0}
# SoundStream Neural Audio Codec

<p align="center">
  <a href="#final-metrics-on-test-clean">Metrics</a> •
  <a href="#download-pretrained-checkpoint">Checkpoint</a> •
  <a href="#training">Training</a> •
  <a href="#evaluation-stoi--nisqa-on-test-clean">Evaluation</a> •
  <a href="#demo">Demo</a> •
  <a href="#analysis">Analysis</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

<p align="center">
<a href="https://arxiv.org/abs/2107.03312">
  <img src="https://img.shields.io/badge/arxiv-SoundStream-blue">
</a>
<a href="https://www.comet.com/theumka2406/soundstream/reports/Jx1puow7dXLqXNPvHc6UbuJwY">
  <img src="https://img.shields.io/badge/CometML-report-orange">
</a>
<a href="LICENSE">
  <img src="https://img.shields.io/badge/license-MIT-blue.svg">
</a>
</p>

Implementation of the [SoundStream](https://arxiv.org/abs/2107.03312) neural audio codec for 16 kHz speech at ~6 kbps. Trained on LibriSpeech `train-clean-100`, evaluated on `test-clean`.

This repo is built on top of the [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template) by Blinorot

## Final metrics on `test-clean`

STOI = 0.9147
NISQA v2.0 = 2.5254

CometML report: https://www.comet.com/theumka2406/soundstream/reports/Jx1puow7dXLqXNPvHc6UbuJwY

## Download pretrained checkpoint

```bash
pip install gdown
python scripts/download_checkpoint.py
```

Edit the Google Drive `id=...` in `scripts/download_checkpoint.py` to point to your own checkpoint

## Training

```bash
python train.py --config-name=sound_stream \
  trainer.n_epochs=45 trainer.epoch_len=1000 \
  datasets.train.data_dir=PATH_TO_LIBRISPEECH/train-clean-100 \
  datasets.test.data_dir=PATH_TO_LIBRISPEECH/test-clean \
  trainer.save_dir=./saved writer.run_name=ss_v2
```

## Evaluation (STOI + NISQA on test-clean)

```bash
python eval.py --ckpt checkpoints/model_best.pth \
  --data_dir PATH_TO_LIBRISPEECH/test-clean
```

Use `--limit N` for a quick smoke test on `N` files.

## Demo

See [`demo.ipynb`](demo.ipynb). Open in Google Colab, `Run all`, then paste your audio URL in the third cell.

## Analysis

See [`report.ipynb`](report.ipynb) for analysis (LibriSpeech, LJSpeech, Russian).

## Credits

Based on the [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template) by Blinorot, which is itself a heavily modified fork of [pytorch-template](https://github.com/victoresque/pytorch-template) and [asr_project_template](https://github.com/WrathOfGrapes/asr_project_template).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)

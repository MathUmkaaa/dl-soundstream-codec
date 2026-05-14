import torch


def collate_fn(dataset_items: list[dict]):
    """
    Collects a batch from dicts {"audio": Tensor[1, 8000], "label": int}
    Returns {"audio": Tensor[B, 1, 8000], "label": Tensor[B]}

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    result_batch = {}

    audios = [elem["audio"] for elem in dataset_items]
    lengths = [audio.shape[-1] for audio in audios]
    if len(set(lengths)) == 1:
        result_batch["audio"] = torch.stack(audios)
    else:
        max_len = max(lengths)
        padded = []
        for audio in audios:
            pad = max_len - audio.shape[-1]
            if pad > 0:
                audio = torch.nn.functional.pad(audio, (0, pad))
            padded.append(audio)
        result_batch["audio"] = torch.stack(padded)
        result_batch["audio_lengths"] = torch.tensor(lengths)
    result_batch["label"] = torch.tensor([elem["label"] for elem in dataset_items])

    return result_batch

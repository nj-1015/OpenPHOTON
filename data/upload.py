"""Shard upload stub (plan slide 31: "storage OFF Drive -- primary HF Hub
private dataset, alt GCS; Drive's ~20-50MB/s starves an A100 outright").

Scaffolding only -- neither function is called anywhere in this repo's
tests or trainer; both need live credentials (HF token / GCS service
account) that this local, zero-CU pass does not have and should not
fabricate. Wire the trainer's `shard_dir` (configs/s1.py) at whichever URI
scheme the caller actually uploaded to (`hf://...` via `datasets`/
`huggingface_hub`, or `gs://...` via a small io shim -- `stream_dataset.py`
only needs a local path today; the io shim is the same kind of seam
`checkpoint/ckpt.py`'s docstring already calls out for its own GCS TODO).
"""
import os


def upload_to_hf_hub(shard_dir: str, repo_id: str, private: bool = True, token: str | None = None) -> str:
    """Primary path (plan slide 31). Uploads every file in `shard_dir`
    (shard .bin/.json + index.json) to a private HF Hub dataset repo.
    Requires `huggingface_hub` (already a transitive dep of `transformers`)
    and a real HF token with write access -- not exercised by any test
    here. Returns the resulting `hf://datasets/{repo_id}` URI.
    """
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(folder_path=shard_dir, repo_id=repo_id, repo_type="dataset",
                       path_in_repo=".", commit_message="upload S1 shards")
    return f"hf://datasets/{repo_id}"


def upload_to_gcs(shard_dir: str, bucket: str, prefix: str = "photon-qwen3-s1-shards") -> str:
    """Alt path (plan slide 31). Requires `google-cloud-storage` (not in
    requirements.txt -- add it only if the GCS path is actually chosen)
    and real bucket credentials. Returns the resulting `gs://...` URI.
    """
    try:
        from google.cloud import storage  # optional dependency
    except ImportError as e:
        raise NotImplementedError(
            "upload_to_gcs requires `pip install google-cloud-storage` -- "
            "not installed and not exercised by this local pass"
        ) from e

    client = storage.Client()
    bucket_obj = client.bucket(bucket)
    for fname in sorted(os.listdir(shard_dir)):
        local_path = os.path.join(shard_dir, fname)
        if not os.path.isfile(local_path):
            continue
        blob = bucket_obj.blob(f"{prefix}/{fname}")
        blob.upload_from_filename(local_path)
    return f"gs://{bucket}/{prefix}"

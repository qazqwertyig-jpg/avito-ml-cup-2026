"""Build the val item priors from the public archive without storing train.

``item_priors.py`` needs the complete train impression shards on disk to count item
exposure and positives per day. The train split does not fit on this machine, so this
script streams the same shards out of ``competition.zip`` through HTTP range requests,
counts the identical quantities with the identical day grid, and then reuses
``item_priors.write_augmented`` to append the four prior columns to a val cache.

Usage:
    python experiments/stream_item_priors.py <val-cache> <out> [--workers 12]
"""
import argparse
import io
import json
import sys
import time
import urllib.request
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from remotezip import RemoteZip

sys.path.insert(0, str(Path(__file__).resolve().parent))

from item_priors import DAY, DAYS, START, write_augmented  # noqa: E402

URL = 'https://storage.yandexcloud.net/avitotech-mlcup-2026/competition.zip'
PREFIX = 'competition/train/impressions/'


def read_member_blob(url, member_info):
    """Download a zip member via HTTP Range, inflate if needed."""
    head = urlopen_range(url, member_info.header_offset, member_info.header_offset + 4096)
    if head[:4] != b'PK\x03\x04':
        raise ValueError(f'bad local header for {member_info.filename}')
    name_length = int.from_bytes(head[26:28], 'little')
    extra_length = int.from_bytes(head[28:30], 'little')
    data_start = member_info.header_offset + 30 + name_length + extra_length
    blob = urlopen_range(url, data_start, data_start + member_info.compress_size)
    if member_info.compress_type == zipfile.ZIP_STORED:
        return blob
    if member_info.compress_type != zipfile.ZIP_DEFLATED:
        raise ValueError(f'unsupported compression {member_info.compress_type}')
    return zlib.decompressobj(-15).decompress(blob)


def urlopen_range(url, start, end):
    """GET with Range header."""
    req = urllib.request.Request(url,
                                 headers={'Range': f'bytes={start}-{end - 1}', 'Host': 'storage.yandexcloud.net'})
    with urllib.request.urlopen(req, timeout=180) as response:
        return response.read()


def shard_counts(url, member_info, mapping):
    """Count flat (day, item) indices and labels for one shard."""
    blob = read_member_blob(url, member_info)
    table = pq.read_table(io.BytesIO(blob),
                          columns=['timestamp', 'item_id', 'target']).combine_chunks()
    item = table['item_id'].chunk(0)
    values = item.flatten().to_numpy()
    local = np.full(len(values), -1, dtype=np.int32)
    valid = (values > 0) & (values < len(mapping))
    local[valid] = mapping[values[valid]]
    valid = local >= 0
    day = (table['timestamp'].to_numpy() - START) // DAY
    if np.any((day < 0) | (day >= DAYS)):
        raise ValueError('train timestamp outside supported days')
    candidate_day = np.repeat(day, np.diff(item.offsets.to_numpy()))
    labels = table['target'].chunk(0).flatten().to_numpy()
    return candidate_day[valid].astype(np.int64), local[valid].astype(np.int32), labels[valid].astype(np.int8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cache', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--url', default=URL)
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--checkpoint', type=int, default=32)
    args = parser.parse_args()

    with np.load(args.cache / 'val.npz') as handle:
        items = np.unique(handle['item'])
    if items[0] < 0:
        raise ValueError('negative item ID in the val cache')
    mapping = np.full(int(items[-1]) + 1, -1, dtype=np.int32)
    mapping[items] = np.arange(len(items), dtype=np.int32)
    size = len(items)
    print(f'{size} val item IDs, mapping {len(mapping)} slots', flush=True)

    with RemoteZip(args.url) as archive:
        infos = [m for m in archive.infolist()
                 if m.filename.startswith(PREFIX) and m.filename.endswith('.parquet')]
    print(f'{len(infos)} train shards', flush=True)

    exposure = np.zeros((DAYS + 1) * size, dtype=np.int32)
    positive = np.zeros_like(exposure)
    started = time.time()
    done = 0

    def process_one(info):
        days, local, labels = shard_counts(args.url, info, mapping)
        flat = days * size + local
        np.add.at(exposure, flat, 1)
        np.add.at(positive, flat[(labels == 1)], 1)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process_one, info) for info in infos]
        for future in futures:
            future.result()
            done += 1
            if done % args.checkpoint == 0 or done == len(infos):
                elapsed = time.time() - started
                print(f'aggregated {done}/{len(infos)} shards, {elapsed:.0f}s, '
                      f'{done * 18.1 / elapsed:.1f} MiB/s', flush=True)
                np.savez(args.out.parent / 'item_prior_counters.npz',
                         exposure=exposure.reshape(DAYS + 1, size),
                         positive=positive.reshape(DAYS + 1, size), items=items)

    exposure = np.cumsum(exposure.reshape(DAYS + 1, size), axis=0, dtype=np.int32)
    positive = np.cumsum(positive.reshape(DAYS + 1, size), axis=0, dtype=np.int32)
    args.out.mkdir(parents=True, exist_ok=True)
    report = {'val': write_augmented(args.cache / 'val.npz', args.out / 'val.npz',
                                     mapping, exposure, positive)}
    (args.out / 'profile.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
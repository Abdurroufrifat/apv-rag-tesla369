"""Build a disk-backed FTS5 index from the verified official FEVER archive."""

import argparse
from pathlib import Path

from apv_rag.fever_corpus import build_index


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path,
                        default=root / 'data/external/fever/heldout_v1/wiki-pages.zip')
    parser.add_argument('--index', type=Path,
                        default=root / 'data/processed/fever/heldout_v1/wiki.sqlite')
    args = parser.parse_args()
    print('Verifying the FEVER archive SHA-256...', flush=True)
    result = build_index(args.archive, args.index,
                         progress=lambda message: print(message, flush=True))
    print(f"FEVER index complete: {result['pages']} pages; archive SHA-256 {result['archive_sha256']}")
    print(f"Source records: {result['source_rows']}; audited empty placeholders: {result['empty_placeholders']}")
    print(f'Index: {args.index}')


if __name__ == '__main__':
    main()

# Copied CAA sycophancy source

`generate_dataset.json` is a byte-identical copy of
[`nrimsky/CAA` `datasets/generate/sycophancy/generate_dataset.json`](https://github.com/nrimsky/CAA/blob/5dabbbd9a0bca5f25e174501e959de378806aa48/datasets/generate/sycophancy/generate_dataset.json)
at revision `5dabbbd9a0bca5f25e174501e959de378806aa48`.

CAA's [raw-dataset script](https://github.com/nrimsky/CAA/blob/5dabbbd9a0bca5f25e174501e959de378806aa48/datasets/raw/sycophancy/make_dataset.py#L4-L9)
combines the NLP survey and political typology sycophancy JSONL files. Its
[processing script](https://github.com/nrimsky/CAA/blob/5dabbbd9a0bca5f25e174501e959de378806aa48/process_raw_datasets.py#L18-L40)
removes `Question:` and `Answer:` text, strips surrounding answer whitespace, shuffles the rows,
uses the first 1,000 for generation, and puts the last 50 in its test split. The script does not
set a random seed. This study uses the committed generation file to fix the source order.

Original data: [Anthropic sycophancy](https://github.com/anthropics/evals/blob/84fcc677e52e1902d696c32cd1a6b663e70d3993/sycophancy/README.md).
Licences: CAA — MIT; Anthropic — CC-BY-4.0. The Anthropic revision used by CAA is
unknown. Only CAA's generation split is copied here.

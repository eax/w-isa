# Evaluation corpus provenance

The downloader produces a local JSON cache containing all 4,358 rows of the
`wikitext-2-raw-v1` test split from
[Salesforce/WikiText](https://huggingface.co/datasets/Salesforce/wikitext).
The corpus is derived from Wikipedia articles. Attribution belongs to the
Wikipedia contributors and the WikiText dataset authors: Stephen Merity,
Caiming Xiong, James Bradbury, and Richard Socher, *Pointer Sentinel Mixture
Models* (2016). Consult the upstream dataset card for CC BY-SA licensing and
redistribution terms. The downloaded text retains its upstream attribution.
The cache is excluded from Git and Python distributions; run the downloader
before corpus evaluation on a new checkout. The code and dataset carry their
respective licenses.

The [downloader](../examples/fetch_wikitext.py) retrieves the public dataset-viewer
rows, validates every row index and rejects truncated cells, then preserves the
row strings unchanged in JSON. Evaluation joins them with two newline characters,
including empty rows. Dataset identity, row count, separator and text SHA-256
are embedded in the file. The recorded perplexity measurements use the test split.
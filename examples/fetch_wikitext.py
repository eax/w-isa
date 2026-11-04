"""Download the complete public WikiText-2 raw test split without datasets/pandas."""

import argparse
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request


def fetch(output: Path):
    rows = []
    total = None
    endpoint = "https://datasets-server.huggingface.co/rows"
    while total is None or len(rows) < total:
        query = urllib.parse.urlencode({"dataset": "Salesforce/wikitext",
                                       "config": "wikitext-2-raw-v1", "split": "test",
                                       "offset": len(rows), "length": 100})
        with urllib.request.urlopen(endpoint + "?" + query, timeout=90) as response:
            payload = json.load(response)
        if total is not None and total != payload["num_rows_total"]:
            raise RuntimeError("dataset size changed during download")
        total = payload["num_rows_total"]
        if not payload["rows"]:
            raise RuntimeError("dataset server returned an incomplete split")
        for row in payload["rows"]:
            if row["row_idx"] != len(rows) or row.get("truncated_cells"):
                raise RuntimeError("missing or truncated dataset row")
            rows.append(row["row"]["text"])
    text = "\n\n".join(rows)
    data = {"dataset": "Salesforce/wikitext", "config": "wikitext-2-raw-v1", "split": "test",
            "source": endpoint, "row_count": len(rows), "separator": "\n\n",
            "text_sha256": hashlib.sha256(text.encode()).hexdigest(), "rows": rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in data.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/wikitext2-test.json"))
    fetch(parser.parse_args().output)
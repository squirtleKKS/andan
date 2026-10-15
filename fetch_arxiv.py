import argparse
from datetime import datetime, timezone
from pathlib import Path
import re
import time
import xml.etree.ElementTree as ET

import pandas as pd
import requests

API_URL = "https://export.arxiv.org/api/query"
NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom",
      "opensearch": "http://a9.com/-/spec/opensearch/1.1/"}
HEADERS = {"User-Agent": "andan-entity-resolution-lab/1.0 (educational research)"}
PHRASES = ["entity resolution", "entity matching", "record linkage", "fuzzy matching", "approximate string matching"]
COLUMNS = ["arxiv_id", "title", "abstract", "published", "authors", "primary_category", "categories"]
FINAL_PATH = Path(__file__).resolve().parent / "data" / "papers.csv"
PAGE_SIZE = 100
PAUSE = 8


def fetch_entries(query, start=0, page_size=PAGE_SIZE, max_attempts=5):
    params = {"search_query": query, "start": start, "max_results": page_size,
              "sortBy": "submittedDate", "sortOrder": "descending"}
    for attempt in range(max_attempts):
        try:
            response = requests.get(API_URL, params=params, timeout=60, headers=HEADERS)
            if 400 <= response.status_code < 500 and response.status_code != 429:
                response.raise_for_status()
            if response.status_code == 200:
                root = ET.fromstring(response.content)
                if any("api/errors" in entry.findtext("atom:id", "", NS)
                       for entry in root.findall("atom:entry", NS)):
                    raise ValueError("arXiv API вернул сообщение об ошибке")
                return root
            reason = f"HTTP {response.status_code}"
        except (requests.ConnectionError, requests.Timeout, ET.ParseError) as error:
            reason = type(error).__name__
        if attempt + 1 < max_attempts:
            delay = min(15 * 2 ** attempt, 180)
            print(f"{reason}; повтор {attempt + 2}/{max_attempts} через {delay} с", flush=True)
            time.sleep(delay)
    raise RuntimeError(f"arXiv API недоступен: {reason}")


def iter_entries(query):
    start = 0
    previous_page = None
    while True:
        root = fetch_entries(query, start=start)
        entries = root.findall("atom:entry", NS)
        total = int(root.findtext("opensearch:totalResults", namespaces=NS))
        actual_start = int(root.findtext("opensearch:startIndex", namespaces=NS))
        if actual_start != start:
            raise RuntimeError(f"API вернул страницу с позиции {actual_start} вместо {start}")
        if not entries:
            if start < total:
                raise RuntimeError(f"Пустая страница: загружено {start} из {total} результатов")
            time.sleep(PAUSE)
            return
        page_ids = tuple(entry.findtext("atom:id", "", NS) for entry in entries)
        if page_ids == previous_page:
            raise RuntimeError("API повторно вернул предыдущую страницу")
        previous_page = page_ids
        yield from entries
        start += len(entries)
        print(f"  Загружено {start} из {total}", flush=True)
        time.sleep(PAUSE)
        if start >= total:
            return


def parse_entry(entry):
    raw_id = entry.findtext("atom:id", "", NS)
    arxiv_id = re.sub(r"v\d+$", "", raw_id.split("/abs/")[-1])
    title = " ".join(entry.findtext("atom:title", "", NS).split())
    abstract = " ".join(entry.findtext("atom:summary", "", NS).split())
    published = entry.findtext("atom:published", "", NS)
    authors = [" ".join(author.findtext("atom:name", "", NS).split())
               for author in entry.findall("atom:author", NS)]
    if not arxiv_id or not title or not abstract or not published or not any(authors):
        return None
    primary = entry.find("arxiv:primary_category", NS)
    return {"arxiv_id": arxiv_id, "title": title, "abstract": abstract,
            "published": published, "authors": "; ".join(a for a in authors if a),
            "primary_category": primary.get("term", "") if primary is not None else "",
            "categories": "; ".join(c.get("term", "") for c in entry.findall("atom:category", NS))}


def validate_corpus(df):
    if len(df) < 300:
        raise ValueError(f"Получено только {len(df)} статей, нужно не менее 300")
    required = ["arxiv_id", "title", "abstract", "published", "authors"]
    if not df[required].notna().all().all() or not df[required].apply(lambda s: s.str.strip().ne("")).all().all():
        raise ValueError("Пропуски в обязательных полях")
    if not df["arxiv_id"].is_unique:
        raise ValueError("Обнаружены дубликаты arXiv id")
    dates = pd.to_datetime(df["published"], utc=True, errors="raise")
    if not dates.between(pd.Timestamp("2020-01-01", tz="UTC"), pd.Timestamp.now(tz="UTC")).all():
        raise ValueError("Дата публикации вне допустимого диапазона")


def main():
    parser = argparse.ArgumentParser(description="Сбор тематического корпуса через arXiv API")
    parser.add_argument("--refresh", action="store_true", help="Повторно собрать корпус и заменить CSV после проверки")
    args = parser.parse_args()
    if FINAL_PATH.exists() and not args.refresh:
        df = pd.read_csv(FINAL_PATH, dtype=str)
        validate_corpus(df)
        print(f"Готовый корпус: {len(df)} статей. Для повторной выгрузки: --refresh")
        return
    now = datetime.now(timezone.utc)
    records = {}
    for year in range(2020, now.year + 1):
        end = min(now, datetime(year, 12, 31, 23, 59, tzinfo=timezone.utc))
        for phrase in PHRASES:
            query = f'all:"{phrase}" AND submittedDate:[{year}01010000 TO {end:%Y%m%d%H%M}]'
            print(f"{year}: {phrase}", flush=True)
            for entry in iter_entries(query):
                record = parse_entry(entry)
                if record is not None and record["published"].startswith(str(year)):
                    records[record["arxiv_id"]] = record
    df = pd.DataFrame(records.values(), columns=COLUMNS)
    if not df.empty:
        df = df.sort_values(["published", "arxiv_id"]).reset_index(drop=True)
    validate_corpus(df)
    FINAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = FINAL_PATH.with_suffix(".csv.tmp")
    df.to_csv(temporary_path, index=False)
    temporary_path.replace(FINAL_PATH)
    print(f"Сохранено {len(df)} статей: {FINAL_PATH}")
    print(df["published"].str[:4].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()

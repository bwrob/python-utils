import random
import time
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm, trange


def make_request(url: str) -> int:
    """Simulate network delay with multiple tasks."""
    for _ in trange(10, desc=f"Processing {url}"):
        time.sleep(random.uniform(0.1, 0.5))
    return 0


def make_requests(
    urls: list[str],
    i: int,
) -> Iterator[int]:
    """Process a batch of URLs in thread pool with a progress bar."""
    with (
        tqdm(total=len(urls), leave=False, desc=f"Batch {i}") as pbar,
        ProcessPoolExecutor(max_workers=5) as executor,
    ):
        futures = [executor.submit(make_request, url) for url in urls]
        for future in as_completed(futures):
            _ = pbar.update(1)
            yield future.result()


def main() -> None:
    urls = ["https://example.com/page" + str(i) for i in range(10)]
    results = [
        result
        for i in trange(5, desc="Main loop", leave=True)
        for result in make_requests(urls, i)
    ]
    print(results)


if __name__ == "__main__":
    main()

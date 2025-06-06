from functools import partial
from multiprocessing import Pool, RLock, freeze_support
from random import random
from time import sleep

from tqdm.auto import tqdm, trange
from tqdm.contrib.concurrent import process_map

NUM_SUBITERS = 30


def progresser(n, auto_position=True, write_safe=False, blocking=True, progress=False):
    interval = random() * 0.5 / (NUM_SUBITERS - n + 2)  # nosec
    total = 100
    text = f"#{n}, est. {interval * total:<04.2g}s"
    for _ in trange(
        total,
        desc=text,
        disable=not progress,
        lock_args=None if blocking else (False,),
        position=None if auto_position else n,
        leave=False,
    ):
        sleep(interval)
    # NB: may not clear instances with higher `position` upon completion
    # since this worker may not know about other bars #796
    if write_safe:  # we think we know about other bars
        if n == 6:
            tqdm.write("n == 6 completed")
    return n + 1


if __name__ == "__main__":
    freeze_support()  # for Windows support
    L = list(range(NUM_SUBITERS))[::-1]

    tqdm.set_lock(RLock())

    print("Simple process mapping")
    process_map(partial(progresser), L, max_workers=4)

    print("Multi-processing")
    p = Pool(initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),))
    p.map(partial(progresser, progress=True), tqdm(L))

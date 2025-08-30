import os
from threading import Thread
from tqdm import tqdm
from src.dataset.src import iterate_urls, get_filepath, get_subfolder, load_audio, AUDIO_DIR, YouTubeURL
from src.btc import LargeBTCExtractor
from queue import Queue, Empty

BASE_PATH = "E:/data/latents/btc_large/"


def process_url_worker(q: Queue[YouTubeURL], results: Queue[str]):
    """Process a single URL"""
    extractor = LargeBTCExtractor()  # Create extractor per worker
    while True:
        try:
            url = q.get_nowait()
        except Empty:
            break

        try:
            audio_path = get_filepath(AUDIO_DIR, url.video_id)
            output_dir = get_subfolder(BASE_PATH, url.video_id)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_path = os.path.join(output_dir, f"{url.video_id}.npz")
            if os.path.exists(output_path):
                results.put(f"Skipped {url.video_id}, already exists.")
                continue
        except Exception as e:
            results.put(f"Error preparing paths for {url.video_id}: {e}")
            continue

        try:
            audio, sr = load_audio(audio_path)
        except Exception as e:
            results.put(f"Error loading audio for {audio_path}: {e}")
            continue

        assert sr is not None

        try:
            latents = extractor.extract(audio, sr)
            latents.save(output_path)
            results.put(f"Processed {url.video_id}, saved to {output_path}.")
        except Exception as e:
            results.put(f"Error processing {url.video_id}: {e}")


def main():
    if not os.path.exists(BASE_PATH):
        os.makedirs(BASE_PATH)
        print(f"Created directory: {BASE_PATH}")

    urls = list(iterate_urls())
    q = Queue()
    results = Queue()
    pbar = tqdm(total=len(urls), desc="Processing URLs")
    for url in urls:
        q.put(url)

    num_threads = 16
    threads: list[Thread] = []
    for _ in range(num_threads):
        t = Thread(target=process_url_worker, args=(q, results))
        t.start()
        threads.append(t)

    while any(t.is_alive() for t in threads) or not results.empty():
        try:
            msg = results.get(timeout=1)
            tqdm.write(msg)
            pbar.update(1)
        except Empty:
            continue

    for t in threads:
        t.join()

    pbar.close()
    print("Processing complete.")


if __name__ == "__main__":
    main()

import os
from tqdm import tqdm
from src.dataset.src import iterate_urls, get_filepath, get_subfolder, load_audio
from src.btc import LargeBTCExtractor
from src.dataset.src.constants import AUDIO_DIR

BASE_PATH = "E:/data/latents/btc_large/"


def main():
    if not os.path.exists(BASE_PATH):
        os.makedirs(BASE_PATH)
        print(f"Created directory: {BASE_PATH}")

    extractor = LargeBTCExtractor()

    for url in tqdm(iterate_urls()):
        try:
            audio_path = get_filepath(AUDIO_DIR, url.video_id)
            output_dir = get_subfolder(BASE_PATH, url.video_id)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_path = os.path.join(output_dir, f"{url.video_id}.npz")
            if os.path.exists(output_path):
                continue
        except Exception as e:
            print(f"Error initializing {url.video_id}: {e}")
            continue

        try:
            audio, sr = load_audio(audio_path)
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            continue

        assert sr is not None

        try:
            latents = extractor.extract(audio, sr)
            latents.save(output_path)
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            continue


if __name__ == "__main__":
    main()

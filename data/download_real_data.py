import urllib.request
import os
import sys

URL = "https://github.com/dbdmg/data-science-lab/raw/master/datasets/online_retail.csv"
OUT_PATH = "data/online_retail.csv"

def download_data():
    print(f"Downloading real-world UK Online Retail transaction dataset (541k rows)...")
    print(f"Source: {URL}")
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    
    # Progress hook to show download feedback
    def progress_hook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(f"\rDownloading... {percent}%")
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(URL, OUT_PATH, reporthook=progress_hook)
        print(f"\n[SUCCESS] Real-world transaction dataset successfully saved to {OUT_PATH}")
    except Exception as e:
        print(f"\n[ERROR] Failed to download dataset: {e}")

if __name__ == "__main__":
    download_data()

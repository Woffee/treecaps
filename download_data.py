import urllib.request
import zipfile
from tqdm import tqdm
import os
from pathlib import Path

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)\


code_classification_data_url = "https://ai4code.s3-ap-southeast-1.amazonaws.com/OJ_data.zip"


output_path = "/data/treecaps"
Path(output_path).mkdir(parents=True, exist_ok=True)
data_filename = output_path + "/OJ_data.zip"

download_url(code_classification_data_url, data_filename)

with zipfile.ZipFile(data_filename) as zf:
    for member in tqdm(zf.infolist(), desc='Extracting '):
        try:
            zf.extract(member, output_path)
        except zipfile.error as e:
            pass

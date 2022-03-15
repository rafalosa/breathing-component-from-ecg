import zipfile
import os
from typing import Optional
import requests
import shutil


class DBGetter:

    def __init__(self, out_dir: Optional[str] = None):
        self._out_dir = out_dir

    def download(self, url: str, db_name: Optional[str] = None) -> None:

        if self._out_dir not in os.listdir() and self._out_dir is not None:
            os.mkdir(self._out_dir)

        if db_name is not None and db_name in os.listdir(self._out_dir):
            print("Database already exists.")
            return

        if self._out_dir is not None:
            temp_path = self._out_dir + "/tmp"

            if "tmp" not in os.listdir(self._out_dir):
                os.mkdir(temp_path)

            filepath = temp_path + "/tempdatabase.zip"

        else:
            temp_path = "tmp"
            if "tmp" not in os.listdir():
                os.mkdir(temp_path)

            filepath = temp_path + "/tempdatabase.zip"

        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(filepath, 'wb') as downloaded_database:
                response.raw.decode_content = True
                shutil.copyfileobj(response.raw, downloaded_database)

        def _extract_and_cleanup(file, temp):

            with zipfile.ZipFile(file) as zipped:
                zipped.extractall(temp)
            os.remove(file)
            unzipped = os.listdir(temp)
            unzipped_name = unzipped[0]
            unzipped_path_ = temp + '/' + unzipped_name

            return unzipped_path_

        unzipped_path = _extract_and_cleanup(filepath, temp_path)

        while unzipped_path.endswith(".zip"):
            unzipped_path = _extract_and_cleanup(unzipped_path, temp_path)

        if db_name is not None:
            db_path = "/".join(unzipped_path.split("/")[:-1]) + "/" + db_name
            os.rename(unzipped_path, db_path)
            shutil.move(db_path, self._out_dir + '/' + db_name)
        else:
            shutil.move(unzipped_path, self._out_dir + '/' + unzipped_path.split("/"[-1]))
        os.rmdir(temp_path)


if __name__ == "__main__":

    dbg = DBGetter("temp")
    dbg.download("https://data.mendeley.com/api/datasets-v2/datasets/7dybx7wyfn/zip/download?version=3", "ecg")

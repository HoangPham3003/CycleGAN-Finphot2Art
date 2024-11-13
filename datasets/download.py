import os
import re
import zipfile
import subprocess

from tqdm.auto import tqdm


def load_data(ROOT_DIR=''):
    ROOT_DATA = os.path.join(ROOT_DIR, 'DATA')
    if not os.path.exists(ROOT_DATA):
        os.mkdir(ROOT_DATA)
    DATA_NAME = 'vangogh2photo'
    DATA_DIR = os.path.join(ROOT_DATA, DATA_NAME)


    if not os.path.exists(DATA_DIR):
        script_file = os.path.join(ROOT_DIR, 'datasets', 'download.sh')
        print(f"Executing download script: {script_file}")
        
        if not os.path.isfile(os.path.join(ROOT_DATA, f"{DATA_NAME}.zip")):
            print(f"\nDownloading zipfile dataset {DATA_NAME}\n")
            try:
                with subprocess.Popen(
                    ['bash', '-x', script_file, DATA_NAME],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                ) as proc:
                    pbar = tqdm(total=100, desc=f"Downloading", unit="%")

                    for line in proc.stderr:
                        match = re.search(r'(\d+)%', line)
                        if match:
                            percentage = int(match.group(1))
                            pbar.n = percentage
                            pbar.refresh()

                    proc.wait()
                    pbar.n = 100  # Set progress bar to complete at 100%
                    pbar.refresh()
                    pbar.close()
                
                print("Shell script executed successfully.")
                
            except subprocess.CalledProcessError as e:
                print("Error occurred while executing the shell script.")
                print(e.stderr.decode())
        
        # Unzip the downloaded file in Python
        zip_path = os.path.join(ROOT_DATA, f"{DATA_NAME}.zip")
        print(f"\nUnzipping {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(ROOT_DATA)
        print("Dataset is ready!")
        return DATA_DIR
    else:
        print("Dataset exists!")
        return DATA_DIR
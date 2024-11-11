import os
import re
import subprocess

from tqdm.auto import tqdm

if __name__ == '__main__':
    
    ROOT_DIR = os.getcwd()
    ROOT_DATA = os.path.join(ROOT_DIR, 'DATA')
    DATA_NAME = 'monet2photo'
    DATA_DIR = os.path.join(ROOT_DATA, DATA_NAME)
    
    
    chk = False
    if not os.path.exists(DATA_DIR):
        # Define the shell script path
        script_file = os.path.join(ROOT_DIR, 'datasets', 'download.sh')
        print(f"Executing download script: {script_file}")

        # Execute the shell script with subprocess
        try:
            with subprocess.Popen(
                ['bash', '-x', script_file, DATA_NAME],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            ) as proc:
                # Initialize tqdm progress bar
                pbar = tqdm(total=100, desc=f"Downloading {DATA_NAME} : ", unit="%")
                
                for line in proc.stderr:  # wget outputs progress to stderr
                    # Match percentage from wget output using regex
                    match = re.search(r'(\d+)%', line)
                    if match:
                        # Update progress bar
                        percentage = int(match.group(1))
                        pbar.n = percentage
                        pbar.refresh()

                # Close the progress bar when done
                pbar.close()
            
            print("Download and extraction complete for dataset: {}".format(DATA_NAME))
        except subprocess.CalledProcessError as e:
            print("Error occurred while executing the shell script.")
            print(e.stderr.decode())
    
        chk = True
    print(DATA_DIR)
    print(chk)
    
    
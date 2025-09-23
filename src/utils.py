import sys
import logging
import os
import urllib.request
# from pyunpack import Archive
from tqdm import tqdm
from datetime import datetime


# ----------------- Display --------------------------

def display_progress_bar_tqdm(progress, total_size, done=False, tqdm_bar=None):
    """Function responsible for displaying the progress bar using tqdm.
    
    If done is True, closes the tqdm progress bar.
    """
    if done:
        if tqdm_bar is not None:
            tqdm_bar.n = total_size
            tqdm_bar.last_print_n = total_size
            tqdm_bar.refresh()
            tqdm_bar.close()
        print(f"\n{total_size / (1024*1024):.2f} MB - Done!")
    else:
        if tqdm_bar is not None:
            tqdm_bar.n = progress
            tqdm_bar.refresh()


# ----------------- Download --------------------------

def download_file(url_base, url_suffix, output_path):
    print(f"Downloading the file: {os.path.basename(output_path)}")
    
    try:
        # Request the file size with a HEAD request
        req = urllib.request.Request(url_base + url_suffix, method='HEAD')
        f = urllib.request.urlopen(req)
        file_size = int(f.headers['Content-Length'])

        tqdm_bar = tqdm(total=file_size, unit='B', unit_scale=True, desc="Downloading")

        # Check if the file already exists and if not, download it
        if not os.path.exists(output_path):
            # Open the connection and the file in write-binary mode
            with urllib.request.urlopen(url_base + url_suffix) as response, open(output_path, 'wb') as out_file:
                block_size = 8192  # Define the block size for downloading in chunks
                progress = 0       # Initialize the progress counter
                
                # Download the file in chunks and write each chunk to the file
                while True:
                    chunk = response.read(block_size)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    progress += len(chunk)
                    display_progress_bar_tqdm(progress, file_size, tqdm_bar=tqdm_bar)  # Update the progress bar

            # After the download is complete, display final progress bar with "Download complete"
            display_progress_bar_tqdm(progress, file_size, done=True, tqdm_bar=tqdm_bar)

            # Verify if the downloaded file size matches the expected size
            downloaded_file_size = os.stat(output_path).st_size
        else:
            downloaded_file_size = os.stat(output_path).st_size
        
        # If the file size doesn't match, remove the file and try downloading again
        if file_size != downloaded_file_size:
            os.remove(output_path)
            print("File size incorrect. Downloading again.")
            download_file(url_base, url_suffix, output_path)
    
    except Exception as e:
        print("Error occurs when downloading file: " + str(e))
        print("Trying to download again")
        download_file(url_base, url_suffix, output_path)


# ----------------- Logger --------------------------
class LoggerWriter:
    def __init__(self, level, name="result"):
        self.level = level
        # Gera o nome do arquivo com base no parâmetro name
        output_file = generate_filename(name)
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Configura o logger para escrever no arquivo especificado e no console
        logging.basicConfig(
            level=logging.INFO,
            format='',
            handlers=[
                logging.FileHandler(output_file), 
                logging.StreamHandler(sys.stdout)
            ]
        )

    def write(self, message):
        if message != '\n':
            self.level(message)

    def flush(self):
        pass

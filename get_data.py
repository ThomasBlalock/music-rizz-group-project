# %%
url = 'https://huggingface.co/datasets/ailsntua/Chordonomicon/resolve/main/chordonomicon_v2.csv?download=true'
import requests
import os

# create data folder
if not os.path.exists("data"):
    os.makedirs("data")

def download_file_requests(url, filename):
    """
    Downloads a file from a URL using the requests library.

    Args:
        url (str): The URL of the file to download.
        filename (str): The local path and name to save the file as.
    """
    try:
        # Send a HTTP GET request to the URL
        response = requests.get(url, stream=True)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

        # Open the file in binary write mode ('wb')
        with open(filename, 'wb') as f:
            # Write the content in chunks to handle large files efficiently
            for chunk in response.iter_content(chunk_size=8192):
                if chunk: # Filter out keep-alive new chunks
                    f.write(chunk)
        print(f"Download completed successfully. File saved as: {os.path.abspath(filename)}")

    except requests.exceptions.RequestException as e:
        print(f"Download failed: {e}")

# Example usage:
save_as = "data/chordonomicon.csv"
download_file_requests(url, save_as)




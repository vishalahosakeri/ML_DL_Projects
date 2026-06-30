
import urllib.request

def downloadFile(url_dir,file_name):
    url = url_dir
    output_file = file_name
 
    # print("Downloading SMS Spam Collection dataset...")
    urllib.request.urlretrieve(url, output_file)
    print(f"Done! Saved as '{output_file}'")
    
    # Quick preview
    with open(output_file, "r") as f:
        lines = f.readlines()
    
    # print(type(lines))
    # print(f"Total messages: {len(lines)}")
    # print("\nFirst 5 rows:")
    for line in lines[:5]:
        label, message = line.strip().split("\t", 1)
        # print(f"  [{label}] {message[:60]}...")


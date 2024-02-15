import requests

def list_folders_in_repo(repo_name, token, path=""):
    base_url = f"https://api.github.com/repos/{repo_name}/contents/{path}"
    headers = {"Authorization": f"token {token}"}
    
    response = requests.get(base_url, headers=headers)
    if response.status_code == 200:
        contents = response.json()
        for item in contents:
            if item["type"] == "dir":
                print(item["path"])
                list_folders_in_repo(repo_name, token, item["path"])
    else:
        print(f"Failed to retrieve repository contents. Status code: {response.status_code}")

repo_name = "JoseCutileiro/MEIC"
token = "YOUR_TOKEN_HERE"

list_folders_in_repo(repo_name, token)
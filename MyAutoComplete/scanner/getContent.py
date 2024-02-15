import requests
import base64

def list_folders_for_user(username, token):
    base_url = f"https://api.github.com/users/{username}/repos"
    headers = {"Authorization": f"token {token}"}
    
    response = requests.get(base_url, headers=headers)
    if response.status_code == 200:
        repos = response.json()
        for repo in repos:
            repo_name = repo["full_name"]
            list_files_in_repo(repo_name, token)
    else:
        print(f"Failed to retrieve user repositories. Status code: {response.status_code}")

def list_files_in_repo(repo_name, token, path=""):
    base_url = f"https://api.github.com/repos/{repo_name}/contents/{path}"
    headers = {"Authorization": f"token {token}"}
    
    response = requests.get(base_url, headers=headers)
    if response.status_code == 200:
        contents = response.json()
        for item in contents:
            if item["type"] == "file":
                file_name = item["name"]
                if file_name.endswith(('.c', '.h')):
                    file_content = get_file_content(repo_name, token, item["path"])
                    if file_content:
                        append_to_code_file(file_name, file_content)
            elif item["type"] == "dir":
                list_files_in_repo(repo_name, token, item["path"])
    else:
        print(f"Failed to retrieve repository contents. Status code: {response.status_code}")

def get_file_content(repo_name, token, file_path):
    base_url = f"https://api.github.com/repos/{repo_name}/contents/{file_path}"
    headers = {"Authorization": f"token {token}"}
    
    response = requests.get(base_url, headers=headers)
    if response.status_code == 200:
        content = response.json()
        if "content" in content:
            return content["content"]
    else:
        print(f"Failed to retrieve file content. Status code: {response.status_code}")
        return None

def append_to_code_file(file_name, file_content):
    decoded_content = base64.b64decode(file_content).decode("latin-1")
    with open("code.txt", "a", encoding="latin-1") as f:
        print(f"Found file: {file_name}")
        f.write(decoded_content)

username = "JoseCutileiro"
token = "YOUR_TOKEN_HERE"

list_folders_for_user(username, token)
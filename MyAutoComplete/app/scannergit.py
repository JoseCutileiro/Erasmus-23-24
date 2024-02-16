import requests
import base64

def list_folders_for_user(username, token, extensions):
    base_url = f"https://api.github.com/users/{username}/repos"
    headers = {"Authorization": f"token {token}"}
    
    response = requests.get(base_url, headers=headers)
    if response.status_code == 200:
        repos = response.json()
        for repo in repos:
            repo_name = repo["full_name"]
            list_files_in_repo(repo_name, token, extensions, username)
    else:
        print(f"Failed to retrieve user repositories. Status code: {response.status_code}")

def list_files_in_repo(repo_name, token,extensions, username,path=""):
    base_url = f"https://api.github.com/repos/{repo_name}/contents/{path}"
    headers = {"Authorization": f"token {token}"}
    
    print("path: " + path)
    
    response = requests.get(base_url, headers=headers)
    if response.status_code == 200:
        contents = response.json()
        for item in contents:
            if item["type"] == "file":
                file_name = item["name"]
                if file_name.endswith(extensions):
                    file_content = get_file_content(repo_name, token, item["path"])
                    if file_content:
                        append_to_code_file(file_name, file_content, username)
            elif item["type"] == "dir":
                list_files_in_repo(repo_name, token, extensions,username,item["path"])
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

def append_to_code_file(file_name, file_content, username):
    decoded_content = base64.b64decode(file_content).decode("latin-1")
    with open("data/" + username +".txt", "a", encoding="latin-1") as f:
        print(f"Found file: {file_name}")
        f.write(decoded_content)
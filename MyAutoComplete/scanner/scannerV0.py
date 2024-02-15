import requests
import base64
import os

# GitHub username
username = "JoseCutileiro"
token = "YOUR_TOKEN_HERE"

# GitHub API endpoint for fetching repositories
repos_url = f"https://api.github.com/users/{username}/repos"

init_url = f"https://api.github.com/repos/{username}"

# Function to fetch repositories
def fetch_repositories():
    response = requests.get(repos_url, headers={"Authorization": f"token {token}"})
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to fetch repositories. Status Code: {response.status_code}")
        print(response.text)
        return []

# Function to fetch file contents from a repository, including files in subdirectories
def fetch_file_contents(repo_name, file_path):
    response = requests.get(f"{init_url}/{repo_name}/contents/{file_path}", headers={"Authorization": f"token {token}"})
    
    non_human_readable_extensions = ['.c','.h']
    
    _, file_extension = os.path.splitext(file_path)
    
    if response.status_code == 200:
        content = response.json()
        if isinstance(content, list):  # Check if content is a list, indicating a directory
            files = []
            for item in content:
                if item["type"] == "file":
                    if file_extension.lower() in non_human_readable_extensions:
                        file_contents = fetch_file_contents(repo_name, item["path"])
                        if file_contents:
                            files.extend(file_contents)
                elif item["type"] == "dir":
                    files.extend(fetch_file_contents(repo_name, f"{file_path}/{item['name']}"))
            return files
        else:  # Content is a file
            content = content.get("content")
            if content and file_extension.lower() in non_human_readable_extensions:
                print("Sucess: " + file_path)
                return [base64.b64decode(content).decode('latin-1')]
    return []

def extract_lines(contents):
    lines = []
    for content in contents:
        if content:
            lines.extend(content.splitlines())
    return lines

# Main function
def main():
    # Fetch repositories
    repositories = fetch_repositories()
    if not repositories:
        return

    # Extract lines from files in repositories
    lines = []
    for repo in repositories:
        if not repo["private"]:  # Only process public repositories
            repo_name = repo["name"]
            repo_url = repo["clone_url"]
            response = requests.get(f"{init_url}/{repo_name}/contents", headers={"Authorization": f"token {token}"})
            print(f"{init_url}/{repo_name}/contents")
            print(response)
            
            if response.status_code == 200:
                contents = response.json()
                for item in contents:
                    if item["type"] == "file":
                        file_contents = fetch_file_contents(repo_name, item["path"])
                        if file_contents:
                            lines.extend(extract_lines(file_contents))

    # Write lines to file
    with open("lines.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

# Entry point
if __name__ == "__main__":
    main()
    
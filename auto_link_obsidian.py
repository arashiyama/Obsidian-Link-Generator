import os
import re
from slugify import slugify

VAULT_PATH = os.getenv("OBSIDIAN_VAULT_PATH", "/Users/jonc/Obsidian/Jonathans Brain")
MD_EXTENSIONS = ('.md', '.markdown')

def get_note_titles(vault_path):
    titles = {}
    for root, dirs, files in os.walk(vault_path):
        for file in files:
            if file.endswith(MD_EXTENSIONS):
                title = os.path.splitext(file)[0]
                titles[title.lower()] = file
    return titles

def auto_link_notes(vault_path, titles):
    for root, dirs, files in os.walk(vault_path):
        for file in files:
            if file.endswith(MD_EXTENSIONS):
                path = os.path.join(root, file)
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Replace mentions of other note titles with links
                for title, linked_file in titles.items():
                    pattern = rf"\b({re.escape(title)})\b"
                    link = f"[[{linked_file[:-3]}]]"
                    content = re.sub(pattern, link, content, flags=re.IGNORECASE)

                with open(path, 'w', encoding='utf-8') as f:
                    f.write(content)

if __name__ == "__main__":
    titles = get_note_titles(VAULT_PATH)
    auto_link_notes(VAULT_PATH, titles)
    print("✅ Done linking notes!")

import os

# Define the folder and file structure
structure = {
    "raptor_rag": [
        "app.py",
        "requirements.txt",
        {
            "utils": [
                "__init__.py",
                "clustering.py",
                "extraction.py",
                "query.py",
                "processing.py"
                "summarization.py",
            ]
        },
        "tests"  # Empty folder
    ]
}

def create_structure(base_path, items):
    for item in items:
        if isinstance(item, str):
            # It's a file or folder
            item_path = os.path.join(base_path, item)
            if "." in item:  # crude check for file
                with open(item_path, "w") as f:
                    f.write(f"# {item}\n")
            else:
                os.makedirs(item_path, exist_ok=True)
        elif isinstance(item, dict):
            for folder, contents in item.items():
                folder_path = os.path.join(base_path, folder)
                os.makedirs(folder_path, exist_ok=True)
                create_structure(folder_path, contents)

# Run the script
if __name__ == "__main__":
    create_structure(".", structure["raptor_rag"])
    print("Project structure created successfully.")

import os
import shutil
import glob

def delete_item(path, is_dir=False):
    """Deletes the file or directory."""
    try:
        if is_dir:
            shutil.rmtree(path)
            print(f"Removed directory: {path}")
        else:
            os.remove(path)
            print(f"Removed file: {path}")
    except Exception as e:
        print(f"Error removing {path}: {e}")

def main():
    """Main function to find and remove unneeded files and directories."""
    print("Starting project cleanup...")
    
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    to_remove = {
        "dirs": [
            os.path.join(project_root, 'app'),
            os.path.join(project_root, '.pytest_cache'),
            os.path.join(project_root, 'build'),
            os.path.join(project_root, 'dist'),
        ],
        "files": [
            os.path.join(project_root, 'requirements.txt'),
        ],
        "patterns": [
            os.path.join(project_root, '**', '__pycache__'),
            os.path.join(project_root, '*.egg-info'),
        ]
    }

    items_to_delete = []

    # Find items to delete
    for dir_path in to_remove["dirs"]:
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            items_to_delete.append((dir_path, True))

    for file_path in to_remove["files"]:
        if os.path.exists(file_path) and os.path.isfile(file_path):
            items_to_delete.append((file_path, False))

    for pattern in to_remove["patterns"]:
        for path in glob.glob(pattern, recursive=True):
            if os.path.exists(path):
                is_dir = os.path.isdir(path)
                items_to_delete.append((path, is_dir))

    if not items_to_delete:
        print("No unneeded files or directories found.")
        return

    print("\nThe following items will be removed:")
    for path, is_dir in items_to_delete:
        print(f"  - {path}{'/' if is_dir else ''}")

    try:
        response = input("\nDo you want to remove all these items? (y/n): ").lower()
        if response == 'y':
            print("\nStarting removal...")
            for path, is_dir in items_to_delete:
                delete_item(path, is_dir)
        else:
            print("\nCleanup aborted by user.")
    except (KeyboardInterrupt, EOFError):
        print("\nCleanup aborted by user.")
    
    print("\nCleanup finished.")

if __name__ == "__main__":
    main()

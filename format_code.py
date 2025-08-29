#!/usr/bin/env python3
"""
Code formatting script for the Drone Image DL Classification project.

This script applies black formatting and isort import organization to all Python files
in the project to ensure consistent code style.
"""

import os
import platform
import subprocess
import sys
from pathlib import Path


def get_activate_cmd():
    """Return the correct activate command for the current OS."""
    current_os = platform.system().lower()
    if current_os == "linux":
        return ". .venv/bin/activate && "
    elif current_os == "darwin":  # macOS
        return ". .venv/bin/activate && "
    elif current_os == "windows":
        return ".venv\\Scripts\\activate && "
    else:
        return ""


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"Running {description}...")
    try:
        result = subprocess.run(
            command, shell=True, check=True, capture_output=True, text=True
        )
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False


def ensure_eof_newlines():
    """Ensure all text-based files end with a newline character."""
    print("Ensuring EOF newlines...")
    project_root = Path(__file__).parent

    # List of file patterns to check
    file_patterns = ["*.py", "*.md", "*.toml", "*.txt", "LICENSE", ".gitignore"]
    # Also check files in the app folder (including pages)
    app_patterns = ["app/*.py", "app/pages/*.py"]

    files_checked = set()
    all_patterns = file_patterns + app_patterns

    for pattern in all_patterns:
        for file in project_root.glob(pattern):
            # Avoid checking the same file twice if it matches multiple patterns
            if file in files_checked:
                continue
            files_checked.add(file)
            try:
                with open(file, "r") as f:
                    content = f.read()
                # Add newline if file doesn't end with one
                if content and not content.endswith("\n"):
                    with open(file, "w") as f:
                        f.write(content + "\n")
                    print(f"  ✅ Added EOF newline to {file.relative_to(project_root)}")
                else:
                    print(
                        f"  ✅ {file.relative_to(project_root)} already has EOF newline"
                    )
            except Exception as e:
                print(f"  ❌ Error processing {file.relative_to(project_root)}: {e}")
    print("✅ EOF newlines check completed")


def main():
    """Main function to format all Python files."""
    print("🎨 Starting code formatting...")

    # Get the project root directory
    project_root = Path(__file__).parent
    print(f"Project root: {project_root}")

    activate_cmd = get_activate_cmd()

    # Change to project directory
    original_cwd = Path.cwd()
    os.chdir(project_root)

    try:
        # Ensure EOF newlines first
        ensure_eof_newlines()

        # Run black formatting
        success = run_command(
            f"{activate_cmd}black *.py app/*.py app/pages/*.py",
            "Black code formatting",
        )
        if not success:
            sys.exit(1)

        # Run isort import organization
        success = run_command(
            f"{activate_cmd}isort *.py app/*.py app/pages/*.py",
            "isort import organization",
        )
        if not success:
            sys.exit(1)

        print("\n🎉 All formatting completed successfully!")
        print("Your code is now formatted according to PEP 8 standards.")

    finally:
        # Change back to original directory
        os.chdir(original_cwd)


if __name__ == "__main__":
    main()

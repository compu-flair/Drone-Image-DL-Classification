#!/usr/bin/env python3
"""
Code formatting script for the Drone Image DL Classification project.

This script applies black formatting and isort import organization to all Python files
in the project to ensure consistent code style.
"""

import subprocess
import sys
from pathlib import Path


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
    """Ensure all Python files end with a newline character."""
    print("Ensuring EOF newlines...")
    project_root = Path(__file__).parent

    for py_file in project_root.glob("*.py"):
        try:
            with open(py_file, "r") as f:
                content = f.read()

            # Add newline if file doesn't end with one
            if content and not content.endswith("\n"):
                with open(py_file, "w") as f:
                    f.write(content + "\n")
                print(f"  ✅ Added EOF newline to {py_file.name}")
            else:
                print(f"  ✅ {py_file.name} already has EOF newline")
        except Exception as e:
            print(f"  ❌ Error processing {py_file.name}: {e}")

    print("✅ EOF newlines check completed")


def main():
    """Main function to format all Python files."""
    print("🎨 Starting code formatting...")

    # Get the project root directory
    project_root = Path(__file__).parent
    print(f"Project root: {project_root}")

    # Change to project directory
    original_cwd = Path.cwd()
    os.chdir(project_root)

    try:
        # Ensure EOF newlines first
        ensure_eof_newlines()

        # Run black formatting
        success = run_command("black *.py", "Black code formatting")
        if not success:
            sys.exit(1)

        # Run isort import organization
        success = run_command("isort *.py", "isort import organization")
        if not success:
            sys.exit(1)

        print("\n🎉 All formatting completed successfully!")
        print("Your code is now formatted according to PEP 8 standards.")

    finally:
        # Change back to original directory
        os.chdir(original_cwd)


if __name__ == "__main__":
    import os

    main()

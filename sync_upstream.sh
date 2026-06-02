#!/bin/bash
# Script to sync with upstream repository and manage merge conflicts
# Usage: ./sync_upstream.sh

set -e

echo "=== Syncing with upstream repository ==="
echo ""

# Fetch latest changes from upstream
echo "1. Fetching latest changes from upstream..."
git fetch upstream

echo ""
echo "2. Current branch status:"
git status

echo ""
echo "3. Commits in upstream/master that are not in your master:"
git log --oneline master..upstream/master | head -10

echo ""
echo "4. Commits in your master that are not in upstream/master:"
git log --oneline upstream/master..master | head -10

echo ""
echo "=== Choose an action ==="
echo "a) Merge upstream/master into current branch (will create merge commit)"
echo "b) Rebase current branch onto upstream/master (will rewrite history)"
echo "c) Just show the diff (no changes)"
echo ""
read -p "Enter choice (a/b/c): " choice

case $choice in
    a)
        echo ""
        echo "Merging upstream/master into current branch..."
        git merge upstream/master
        echo ""
        echo "Merge completed! If there were conflicts, resolve them and run:"
        echo "  git add <resolved-files>"
        echo "  git commit"
        ;;
    b)
        echo ""
        echo "Rebasing current branch onto upstream/master..."
        echo "WARNING: This will rewrite your commit history!"
        read -p "Are you sure? (yes/no): " confirm
        if [ "$confirm" = "yes" ]; then
            git rebase upstream/master
            echo ""
            echo "Rebase completed! If there were conflicts, resolve them and run:"
            echo "  git add <resolved-files>"
            echo "  git rebase --continue"
        else
            echo "Rebase cancelled."
        fi
        ;;
    c)
        echo ""
        echo "Showing diff between master and upstream/master..."
        git diff master..upstream/master --stat
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac


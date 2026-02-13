#!/bin/bash

# 1. Add all changes
git add .

# 2. Commit with a timestamp msg if none provided
if [ -z "$1" ]; then
    msg="Update $(date)"
else
    msg="$1"
fi

git commit -m "$msg"

# 3. Push to GitHub
git push origin main

echo "--------------------------------"
echo "✅ Success! Code updated on GitHub."

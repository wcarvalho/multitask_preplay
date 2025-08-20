#!/bin/bash

# Check if a pattern was provided as an argument
PATTERN=$1

# List all fly apps and store them in an array
ALL_APPS=($(flyctl apps list --json | jq -r '.[].Name'))

if [ ${#ALL_APPS[@]} -eq 0 ]; then
    echo "No Fly.io applications found"
    exit 0
fi

# Filter apps if a pattern was provided
if [ -n "$PATTERN" ]; then
    # Create a new array with matching apps
    apps=()
    for app in "${ALL_APPS[@]}"; do
        if [[ "$app" =~ $PATTERN ]]; then
            apps+=("$app")
        fi
    done
    echo "Found ${#apps[@]} applications matching pattern '$PATTERN' out of ${#ALL_APPS[@]} total applications"
else
    # Use all apps if no pattern was provided
    apps=("${ALL_APPS[@]}")
    echo "Found ${#apps[@]} applications"
fi

if [ ${#apps[@]} -eq 0 ]; then
    echo "No applications match the given pattern"
    exit 0
fi

echo "Applications to be destroyed:"
printf '%s\n' "${apps[@]}"

# Ask for confirmation
read -p "Are you sure you want to destroy all these applications? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    for app in "${apps[@]}"
    do
        echo "Destroying $app..."
        flyctl apps destroy "$app" --yes
    done
    echo "All applications have been destroyed"
else
    echo "Operation cancelled"
fi 
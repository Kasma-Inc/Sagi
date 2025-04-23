#!/bin/bash
# Check if a file is specified as an argument
if [ $# -eq 1 ]; then
    # Format the specified file
    FILE_TO_FORMAT="$1"
    echo "Formatting single file: $FILE_TO_FORMAT"
    
    # Only proceed if the file exists
    if [ -f "$FILE_TO_FORMAT" ]; then
        black "$FILE_TO_FORMAT"
        blackdoc "$FILE_TO_FORMAT"
        isort "$FILE_TO_FORMAT" --profile black
        autoflake --in-place --remove-all-unused-imports "$FILE_TO_FORMAT"
        exit 0
    else
        echo "Error: File $FILE_TO_FORMAT does not exist"
        exit 1
    fi
fi

# If no file is specified, format all Python files tracked by git
echo "Formatting all Python files tracked by git"
FILES_TO_FORMAT=$(git ls-files '*.py')

black $FILES_TO_FORMAT
blackdoc $FILES_TO_FORMAT
isort $FILES_TO_FORMAT --profile black
autoflake --in-place --remove-all-unused-imports $FILES_TO_FORMAT

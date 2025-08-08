#!/bin/bash

dirProj="~/Shared/Sync/Private/Work/Projects/apr25_bds_rakuten_2"

filePrefix='output_'
filePrefix=''

# Define the folder paths
# folder_selected='image_train'
folder_selected='image_test'
folder_a="$dirProj/data/images/$folder_selected"
folder_b="$dirProj/data/images_bgrm/$folder_selected"
folder_a_done="$dirProj/data/images/${folder_selected}_done"

# Check if both folders exist
if [ ! -d "$folder_a" ] || [ ! -d "$folder_b" ]; then
  echo "Error: One or both of the specified folders do not exist."
  exit 1
fi

echo "Looping.."
# Find files with the same name in both folders
while IFS= read -r -d $'\0' file_a; do
  filename=$(basename "$file_a")
#   echo $filename
  filenameWOext=${filename//.jpg/}
  echo $filenameWOext
#   exit
  if [ -f "$folder_b/$filePrefix$filenameWOext.png" ]; then
    echo "===> Removing '$filename' from '$folder_a'"
    echo "mv \"$folder_a/$filename\" \"$folder_a_done/\""
    mv "$folder_a/$filename" "$folder_a_done/"
    # rm -f "$file_a"
  fi
done < <(find "$folder_a" -type f -print0)

echo "Comparison and removal process complete."

exit 0
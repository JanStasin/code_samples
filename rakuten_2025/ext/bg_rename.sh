#!/bin/bash

dirProj="~/Shared/Sync/Private/Work/Projects/apr25_bds_rakuten_2"

# Define the folder paths
folder_0="$dirProj/data/images_bgrm"
folder_1="$folder_0/image_test"
folder_2="$folder_0/image_train"

# Check if both folders exist
if [ ! -d "$folder_1" ] || [ ! -d "$folder_2" ]; then
  echo "Error: One or both of the specified folders do not exist."
  exit 1
fi

# folder_selected=$folder_1
folder_selected=$folder_2
echo "Looping.. $folder_selected"
for i in $folder_selected/output_*.jpg; do 
  newName=${i/output_/}
  newName=${newName/.jpg/.png}
  echo "mv \"$i\" \"$newName\"" 
  mv "$i" "$newName"
done

echo "Rename process complete."

echo "tar czvf $folder_0.tar.gz $folder_0"

exit 0
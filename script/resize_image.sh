#!/bin/bash
if [[ ! $1 ]] || [[ ! $2 ]]; then
    echo "Usage: $0 /path/to/image/directory percent"
    exit 0
fi

for f in $(ls "$1" | egrep '\.jpg$|\.png$|\.JPG$|\.JPEG$|\.PNG$')
do
	path="$1/$f"
    H=$(sips -g pixelHeight "$path" | grep 'pixelHeight' | cut -d: -f2)

    H=$(($H * $2 / 100))

	echo "Resize image $f to size with height $H"

	sips --resampleHeight "$H" "$path" >/dev/null
done
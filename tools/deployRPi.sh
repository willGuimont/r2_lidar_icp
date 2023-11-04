#!/usr/bin/env bash
folder_name=$(basename "$(pwd)")
rsync -avr -e ssh --exclude-from=.rsync_exclude ../"$folder_name" ubuntu@10.42.0.1:~/"$folder_name"

#!/usr/bin/env bash
rsync -avr -e ssh --exclude-from=.rsync_exclude ../diff_icp/ ubuntu@10.42.0.1:~/diff_icp

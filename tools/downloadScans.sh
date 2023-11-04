#!/usr/bin/env bash
rsync -avr -e ssh ubuntu@10.42.0.1:~/diff_icp/live_scans/* ./live_scans/pi

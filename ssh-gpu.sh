#!/usr/bin/env bash
# SSH to A100 pod at 206.1.35.5:20038
ssh -o StrictHostKeyChecking=no -o BatchMode=yes -i ~/.ssh/id_ad12340 root@206.1.35.5 -p 20038 "$@"
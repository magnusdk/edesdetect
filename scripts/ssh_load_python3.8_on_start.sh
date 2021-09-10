#!/bin/bash
module load Python/3.8.6-GCCcore-10.2.0

if [ -n "$SSH_ORIGINAL_COMMAND" ]
then
    $SSH_ORIGINAL_COMMAND
else
    bash
fi


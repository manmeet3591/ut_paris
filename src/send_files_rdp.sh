#!/bin/bash

# FTP server details
FTP_SERVER="ftp.umr-cnrm.fr"
FTP_USERNAME="depot_rdp2024"
FTP_PASSWORD="!FrapyTFA3135"
FTP_DIRECTORY="/MODELLING/REALTIME/UT-MeteoGAN"

# List of files to upload
FILES=$(ls UT_METEOGAN_TEXUS_*.nc)

# Connect to FTP and upload each file
ftp -inv $FTP_SERVER <<EOF
user $FTP_USERNAME $FTP_PASSWORD
cd $FTP_DIRECTORY
binary
$(for file in $FILES; do echo "put $file"; done)
bye
EOF

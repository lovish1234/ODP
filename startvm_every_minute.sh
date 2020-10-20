#!/bin/bash

while :;
  do
  clear
  date
  gcloud compute instances start <Your Machine Name>  --zone <Your Zone>
  sleep 60
done

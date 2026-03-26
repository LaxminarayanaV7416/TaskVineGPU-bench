
Bash utlity: we can use command ```compgen -c condor``` to see all commands available as executable on linux that starts with condor
Fish utlity: we can use command ```complete -C"vine"``` to see all commands available as executable on ArchLinux that starts with vine.

```{bash}
vine_factory -T local --manager-name "helloworld"
condor_status -long qa-a10-031.crc.nd.edu | grep GPU
nvidia-smi
echo $CUDA_VISIBLE_DEVICES
condor_q
vine_factory -T condor --min-workers=1 --max-workers=1 --gpus=2 crcfe01.crc.nd.edu 9000 &
condor_ssh_to_job 1367
condor_q --better-analyze
vine_factory -T condor --min-workers=1 --max-workers=1 --gpus=2 --cores=1 --memory=500 --disk=500 crcfe01.crc.nd.edu 9000
condor_ssh_to_job 1368

condor_gpu_discovery

# to see all the node names that have GPU's installed
condor_status -constraint 'GPUs > 0' -af Name GPUs


```

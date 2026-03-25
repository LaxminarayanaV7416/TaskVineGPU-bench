
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

```
## Produce dataset
Please edit the `dataset_directory` in `runCondor.py`. This should be where you download the CaloChallenge files and also store the point-cloud dataset you produce.
To produce the dataset, run (`--test` means you don't submit to condor but only produce shell file. `--store_geometric` means you store in torch_geometric object, otherwise store torch.tensor)
```
python runCondor.py --dataset [1/2/3] [--store_geometric] [--test] --coordinate [polar/euclidian] [--zero_pedding]
```
After all the jobs are finished, run to merge them to one files (This step can be ignored since the files may be too large)
```
python runCondor.py --dataset[1/2/3] [--store_geometric] --coordinate [polar/euclidian] [--zero_pedding] --merge
```

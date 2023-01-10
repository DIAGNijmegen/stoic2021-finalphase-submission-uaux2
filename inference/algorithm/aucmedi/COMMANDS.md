# how to run AUCMEDI training


```
nohup sh -c "python algorithm/aucmedi/training.py --fold 0 --data /storage/stoic2021-training.v2/data/lungseg/ -g 0" &> logs.training.kfold_0.txt &
nohup sh -c "python algorithm/aucmedi/training.py --fold 1 --data /storage/stoic2021-training.v2/data/lungseg/ -g 1" &> logs.training.kfold_1.txt &
nohup sh -c "python algorithm/aucmedi/training.py --fold 2 --data /storage/stoic2021-training.v2/data/lungseg/ -g 2" &> logs.training.kfold_2.txt &
nohup sh -c "python algorithm/aucmedi/training.py --fold 3 --data /storage/stoic2021-training.v2/data/lungseg/ -g 3" &> logs.training.kfold_3.txt &


nohup sh -c "python algorithm/aucmedi/training.py --fold 4 --data /storage/stoic2021-training.v2/data/lungseg/ -g 1" &> logs.training.kfold_4.txt &


nohup sh -c "python algorithm/aucmedi/lr_pred.py --data /share/stoic2021-training.v2/data/lungseg/ -g 0" &> logs.lr_pred.txt &
nohup sh -c "python algorithm/aucmedi/lr_pred.py --data /storage/stoic2021-training.v2/data/lungseg/ -g 1" &> logs.lr_pred.txt &

```


```
# Severity - MISIT-Workstation
nohup sh -c "python algorithm/aucmedi/training.severity.py --fold 0 --data /share/stoic2021-training.v2/data/lungseg/ -g 2" &> logs.training.severity.kfold_0.txt &
nohup sh -c "python algorithm/aucmedi/training.severity.py --fold 1 --data /share/stoic2021-training.v2/data/lungseg/ -g 3" &> logs.training.severity.kfold_1.txt &
# Severity - DGX-Workstation
nohup sh -c "python algorithm/aucmedi/training.severity.py --fold 2 --data /storage/stoic2021-training.v2/data/lungseg/ -g 0" &> logs.training.severity.kfold_2.txt &
nohup sh -c "python algorithm/aucmedi/training.severity.py --fold 3 --data /storage/stoic2021-training.v2/data/lungseg/ -g 2" &> logs.training.severity.kfold_3.txt &
nohup sh -c "python algorithm/aucmedi/training.severity.py --fold 4 --data /storage/stoic2021-training.v2/data/lungseg/ -g 3" &> logs.training.severity.kfold_4.txt &

# Severity - Preds
nohup sh -c "python algorithm/aucmedi/lr_pred.severity.py --data /share/stoic2021-training.v2/data/lungseg/ -g 0" &> logs.lr_pred.severity.txt &
```

```
# Severity DenseNet on original Images - MISIT-Workstation
nohup sh -c "python algorithm/aucmedi/training.severity.densenet.py --fold 0 --data /share/stoic2021-training.v2/data/mha/ -g 0" &> logs.training.severity.densenet.kfold_0.txt &
nohup sh -c "python algorithm/aucmedi/training.severity.densenet.py --fold 1 --data /share/stoic2021-training.v2/data/mha/ -g 2" &> logs.training.severity.densenet.kfold_1.txt &
nohup sh -c "python algorithm/aucmedi/training.severity.densenet.py --fold 2 --data /share/stoic2021-training.v2/data/mha/ -g 3" &> logs.training.severity.densenet.kfold_2.txt &

nohup sh -c "python algorithm/aucmedi/training.severity.densenet.py --fold 3 --data /share/stoic2021-training.v2/data/mha/ -g 0" &> logs.training.severity.densenet.kfold_3.txt &
nohup sh -c "python algorithm/aucmedi/training.severity.densenet.py --fold 4 --data /share/stoic2021-training.v2/data/mha/ -g 2" &> logs.training.severity.densenet.kfold_4.txt &
# Severity - Preds
nohup sh -c "python algorithm/aucmedi/lr_pred.severity.densenet.py --data /share/stoic2021-training.v2/data/mha/ -g 0" &> logs.lr_pred.severity.densenet.txt &

```


```
# Severity Meta - MISIT-Workstation
nohup sh -c "python algorithm/aucmedi/training.severity.py --fold 0 --data /share/stoic2021-training.v2/data/lungseg/ -g 0" &> logs.training.severity.metadata.kfold_0.txt &
# Severity Meta - DGX-Workstation
nohup sh -c "python algorithm/aucmedi/training.severity.py --fold 1 --data /storage/stoic2021-training.v2/data/lungseg/ -g 0" &> logs.training.severity.metadata.kfold_1.txt &
nohup sh -c "python algorithm/aucmedi/training.severity.py --fold 2 --data /storage/stoic2021-training.v2/data/lungseg/ -g 1" &> logs.training.severity.metadata.kfold_2.txt &
nohup sh -c "python algorithm/aucmedi/training.severity.py --fold 3 --data /storage/stoic2021-training.v2/data/lungseg/ -g 2" &> logs.training.severity.metadata.kfold_3.txt &
nohup sh -c "python algorithm/aucmedi/training.severity.py --fold 4 --data /storage/stoic2021-training.v2/data/lungseg/ -g 3" &> logs.training.severity.metadata.kfold_4.txt &

```

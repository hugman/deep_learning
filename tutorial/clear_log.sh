echo Delete log files
rm /local/log-tensorboard/*
echo Terminiate Tensorboard
kill -9 $(ps opid= -C tensorboard)
echo Restart Tensorboard
nohup tensorboard --logdir /local/log-tensorboard &

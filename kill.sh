
ps -ef | grep $1 | awk '{print $2}' | xargs kill -9
rm -rf ./train_log/$1

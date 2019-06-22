# Basically just set paths for now.

if [ -z ${INTENTPRED_PATH+x} ]; then 
    echo "Setting INTENTPRED_PATH to `pwd`"; 
    echo "" >> ${HOME}/.bashrc
    echo "### Intention Prediction ###" >> ${HOME}/.bashrc
    echo "export INTENTPRED_PATH=`pwd`" >> ${HOME}/.bashrc
else 
    echo "INTENTPRED_PATH is already set to '$INTENTPRED_PATH'"; 
fi

wget https://github.com/djp42/IntentionPrediction/releases/download/v0.1/data.tar.gz -P res
tar -xzvf res/data.tar.gz -C res
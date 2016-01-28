# lstm-parser
Transition based dependency parser with state embeddings computed by LSTM RNNs

# Checking out the project for the first time

The first time you clone the repository, you need to sync the `cnn/` submodule.

    git submodule init
    git submodule update

# Build instructions

    mkdir build
    cd build
    cmake -DEIGEN3_INCLUDE_DIR=/opt/tools/eigen-dev ..
    (in allegro: cmake -DEIGEN3_INCLUDE_DIR=/opt/tools/eigen-dev -G 'Unix Makefiles')
    make -j2

# Update cnn instructions
To sync the most recent version of `cnn`, you need to issue the following command:
 
    git submodule foreach git pull origin master
    
    
# Command to run the parser (in allegro): 

    parser/lstm-parse -h
    
    parser/lstm-parse -T /usr0/home/cdyer/projects/lstm-parser/train.txt -d /usr0/home/cdyer/projects/lstm-parser/dev.txt --hidden_dim 100 --lstm_input_dim 100 -w /usr3/home/lingwang/chris/sskip.100.vectors --pretrained_dim 100 --rel_dim 20 --action_dim 20 -t -P
    
# How to get the arc-std oracles (traning set and dev set of the parser) having a CoNLL 2006 file:
   
    java -jar ParserOracleArcStd.jar -t -1 -l 1 -c train10.conll -i train10.conll > oracleTrainArcStd.txt
    (oracle code is in: /usr2/home/miguel/ParserOracle)
    (the train10.conll file should be fully projective)
    



# Dependencies
* boost-1.60.0
* eigen `hg clone https://bitbucket.org/eigen/eigen`

# How to use?
```
# setup repository #
cd
mkdir git ; cd git/
git clone git@github.com:clab/language-universal-parser.git
cd language-universal-parser
git submodule init
git submodule update
cd dynet
git pull origin master
cd ../

# build the parser (with latest version of dynet) #
cd ~/git/language-universal-parser/dynet
git pull origin master
cd .. ; mkdir build-gpu ; cd build-gpu
cmake -DEIGEN3_INCLUDE_DIR=$EIGEN_ROOT ..  # -DBACKEND=cuda is not supported just yet
make -j 10

# train the parser on small data #
~/git/language-universal-parser/build-gpu/parser/lstm-parse --train -P --training_data $TRAIN_ARCSTD --dev_data $DEV_ARCSTD --pretrained_dim 50 --pretrained $PRETRAINED_EMBEDDINGS --brown_clusters $PRETRAINED_CLUSTERS --epochs 1
```
# How to generate arc-standard transitions?
The parser expects projective treebanks with arc-standard transitions as input (see command lines below). To convert nonprojective treebanks in CoNLL 2006 format to the arc-std oracle files of the pseudo-projective treebanks:
```    
java -jar maltparser-1.8.1.jar -c pproj -m proj -i $split_lc -o $split_projective -pp baseline
java -jar ParserOracleArcStd.jar -t -1 -l 1 -c treebank.conll -i treebank.conll > treebank.arcstd
```

We recommend that you lowercase word tokens/types in all input files (e.g., pretrained embeddings, Brown clusters, train/dev/test treebanks) before calling the parser.

# Language typology embeddings
To enable language typology embeddings, use the following command line argument `--typological_properties typology_file`. Sample typology files have been provided in the subdirectory `typological_properties/`. If you enable typology embeddings, please prefix each word in the input files (e.g., `en:book` instead of `book`). The two-letter prefix should match the first field in the typology file.

# What to cite?
[Many Languages, One Parser](http://arxiv.org/abs/1602.01595) TACL 2016 (to appear)
Waleed Ammar, George Mulcaire, Miguel Ballesteros, Chris Dyer, Noah A. Smith

[results](https://github.com/clab/language-universal-parser/tree/084eed3b1510fc893c4c92474cdcea1d7c58aa7c)

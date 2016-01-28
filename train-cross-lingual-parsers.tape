#!/usr/bin/env ducttape

global {
  ducttape_experimental_submitters=enable
  ducttape_experimental_imports=enable

  # experiments directory
  ducttape_output="/usr4/home/wammar/exp-universal-embeddings"

  # language to evaluate on
  target_language=(Target: cs="cs" de="de" en="en" es="es" fi="fi" fr="fr" ga="ga" hu="hu" it="it" sv="sv")

  ###########
  # CONFIGS #
  ###########
  use_pretrained="yes" #(UsePretrained: yes="yes" no="")
  pretrained_dim=(PretrainedDim: hundred="100" fifty="50")
  use_typology=(UseTypology: no="" yes="yes")
  pos_dim="12" #(PosDim: default="12" hundred="100" twohundreds="200")
  hidden_dim="100" #(HiddenDim: hundred="100" default="64")
  lstm_input_dim="100" #(LstmInputDim: hundred="100" default="60")
  rel_dim="20" #(RelDim: twenty="20" default="10")
  action_dim="20" #(ActionDim: twenty="20" default="16")
  unk_prob="0.2" #(UnkProb: default="0.2")
  layers="2" #(Layers: default="2")
  input_dim="0" #(InputDim: zero="0" hundred="100")
  use_spelling=(UseSpelling: no="" yes="yes")

  ##############
  # DATA FILES #
  ##############

  # tokenized and lowercased monolingual corpora
  corpus_cs="/usr3/home/wammar/corpora/monolingual/plain-cs/ud-v1x40+news.2008-2012+nc.v8.tok.lc.1m"
  corpus_de="/usr3/home/wammar/corpora/monolingual/plain-de/ud-v1x40+news.2008-2012+nc.v8+ud.v1.lc.tok.1m"
  corpus_en="/usr3/home/wammar/corpora/monolingual/plain-en/ud-v1x40+news.2008-2012+nc.v8+ud.v1.lc.tok.1m"
  corpus_es="/usr3/home/wammar/corpora/monolingual/plain-es/ud-v1x40+news.2008-2012+nc.v8+ud.v1.lc.tok.1m"
  corpus_fi="/usr3/home/wammar/corpora/monolingual/plain-fi/ud-v1x40+wmt15-mono+ud.v1.tok.lc.1m"
  corpus_fr="/usr3/home/wammar/corpora/monolingual/plain-fr/ud-v1x40+news.2008-2012+nc.v8+ud.v1.lc.tok.1m"
  corpus_ga="/usr3/home/wammar/corpora/monolingual/plain-ga/ud-v1x40+DGT+EUbookshop+EUconst+GNOME+KDE4+Tatoeba+Ubuntu.tok.lc.1m"
  corpus_hu="/usr3/home/wammar/corpora/monolingual/hungarian/ud-v1x40+newscrawl_2011_100K+europarl.tok.lc.1m"
  corpus_it="/usr3/home/wammar/corpora/monolingual/italian/ud-v1x40+news_2010_100K+europarl.tok.lc.1m"
  corpus_sv="/usr3/home/wammar/corpora/monolingual/plain-sv/ud-v1x40+europarl.sv.lc.1m"

  # wiktionary dump
  wiktionary_dump="/usr3/home/wammar/corpora/parallel/wiktionary-bilingual-dicts/wiktionary_de+en_2012-04-01_translations.csv.gz"

  # the alignments used by Jiang in his ACL 2015 paper, available for download at 
  # https://github.com/jiangfeng1124/acl15-clnndep/tree/master/resources/align
  jiangacl15_alignments_de_en="/usr3/home/wammar/corpora/parallel/acl15-clnndep-alignments/de-en.align"
  jiangacl15_alignments_es_en="/usr3/home/wammar/corpora/parallel/acl15-clnndep-alignments/es-en.align"
  jiangacl15_alignments_fr_en="/usr3/home/wammar/corpora/parallel/acl15-clnndep-alignments/fr-en.align"
  
  # parallel data to be word-aligned then used to extract bilingual dictionaries.
  bitext_cs_en="/usr3/home/wammar/corpora/parallel/czech/100k+aer+kazuya+nc_v8.tok.lc.cs-en"
  bitext_de_en="/usr3/home/wammar/corpora/parallel/german/wmt14-syntaxcorpus.tok.lc.de-en"
  bitext_es_en="/usr3/home/wammar/corpora/parallel/spanish/nc10+europarl5+kazuya.tok.lc.es-en"
  bitext_fi_en="/usr3/home/wammar/corpora/parallel/finnish/wmt-train.tok.lc.fi-en"
  bitext_fr_en="/usr3/home/wammar/corpora/parallel/french/kazuya+10k+nc10.tok.lc.fr-en"
  bitext_ga_en="/usr3/home/wammar/corpora/parallel/irish/DGT+EUbookshop+EUconst+GNOME+KDE4+Tatoeba+Ubuntu.tok.lc.ga-en"
  bitext_hu_en="/usr3/home/wammar/corpora/parallel/hungarian/kazuya.tok.lc.hu-en"
  bitext_it_en="/usr3/home/wammar/corpora/parallel/italian/kazuya.tok.lc.it-en"
  bitext_sv_en="/usr3/home/wammar/corpora/parallel/swedish/kazuya.tok.lc.sv-en"

  # wikipedia dump files needed to extract parallel wikipedia titles as detailed in
  # https://github.com/clab/wikipedia-parallel-titles
  wikipedia_base_cs="/usr3/home/wammar/corpora/parallel/wikipedia-parallel-titles/cswiki-20150602"
  wikipedia_base_de="/usr3/home/wammar/corpora/parallel/wikipedia-parallel-titles/dewiki-20150602"
  wikipedia_base_es="/usr3/home/wammar/corpora/parallel/wikipedia-parallel-titles/eswiki-20150602"
  wikipedia_base_fi="/usr3/home/wammar/corpora/parallel/wikipedia-parallel-titles/fiwiki-20150602"
  wikipedia_base_fr="/usr3/home/wammar/corpora/parallel/wikipedia-parallel-titles/frwiki-20150602"
  wikipedia_base_ga="/usr3/home/wammar/corpora/parallel/wikipedia-parallel-titles/gawiki-20150603"
  wikipedia_base_hu="/usr3/home/wammar/corpora/parallel/wikipedia-parallel-titles/huwiki-20150602"
  wikipedia_base_it="/usr3/home/wammar/corpora/parallel/wikipedia-parallel-titles/itwiki-20150602"
  wikipedia_base_sv="/usr3/home/wammar/corpora/parallel/wikipedia-parallel-titles/svwiki-20150602"

  # multilingual punctuation clusters
  # TODO: this file only has de+en+es+fr punctuations. Please add the language specific 
  # version of these puncutations in other languages.
  punctuation_clusters="/usr3/home/wammar/corpora/parallel/wiktionary-bilingual-dicts/punctuation-mappings"
  
  # typological properties
  typological_properties="/usr0/home/wammar/git/internal-lstm-parser/typological_properties.txt"

  # universal dependency treebanks (train)
  treebank_train_cs="/usr3/home/wammar/corpora/treebanks/universal-dependencies-1.1/cs/cs-ud-train.conllu"
  treebank_train_de="/usr3/home/wammar/corpora/treebanks/universal-dependencies-1.1/de/de-ud-train.conllu"
  treebank_train_en="/usr3/home/wammar/corpora/treebanks/universal-dependencies-1.1/en/en-ud-train.conllu"
  treebank_train_es="/usr3/home/wammar/corpora/treebanks/universal-dependencies-1.1/es/es-ud-train.conllu"
  treebank_train_fi="/usr3/home/wammar/corpora/treebanks/universal-dependencies-1.1/fi/fi-ud-train.conllu"
  treebank_train_fr="/usr3/home/wammar/corpora/treebanks/universal-dependencies-1.1/fr/fr-ud-train.conllu"
  treebank_train_ga="/usr3/home/wammar/corpora/treebanks/universal-dependencies-1.1/ga/ga-ud-train.conllu"
  treebank_train_hu="/usr3/home/wammar/corpora/treebanks/universal-dependencies-1.1/hu/hu-ud-train.conllu"
  treebank_train_it="/usr3/home/wammar/corpora/treebanks/universal-dependencies-1.1/it/it-ud-train.conllu"
  treebank_train_sv="/usr3/home/wammar/corpora/treebanks/universal-dependencies-1.1/sv/sv-ud-train.conllu"

  # universal dependency treebanks (dev)
  treebank_dev_cs="/usr3/home/wammar/corpora/treebanks/universal-dependencies-1.1/cs/cs-ud-dev.conllu" 
  treebank_dev_de="/usr3/home/wammar/corpora/treebanks/universal-dependencies-1.1/de/de-ud-dev.conllu"
  treebank_dev_en="/usr3/home/wammar/corpora/treebanks/universal-dependencies-1.1/en/en-ud-dev.conllu"
  treebank_dev_es="/usr3/home/wammar/corpora/treebanks/universal-dependencies-1.1/es/es-ud-dev.conllu"
  treebank_dev_fi="/usr3/home/wammar/corpora/treebanks/universal-dependencies-1.1/fi/fi-ud-dev.conllu"
  treebank_dev_fr="/usr3/home/wammar/corpora/treebanks/universal-dependencies-1.1/fr/fr-ud-dev.conllu"
  treebank_dev_ga="/usr3/home/wammar/corpora/treebanks/universal-dependencies-1.1/ga/ga-ud-dev.conllu"
  treebank_dev_hu="/usr3/home/wammar/corpora/treebanks/universal-dependencies-1.1/hu/hu-ud-dev.conllu"
  treebank_dev_it="/usr3/home/wammar/corpora/treebanks/universal-dependencies-1.1/it/it-ud-dev.conllu"
  treebank_dev_sv="/usr3/home/wammar/corpora/treebanks/universal-dependencies-1.1/sv/sv-ud-dev.conllu"
  
  # universal dependency treebanks (test)
  treebank_test_cs="/usr3/home/wammar/corpora/treebanks/universal-dependencies-1.1/cs/cs-ud-test.conllu" 
  treebank_test_de="/usr3/home/wammar/corpora/treebanks/universal-dependencies-1.1/de/de-ud-test.conllu"
  treebank_test_en="/usr3/home/wammar/corpora/treebanks/universal-dependencies-1.1/en/en-ud-test.conllu"
  treebank_test_es="/usr3/home/wammar/corpora/treebanks/universal-dependencies-1.1/es/es-ud-test.conllu"
  treebank_test_fi="/usr3/home/wammar/corpora/treebanks/universal-dependencies-1.1/fi/fi-ud-test.conllu"
  treebank_test_fr="/usr3/home/wammar/corpora/treebanks/universal-dependencies-1.1/fr/fr-ud-test.conllu"
  treebank_test_ga="/usr3/home/wammar/corpora/treebanks/universal-dependencies-1.1/ga/ga-ud-test.conllu"
  treebank_test_hu="/usr3/home/wammar/corpora/treebanks/universal-dependencies-1.1/hu/hu-ud-test.conllu"
  treebank_test_it="/usr3/home/wammar/corpora/treebanks/universal-dependencies-1.1/it/it-ud-test.conllu"
  treebank_test_sv="/usr3/home/wammar/corpora/treebanks/universal-dependencies-1.1/sv/sv-ud-test.conllu"

  # the CCA word embeddings used by Jiang in his ACL15 paper, available at 
  # https://drive.google.com/file/d/0B1z04ix6jD_Db3REdHlnREpjMmc/view?usp=sharing
  jiangacl15_cca_ende_en="/usr3/home/wammar/corpora/monolingual/universal/acl15-cl-wemb/cca/en-de/en.50.w2v"
  jiangacl15_cca_ende_de="/usr3/home/wammar/corpora/monolingual/universal/acl15-cl-wemb/cca/en-de/de.50.w2v"
  jiangacl15_cca_enfr_en="/usr3/home/wammar/corpora/monolingual/universal/acl15-cl-wemb/cca/en-fr/en.50.w2v"
  jiangacl15_cca_enfr_fr="/usr3/home/wammar/corpora/monolingual/universal/acl15-cl-wemb/cca/en-fr/fr.50.w2v"
  jiangacl15_cca_enes_en="/usr3/home/wammar/corpora/monolingual/universal/acl15-cl-wemb/cca/en-es/en.50.w2v"
  jiangacl15_cca_enes_es="/usr3/home/wammar/corpora/monolingual/universal/acl15-cl-wemb/cca/en-es/es.50.w2v"

  # the PROJ word embeddings used by Jiang in his ACL15 paper, available at
  # https://drive.google.com/file/d/0B1z04ix6jD_Db3REdHlnREpjMmc/view?usp=sharing
  jiangacl15_proj_en="/usr3/home/wammar/corpora/monolingual/universal/acl15-cl-wemb/projected/en.50"
  jiangacl15_proj_de="/usr3/home/wammar/corpora/monolingual/universal/acl15-cl-wemb/projected/de.50"
  jiangacl15_proj_fr="/usr3/home/wammar/corpora/monolingual/universal/acl15-cl-wemb/projected/fr.50"
  jiangacl15_proj_es="/usr3/home/wammar/corpora/monolingual/universal/acl15-cl-wemb/projected/es.50"

  #########
  # TOOLS #
  #########

  lstm_parser="/usr0/home/wammar/git/internal-lstm-parser/"
  incremental_word2vec="/usr0/home/wammar/git/incremental-word2vec/"
  conll_eval="/usr0/home/wammar/wammar-utils/conllx-eval.v1_8.pl"
  wammar_utils="/usr0/home/wammar/wammar-utils/"
  parse_oracle_arc_std_swap_jar="/usr3/home/wammar/corpora/treebanks/universal-dependencies-1.0/treebanks/ParserOracleArcStdWithSwap.jar"
  #parse_oracle_arc_std_swap_jar="/usr3/home/wammar/corpora/treebanks/universal-dependencies-1.0/treebanks/ParseOracleArcStd.jar"
  wikipedia_parallel_titles="/usr0/home/wammar/git/wikipedia-parallel-titles/"
  fast_align="use_giza" #"/usr0/home/wammar/git/fast_align/"
  cdec="/usr0/home/wammar/cdec/"
}

# TODO: controlled experiments between delex and lex
# TODO: controlled experiments to decide which subset of typological properties are best
plan Full {
  #reach TrainEmbeddingsOfWordClusters
  reach ProcessIndividualTreebanks
  reach AggregateTrainTreebanks
  reach AggregateDevTreebanks
}

task EvaluateParserOnTest
   :: conll_eval=@
   :: use_typology=@
   < train=(TrainLang:    cs=$treebank_train_cs_arcstdswap@ProcessIndividualTreebanks
                  			  de=$treebank_train_de_arcstdswap@ProcessIndividualTreebanks
			                    en=$treebank_train_en_arcstdswap@ProcessIndividualTreebanks
			                    es=$treebank_train_es_arcstdswap@ProcessIndividualTreebanks
			                    fi=$treebank_train_fi_arcstdswap@ProcessIndividualTreebanks
			                    fr=$treebank_train_fr_arcstdswap@ProcessIndividualTreebanks
			                    ga=$treebank_train_ga_arcstdswap@ProcessIndividualTreebanks
			                    hu=$treebank_train_hu_arcstdswap@ProcessIndividualTreebanks
			                    it=$treebank_train_it_arcstdswap@ProcessIndividualTreebanks
			                    sv=$treebank_train_sv_arcstdswap@ProcessIndividualTreebanks
                          v1_but_en=$v1_but_en@AggregateTrainTreebanks
                          v1_but_cs=$v1_but_cs@AggregateTrainTreebanks
                          v1_but_de=$v1_but_de@AggregateTrainTreebanks
                          v1_but_es=$v1_but_es@AggregateTrainTreebanks
                          v1_but_fi=$v1_but_fi@AggregateTrainTreebanks
                          v1_but_fr=$v1_but_fr@AggregateTrainTreebanks
                          v1_but_ga=$v1_but_ga@AggregateTrainTreebanks
                          v1_but_hu=$v1_but_hu@AggregateTrainTreebanks
                          v1_but_it=$v1_but_it@AggregateTrainTreebanks
                          v1_but_sv=$v1_but_sv@AggregateTrainTreebanks
                          v1=$v1@AggregateTrainTreebanks
                          )
  < embeddings=(EmbeddingsType: 
        jiangacl15_cca_ende=$ende_embeddings@PrepareJiangAcl15CcaEmbeddings
				jiangacl15_cca_enes=$enes_embeddings@PrepareJiangAcl15CcaEmbeddings
        jiangacl15_cca_enfr=$enfr_embeddings@PrepareJiangAcl15CcaEmbeddings
				jiangacl15_proj=$embeddings@PrepareJiangAcl15ProjEmbeddings
				word_clusters=$cluster_embeddings@TrainEmbeddingsOfWordClusters
        none="")
   :: use_pretrained=@
   :: lstm_parser=@
   < model=$latest_model@TrainParser
   < test=(EvalLang: cs=$treebank_test_cs_arcstdswap@ProcessIndividualTreebanks
			  de=$treebank_test_de_arcstdswap@ProcessIndividualTreebanks
		    en=$treebank_test_en_arcstdswap@ProcessIndividualTreebanks
		    es=$treebank_test_es_arcstdswap@ProcessIndividualTreebanks
		    fi=$treebank_test_fi_arcstdswap@ProcessIndividualTreebanks
		    fr=$treebank_test_fr_arcstdswap@ProcessIndividualTreebanks
		    ga=$treebank_test_ga_arcstdswap@ProcessIndividualTreebanks
		    hu=$treebank_test_hu_arcstdswap@ProcessIndividualTreebanks
		    it=$treebank_test_it_arcstdswap@ProcessIndividualTreebanks
		    sv=$treebank_test_sv_arcstdswap@ProcessIndividualTreebanks)
   < test_conll=(EvalLang:
              cs=$treebank_test_cs_conllx_lc_coarse_langprefix_clusterid@ProcessIndividualTreebanks
		          de=$treebank_test_de_conllx_lc_coarse_langprefix_clusterid@ProcessIndividualTreebanks
		          en=$treebank_test_en_conllx_lc_coarse_langprefix_clusterid@ProcessIndividualTreebanks
		          es=$treebank_test_es_conllx_lc_coarse_langprefix_clusterid@ProcessIndividualTreebanks
		          fi=$treebank_test_fi_conllx_lc_coarse_langprefix_clusterid@ProcessIndividualTreebanks
		          fr=$treebank_test_fr_conllx_lc_coarse_langprefix_clusterid@ProcessIndividualTreebanks
		          ga=$treebank_test_ga_conllx_lc_coarse_langprefix_clusterid@ProcessIndividualTreebanks
		          hu=$treebank_test_hu_conllx_lc_coarse_langprefix_clusterid@ProcessIndividualTreebanks
		          it=$treebank_test_it_conllx_lc_coarse_langprefix_clusterid@ProcessIndividualTreebanks
		          sv=$treebank_test_sv_conllx_lc_coarse_langprefix_clusterid@ProcessIndividualTreebanks)
   > predictions
   > score
   > stderr
  :: pos_dim=@
  :: hidden_dim=@
  :: lstm_input_dim=@
  :: rel_dim=@
  :: action_dim=@
  :: unk_prob=@
  :: layers=@
  :: input_dim=@
{
  # rebuild the parser
  pushd $lstm_parser
  make -j2
  popd

  # construct the command
  command="$lstm_parser/parser/lstm-parse -T $train --model $model -d $test -P"

  if [ $input_dim ]; then
    command="$command --input_dim $input_dim "
  fi

  if [ $use_pretrained ]; then
    command="$command -w $embeddings --pretrained_dim 100 "
  fi

  if [ $use_typology ]; then
    command="$command -y $lstm_parser/typological_properties.txt "
  fi

  if [ $pos_dim ]; then
    command="$command --pos_dim $pos_dim "
  fi

  if [ $hidden_dim ]; then
    command="$command --hidden_dim $hidden_dim "
  fi

  if [ $lstm_input_dim ]; then
    command="$command --lstm_input_dim $lstm_input_dim "
  fi

  if [ $rel_dim ]; then
    command="$command --rel_dim $rel_dim "
  fi

  if [ $action_dim ]; then
    command="$command --action_dim $action_dim "
  fi

  if [ $unk_prob ]; then
    command="$command --unk_prob $unk_prob "
  fi

  if [ $layers ]; then
    command="$command --layers $layers "
  fi

  # execute command
  echo "executing $command..."
  $command 2> $stderr > $predictions

  # evaluate
  perl $conll_eval -g $test_conll -s $predictions -q > $score
}

task EvaluateParserOnDev
   :: conll_eval=@
   < train=(TrainLang:    cs=$treebank_train_cs_arcstdswap@ProcessIndividualTreebanks
                  			  de=$treebank_train_de_arcstdswap@ProcessIndividualTreebanks
			                    en=$treebank_train_en_arcstdswap@ProcessIndividualTreebanks
			                    es=$treebank_train_es_arcstdswap@ProcessIndividualTreebanks
			                    fi=$treebank_train_fi_arcstdswap@ProcessIndividualTreebanks
			                    fr=$treebank_train_fr_arcstdswap@ProcessIndividualTreebanks
			                    ga=$treebank_train_ga_arcstdswap@ProcessIndividualTreebanks
			                    hu=$treebank_train_hu_arcstdswap@ProcessIndividualTreebanks
			                    it=$treebank_train_it_arcstdswap@ProcessIndividualTreebanks
			                    sv=$treebank_train_sv_arcstdswap@ProcessIndividualTreebanks
                          v1_but_en=$v1_but_en@AggregateTrainTreebanks
                          v1_but_cs=$v1_but_cs@AggregateTrainTreebanks
                          v1_but_de=$v1_but_de@AggregateTrainTreebanks
                          v1_but_es=$v1_but_es@AggregateTrainTreebanks
                          v1_but_fi=$v1_but_fi@AggregateTrainTreebanks
                          v1_but_fr=$v1_but_fr@AggregateTrainTreebanks
                          v1_but_ga=$v1_but_ga@AggregateTrainTreebanks
                          v1_but_hu=$v1_but_hu@AggregateTrainTreebanks
                          v1_but_it=$v1_but_it@AggregateTrainTreebanks
                          v1_but_sv=$v1_but_sv@AggregateTrainTreebanks
                          v1=$v1@AggregateTrainTreebanks
                          )
  < embeddings=(EmbeddingsType: 
        jiangacl15_cca_ende=$ende_embeddings@PrepareJiangAcl15CcaEmbeddings
				jiangacl15_cca_enes=$enes_embeddings@PrepareJiangAcl15CcaEmbeddings
        jiangacl15_cca_enfr=$enfr_embeddings@PrepareJiangAcl15CcaEmbeddings
				jiangacl15_proj=$embeddings@PrepareJiangAcl15ProjEmbeddings
				word_clusters=$cluster_embeddings@TrainEmbeddingsOfWordClusters
        none="")
   :: use_pretrained=@
   :: lstm_parser=@
   < model=$latest_model@TrainParser
   < dev=(EvalLang: cs=$treebank_dev_cs_arcstdswap@ProcessIndividualTreebanks
            		    de=$treebank_dev_de_arcstdswap@ProcessIndividualTreebanks
		                en=$treebank_dev_en_arcstdswap@ProcessIndividualTreebanks
		                es=$treebank_dev_es_arcstdswap@ProcessIndividualTreebanks
		                fi=$treebank_dev_fi_arcstdswap@ProcessIndividualTreebanks
		                fr=$treebank_dev_fr_arcstdswap@ProcessIndividualTreebanks
		                ga=$treebank_dev_ga_arcstdswap@ProcessIndividualTreebanks
		                hu=$treebank_dev_hu_arcstdswap@ProcessIndividualTreebanks
		                it=$treebank_dev_it_arcstdswap@ProcessIndividualTreebanks
		                sv=$treebank_dev_sv_arcstdswap@ProcessIndividualTreebanks)
   < dev_conll=(EvalLang: cs=$treebank_dev_cs_conllx_lc_coarse_langprefix_clusterid@ProcessIndividualTreebanks
	                        de=$treebank_dev_de_conllx_lc_coarse_langprefix_clusterid@ProcessIndividualTreebanks
		                      en=$treebank_dev_en_conllx_lc_coarse_langprefix_clusterid@ProcessIndividualTreebanks
		                      es=$treebank_dev_es_conllx_lc_coarse_langprefix_clusterid@ProcessIndividualTreebanks
		                      fi=$treebank_dev_fi_conllx_lc_coarse_langprefix_clusterid@ProcessIndividualTreebanks
		                      fr=$treebank_dev_fr_conllx_lc_coarse_langprefix_clusterid@ProcessIndividualTreebanks
		                      ga=$treebank_dev_ga_conllx_lc_coarse_langprefix_clusterid@ProcessIndividualTreebanks
		                      hu=$treebank_dev_hu_conllx_lc_coarse_langprefix_clusterid@ProcessIndividualTreebanks
		                      it=$treebank_dev_it_conllx_lc_coarse_langprefix_clusterid@ProcessIndividualTreebanks
		                      sv=$treebank_dev_sv_conllx_lc_coarse_langprefix_clusterid@ProcessIndividualTreebanks)
   > predictions
  :: use_typology=@
   > score
   > stderr
  :: pos_dim=@
  :: hidden_dim=@
  :: lstm_input_dim=@
  :: rel_dim=@
  :: action_dim=@
  :: unk_prob=@
  :: layers=@
  :: input_dim=@

{
  # rebuild the parser
  pushd $lstm_parser
  make -j2
  popd

  # construct the command
  command="$lstm_parser/parser/lstm-parse -T $train --model $model -d $dev -P"

  if [ $input_dim ]; then
    command="$command --input_dim $input_dim "
  fi

  if [ $use_pretrained ]; then
    command="$command -w $embeddings --pretrained_dim 100 "
  fi

  if [ $use_typology ]; then
    command="$command -y $lstm_parser/typological_properties.txt "
  fi

  if [ $pos_dim ]; then
    command="$command --pos_dim $pos_dim "
  fi

  if [ $hidden_dim ]; then
    command="$command --hidden_dim $hidden_dim "
  fi

  if [ $lstm_input_dim ]; then
    command="$command --lstm_input_dim $lstm_input_dim "
  fi

  if [ $rel_dim ]; then
    command="$command --rel_dim $rel_dim "
  fi

  if [ $action_dim ]; then
    command="$command --action_dim $action_dim "
  fi

  if [ $unk_prob ]; then
    command="$command --unk_prob $unk_prob "
  fi

  if [ $layers ]; then
    command="$command --layers $layers "
  fi

  # execute command
  echo "executing $command..."
  $command 2> $stderr > $predictions

  # evaluate
  perl $conll_eval -g $dev_conll -s $predictions -q > $score
} 

task TrainParser
  :: use_typology=@
  :: lstm_parser=@
   < global_clustermap=$global_clustermap@ProcessIndividualTreebanks
   < train=(TrainLang:    cs=$treebank_train_cs_arcstdswap@ProcessIndividualTreebanks
                  			  de=$treebank_train_de_arcstdswap@ProcessIndividualTreebanks
			                    en=$treebank_train_en_arcstdswap@ProcessIndividualTreebanks
			                    es=$treebank_train_es_arcstdswap@ProcessIndividualTreebanks
			                    fi=$treebank_train_fi_arcstdswap@ProcessIndividualTreebanks
			                    fr=$treebank_train_fr_arcstdswap@ProcessIndividualTreebanks
			                    ga=$treebank_train_ga_arcstdswap@ProcessIndividualTreebanks
			                    hu=$treebank_train_hu_arcstdswap@ProcessIndividualTreebanks
			                    it=$treebank_train_it_arcstdswap@ProcessIndividualTreebanks
			                    sv=$treebank_train_sv_arcstdswap@ProcessIndividualTreebanks
                          v1_but_en=$v1_but_en@AggregateTrainTreebanks
                          v1_but_cs=$v1_but_cs@AggregateTrainTreebanks
                          v1_but_de=$v1_but_de@AggregateTrainTreebanks
                          v1_but_es=$v1_but_es@AggregateTrainTreebanks
                          v1_but_fi=$v1_but_fi@AggregateTrainTreebanks
                          v1_but_fr=$v1_but_fr@AggregateTrainTreebanks
                          v1_but_ga=$v1_but_ga@AggregateTrainTreebanks
                          v1_but_hu=$v1_but_hu@AggregateTrainTreebanks
                          v1_but_it=$v1_but_it@AggregateTrainTreebanks
                          v1_but_sv=$v1_but_sv@AggregateTrainTreebanks
                          v1=$v1@AggregateTrainTreebanks
                          )
   < dev=(TrainLang:    cs=$treebank_dev_cs_arcstdswap@ProcessIndividualTreebanks
                  			  de=$treebank_dev_de_arcstdswap@ProcessIndividualTreebanks
			                    en=$treebank_dev_en_arcstdswap@ProcessIndividualTreebanks
			                    es=$treebank_dev_es_arcstdswap@ProcessIndividualTreebanks
			                    fi=$treebank_dev_fi_arcstdswap@ProcessIndividualTreebanks
			                    fr=$treebank_dev_fr_arcstdswap@ProcessIndividualTreebanks
			                    ga=$treebank_dev_ga_arcstdswap@ProcessIndividualTreebanks
			                    hu=$treebank_dev_hu_arcstdswap@ProcessIndividualTreebanks
			                    it=$treebank_dev_it_arcstdswap@ProcessIndividualTreebanks
			                    sv=$treebank_dev_sv_arcstdswap@ProcessIndividualTreebanks
                          v1_but_en=$v1_but_en@AggregateDevTreebanks
                          v1_but_cs=$v1_but_cs@AggregateDevTreebanks
                          v1_but_de=$v1_but_de@AggregateDevTreebanks
                          v1_but_es=$v1_but_es@AggregateDevTreebanks
                          v1_but_fi=$v1_but_fi@AggregateDevTreebanks
                          v1_but_fr=$v1_but_fr@AggregateDevTreebanks
                          v1_but_ga=$v1_but_ga@AggregateDevTreebanks
                          v1_but_hu=$v1_but_hu@AggregateDevTreebanks
                          v1_but_it=$v1_but_it@AggregateDevTreebanks
                          v1_but_sv=$v1_but_sv@AggregateDevTreebanks
                          v1=$v1@AggregateDevTreebanks
                          )
   < embeddings=(EmbeddingsType: 
        jiangacl15_cca_ende=$ende_embeddings@PrepareJiangAcl15CcaEmbeddings
				jiangacl15_cca_enes=$enes_embeddings@PrepareJiangAcl15CcaEmbeddings
        jiangacl15_cca_enfr=$enfr_embeddings@PrepareJiangAcl15CcaEmbeddings
				jiangacl15_proj=$embeddings@PrepareJiangAcl15ProjEmbeddings
				word_clusters=$cluster_embeddings@TrainEmbeddingsOfWordClusters
        none="")
  :: use_pretrained=@
  > stderr
  > stdout
  > latest_model
  :: pos_dim=@
  :: hidden_dim=@
  :: lstm_input_dim=@
  :: rel_dim=@
  :: action_dim=@
  :: unk_prob=@
  :: layers=@
  :: input_dim=@
  :: use_spelling=@
{
  # rebuild the parser
  pushd $lstm_parser
  make -j2
  popd

  # construct the command
  command="$lstm_parser/parser/lstm-parse -T $train -d $dev -t -P --clusters $global_clustermap"

  if [ $input_dim ]; then
    command="$command --input_dim $input_dim "
  fi

  if [ $use_pretrained ]; then
    command="$command -w $embeddings --pretrained_dim 100 "
  fi

  if [ $use_typology ]; then
    command="$command -y $lstm_parser/typological_properties.txt "
  fi

  if [ $pos_dim ]; then
    command="$command --pos_dim $pos_dim "
  fi

  if [ $hidden_dim ]; then
    command="$command --hidden_dim $hidden_dim "
  fi

  if [ $lstm_input_dim ]; then
    command="$command --lstm_input_dim $lstm_input_dim "
  fi

  if [ $rel_dim ]; then
    command="$command --rel_dim $rel_dim "
  fi

  if [ $action_dim ]; then
    command="$command --action_dim $action_dim "
  fi

  if [ $unk_prob ]; then
    command="$command --unk_prob $unk_prob "
  fi

  if [ $use_spelling ]; then
    command="$command --use_spelling "
  fi

  if [ $layers ]; then
    command="$command --layers $layers "
  fi

  # execute command
  echo "executing $command..."
  $command 2> $stderr > $stdout
}

# Aggregate treebanks of two or more source languages
# arrow #14, #15
task AggregateTrainTreebanks
   ## individual files (arcstdswap)
   < en=$treebank_train_en_arcstdswap@ProcessIndividualTreebanks
   < cs=$treebank_train_cs_arcstdswap@ProcessIndividualTreebanks
   < de=$treebank_train_de_arcstdswap@ProcessIndividualTreebanks
   < es=$treebank_train_es_arcstdswap@ProcessIndividualTreebanks
   < fi=$treebank_train_fi_arcstdswap@ProcessIndividualTreebanks
   < fr=$treebank_train_fr_arcstdswap@ProcessIndividualTreebanks
   < ga=$treebank_train_ga_arcstdswap@ProcessIndividualTreebanks
   < hu=$treebank_train_hu_arcstdswap@ProcessIndividualTreebanks
   < it=$treebank_train_it_arcstdswap@ProcessIndividualTreebanks
   < sv=$treebank_train_sv_arcstdswap@ProcessIndividualTreebanks

   # outputs files (arcstdswp)

   > v1_but_en
   > v1_but_sv
   > v1_but_cs
   > v1_but_de
   > v1_but_es
   > v1_but_fi
   > v1_but_fr
   > v1_but_ga
   > v1_but_hu	
   > v1_but_it
   > v1

{
  # train with nine treebanks (arcstdswp)
  cat $cs $de $es $fi $fr $ga $hu $it $sv > $v1_but_en # target = any but en
  cat $en $cs $de $es $fi $fr $ga $hu $it > $v1_but_sv # target = any but cs, de, es, fi, fr, ga, hu, it
  cat $en $de $es $fi $fr $ga $hu $it $sv > $v1_but_cs # target = cs
  cat $en $cs $es $fi $fr $ga $hu $it $sv > $v1_but_de # target = de
  cat $en $cs $de $fi $fr $ga $hu $it $sv > $v1_but_es # target = es
  cat $en $cs $de $es $fr $ga $hu $it $sv > $v1_but_fi # target = fi
  cat $en $cs $de $es $fi $ga $hu $it $sv > $v1_but_fr # target = fr
  cat $en $cs $de $es $fi $fr $hu $it $sv > $v1_but_ga # target = ga
  cat $en $cs $de $es $fi $fr $ga $it $sv > $v1_but_hu # target = hu
  cat $en $cs $de $es $fi $fr $ga $hu $sv > $v1_but_it # target = it

  # train with ten treebanks (arcstdswp)
  cat $en $cs $de $es $fi $fr $ga $hu $it $sv > $v1

  cat v1_but_cs  | sed -r 's/-ARC\(.*\)$/-ARC(default)/' > v1_but_cs_nolabel
  cat v1_but_de  | sed -r 's/-ARC\(.*\)$/-ARC(default)/' > v1_but_de_nolabel 
  cat v1_but_en  | sed -r 's/-ARC\(.*\)$/-ARC(default)/' > v1_but_en_nolabel
  cat v1_but_es  | sed -r 's/-ARC\(.*\)$/-ARC(default)/' > v1_but_es_nolabel
  cat v1_but_fi  | sed -r 's/-ARC\(.*\)$/-ARC(default)/' > v1_but_fi_nolabel
  cat v1_but_fr  | sed -r 's/-ARC\(.*\)$/-ARC(default)/' > v1_but_fr_nolabel
  cat v1_but_ga  | sed -r 's/-ARC\(.*\)$/-ARC(default)/' > v1_but_ga_nolabel
  cat v1_but_hu  | sed -r 's/-ARC\(.*\)$/-ARC(default)/' > v1_but_hu_nolabel
  cat v1_but_it  | sed -r 's/-ARC\(.*\)$/-ARC(default)/' > v1_but_it_nolabel
  cat v1_but_sv  | sed -r 's/-ARC\(.*\)$/-ARC(default)/' > v1_but_sv_nolabel
  cat v1  | sed -r 's/-ARC\(.*\)$/-ARC(default)/' > v1_nolabel
}


# Aggregate dev sections of the treebanks of two or more source languages
# arrow #14, #15
task AggregateDevTreebanks
   ## input files (arcstdswap)
   < en=$treebank_dev_en_arcstdswap@ProcessIndividualTreebanks
   < cs=$treebank_dev_cs_arcstdswap@ProcessIndividualTreebanks
   < de=$treebank_dev_de_arcstdswap@ProcessIndividualTreebanks
   < es=$treebank_dev_es_arcstdswap@ProcessIndividualTreebanks
   < fi=$treebank_dev_fi_arcstdswap@ProcessIndividualTreebanks
   < fr=$treebank_dev_fr_arcstdswap@ProcessIndividualTreebanks
   < ga=$treebank_dev_ga_arcstdswap@ProcessIndividualTreebanks
   < hu=$treebank_dev_hu_arcstdswap@ProcessIndividualTreebanks
   < it=$treebank_dev_it_arcstdswap@ProcessIndividualTreebanks
   < sv=$treebank_dev_sv_arcstdswap@ProcessIndividualTreebanks

  ## output files (arcstdswap)
   > v1_but_sv
   > v1_but_cs
   > v1_but_de
   > v1_but_en
   > v1_but_es
   > v1_but_fi
   > v1_but_fr
   > v1_but_ga
   > v1_but_hu	
   > v1_but_it
   > v1

  :: wammar_utils=@
{
  # only use the first 150 sentences from each language (arcstdswap)
  python $wammar_utils/clip-file-after-kth-match.py -i $cs -o cs.150 -k 150 -p "\n\n"
  python $wammar_utils/clip-file-after-kth-match.py -i $de -o de.150 -k 150 -p "\n\n"
  python $wammar_utils/clip-file-after-kth-match.py -i $en -o en.150 -k 150 -p "\n\n"
  python $wammar_utils/clip-file-after-kth-match.py -i $es -o es.150 -k 150 -p "\n\n"
  python $wammar_utils/clip-file-after-kth-match.py -i $fi -o fi.150 -k 150 -p "\n\n"
  python $wammar_utils/clip-file-after-kth-match.py -i $fr -o fr.150 -k 150 -p "\n\n"
  python $wammar_utils/clip-file-after-kth-match.py -i $ga -o ga.150 -k 150 -p "\n\n"
  python $wammar_utils/clip-file-after-kth-match.py -i $hu -o hu.150 -k 150 -p "\n\n"
  python $wammar_utils/clip-file-after-kth-match.py -i $it -o it.150 -k 150 -p "\n\n"
  python $wammar_utils/clip-file-after-kth-match.py -i $sv -o sv.150 -k 150 -p "\n\n"

  # train with nine treebanks
  cat cs.150 de.150 es.150 fi.150 fr.150 ga.150 hu.150 it.150 sv.150 > $v1_but_en # target = en
  cat en.150 cs.150 de.150 es.150 fi.150 fr.150 ga.150 hu.150 it.150 > $v1_but_sv # target = sv
  cat en.150 de.150 es.150 fi.150 fr.150 ga.150 hu.150 it.150 sv.150 > $v1_but_cs # target = cs
  cat en.150 cs.150 es.150 fi.150 fr.150 ga.150 hu.150 it.150 sv.150 > $v1_but_de # target = de
  cat en.150 cs.150 de.150 fi.150 fr.150 ga.150 hu.150 it.150 sv.150 > $v1_but_es # target = es
  cat en.150 cs.150 de.150 es.150 fr.150 ga.150 hu.150 it.150 sv.150 > $v1_but_fi # target = fi
  cat en.150 cs.150 de.150 es.150 fi.150 ga.150 hu.150 it.150 sv.150 > $v1_but_fr # target = fr
  cat en.150 cs.150 de.150 es.150 fi.150 fr.150 hu.150 it.150 sv.150 > $v1_but_ga # target = ga
  cat en.150 cs.150 de.150 es.150 fi.150 fr.150 ga.150 it.150 sv.150 > $v1_but_hu # target = hu
  cat en.150 cs.150 de.150 es.150 fi.150 fr.150 ga.150 hu.150 sv.150 > $v1_but_it # target = it
 
  # train with ten treebanks
  cat en.150 cs.150 de.150 es.150 fi.150 fr.150 ga.150 hu.150 it.150 sv.150 > $v1

  # no label version
  cat v1_but_cs  | sed -r 's/-ARC\(.*\)$/-ARC(default)/' > v1_but_cs_nolabel
  cat v1_but_de  | sed -r 's/-ARC\(.*\)$/-ARC(default)/' > v1_but_de_nolabel 
  cat v1_but_en  | sed -r 's/-ARC\(.*\)$/-ARC(default)/' > v1_but_en_nolabel
  cat v1_but_es  | sed -r 's/-ARC\(.*\)$/-ARC(default)/' > v1_but_es_nolabel
  cat v1_but_fi  | sed -r 's/-ARC\(.*\)$/-ARC(default)/' > v1_but_fi_nolabel
  cat v1_but_fr  | sed -r 's/-ARC\(.*\)$/-ARC(default)/' > v1_but_fr_nolabel
  cat v1_but_ga  | sed -r 's/-ARC\(.*\)$/-ARC(default)/' > v1_but_ga_nolabel
  cat v1_but_hu  | sed -r 's/-ARC\(.*\)$/-ARC(default)/' > v1_but_hu_nolabel
  cat v1_but_it  | sed -r 's/-ARC\(.*\)$/-ARC(default)/' > v1_but_it_nolabel
  cat v1_but_sv  | sed -r 's/-ARC\(.*\)$/-ARC(default)/' > v1_but_sv_nolabel
  cat v1  | sed -r 's/-ARC\(.*\)$/-ARC(default)/' > v1_nolabel

}

# Process individual conllu files in the universal dependencies treebanks, transforming each into
# the arc-standard format (with swap actions).
# arrow #12
task ProcessIndividualTreebanks
  :: wammar_utils=@
  :: parse_oracle_arc_std_swap_jar=@

  :: treebank_train_cs=@
  :: treebank_train_de=@
  :: treebank_train_en=@
  :: treebank_train_es=@
  :: treebank_train_fi=@
  :: treebank_train_fr=@
  :: treebank_train_ga=@
  :: treebank_train_hu=@
  :: treebank_train_it=@
  :: treebank_train_sv=@

  :: treebank_dev_cs=@
  :: treebank_dev_de=@
  :: treebank_dev_en=@
  :: treebank_dev_es=@
  :: treebank_dev_fi=@
  :: treebank_dev_fr=@
  :: treebank_dev_ga=@
  :: treebank_dev_hu=@
  :: treebank_dev_it=@
  :: treebank_dev_sv=@

  :: treebank_test_cs=@
  :: treebank_test_de=@
  :: treebank_test_en=@
  :: treebank_test_es=@
  :: treebank_test_fi=@
  :: treebank_test_fr=@
  :: treebank_test_ga=@
  :: treebank_test_hu=@
  :: treebank_test_it=@
  :: treebank_test_sv=@

   < word_clusters=$augmented_word_clusters@TrainEmbeddingsOfWordClusters

   > treebank_train_cs_arcstdswap
   > treebank_train_de_arcstdswap
   > treebank_train_en_arcstdswap
   > treebank_train_es_arcstdswap
   > treebank_train_fi_arcstdswap
   > treebank_train_fr_arcstdswap
   > treebank_train_ga_arcstdswap
   > treebank_train_hu_arcstdswap
   > treebank_train_it_arcstdswap
   > treebank_train_sv_arcstdswap

   > treebank_dev_cs_arcstdswap
   > treebank_dev_de_arcstdswap
   > treebank_dev_en_arcstdswap
   > treebank_dev_es_arcstdswap
   > treebank_dev_fi_arcstdswap
   > treebank_dev_fr_arcstdswap
   > treebank_dev_ga_arcstdswap
   > treebank_dev_hu_arcstdswap
   > treebank_dev_it_arcstdswap
   > treebank_dev_sv_arcstdswap

   > treebank_test_cs_arcstdswap
   > treebank_test_de_arcstdswap
   > treebank_test_en_arcstdswap
   > treebank_test_es_arcstdswap
   > treebank_test_fi_arcstdswap
   > treebank_test_fr_arcstdswap
   > treebank_test_ga_arcstdswap
   > treebank_test_hu_arcstdswap
   > treebank_test_it_arcstdswap
   > treebank_test_sv_arcstdswap

   ## clustermap
   > global_clustermap

   ## clusterids
   > treebank_dev_cs_conllx_lc_coarse_langprefix_clusterid
   > treebank_dev_de_conllx_lc_coarse_langprefix_clusterid
   > treebank_dev_en_conllx_lc_coarse_langprefix_clusterid
   > treebank_dev_es_conllx_lc_coarse_langprefix_clusterid
   > treebank_dev_fi_conllx_lc_coarse_langprefix_clusterid
   > treebank_dev_fr_conllx_lc_coarse_langprefix_clusterid
   > treebank_dev_ga_conllx_lc_coarse_langprefix_clusterid
   > treebank_dev_hu_conllx_lc_coarse_langprefix_clusterid
   > treebank_dev_it_conllx_lc_coarse_langprefix_clusterid
   > treebank_dev_sv_conllx_lc_coarse_langprefix_clusterid

   > treebank_test_cs_conllx_lc_coarse_langprefix_clusterid
   > treebank_test_de_conllx_lc_coarse_langprefix_clusterid
   > treebank_test_en_conllx_lc_coarse_langprefix_clusterid
   > treebank_test_es_conllx_lc_coarse_langprefix_clusterid
   > treebank_test_fi_conllx_lc_coarse_langprefix_clusterid
   > treebank_test_fr_conllx_lc_coarse_langprefix_clusterid
   > treebank_test_ga_conllx_lc_coarse_langprefix_clusterid
   > treebank_test_hu_conllx_lc_coarse_langprefix_clusterid
   > treebank_test_it_conllx_lc_coarse_langprefix_clusterid
   > treebank_test_sv_conllx_lc_coarse_langprefix_clusterid

{
 # Setting the class path didn't work for me. Just copy the jar file to the current directory.
 cp $parse_oracle_arc_std_swap_jar ./ParseOracleArcStdSwap.jar

 # For each language and each of train/dev/test:
 #   * Remove phrases and comments, which were recently introduced in the conllu format, and is not
 #     supported by the oracle parse arc-std tool nor the LSTM parser.
 #   * Lowercase everything.
 #   * Swap the coarse vs. fine POS tag columns. We want to use the coarse one because it's universal,  
 #     but the arcstd oracle parse tool uses the fine POS. Hence the need for this step.
 #   * Extract oracle actions for training a transition-based parser.

  python $wammar_utils/convert-conllu-to-conll06.py -i $treebank_train_cs -o conllx
  python $wammar_utils/lowercase.py -i conllx -o conllx.lc
  python $wammar_utils/swap-conll-columns.py -i conllx.lc -o conllx.lc.coarse -j 4 -k 5
  python $wammar_utils/add-prefix-to-conll-column.py -i conllx.lc.coarse -o conllx.lc.coarse.langprefix -j 2 -p "cs:"
  python $wammar_utils/replace-words-in-conll-corpus.py -i conllx.lc.coarse.langprefix -o conllx.lc.coarse.langprefix.clusterid -d $word_clusters -p -c treebank_train_cs_clustermap
  java -jar ParseOracleArcStdSwap.jar -t -1 -l 1 -c conllx.lc.coarse.langprefix.clusterid -i conllx.lc.coarse.langprefix.clusterid > $treebank_train_cs_arcstdswap
  cp conllx.lc.coarse.langprefix.clusterid treebank_train_cs_conllx_lc_coarse_langprefix_clusterid
  cat $treebank_train_cs_arcstdswap  | sed -r 's/-ARC\(.*\)$/-ARC(default)/' > treebank_train_cs_arcstdswap_nolabel
  
  python $wammar_utils/convert-conllu-to-conll06.py -i $treebank_train_de -o conllx
  python $wammar_utils/lowercase.py -i conllx -o conllx.lc
  python $wammar_utils/swap-conll-columns.py -i conllx.lc -o conllx.lc.coarse -j 4 -k 5
  python $wammar_utils/add-prefix-to-conll-column.py -i conllx.lc.coarse -o conllx.lc.coarse.langprefix -j 2 -p "de:"
  python $wammar_utils/replace-words-in-conll-corpus.py -i conllx.lc.coarse.langprefix -o conllx.lc.coarse.langprefix.clusterid -d $word_clusters -p -c treebank_train_de_clustermap 
  java -jar ParseOracleArcStdSwap.jar -t -1 -l 1 -c conllx.lc.coarse.langprefix.clusterid -i conllx.lc.coarse.langprefix.clusterid > $treebank_train_de_arcstdswap
  cp conllx.lc.coarse.langprefix.clusterid treebank_train_de_conllx_lc_coarse_langprefix_clusterid
  cat $treebank_train_de_arcstdswap  | sed -r 's/-ARC\(.*\)$/-ARC(default)/' > treebank_train_de_arcstdswap_nolabel

  python $wammar_utils/convert-conllu-to-conll06.py -i $treebank_train_en -o conllx
  python $wammar_utils/lowercase.py -i conllx -o conllx.lc
  python $wammar_utils/swap-conll-columns.py -i conllx.lc -o conllx.lc.coarse -j 4 -k 5
  python $wammar_utils/add-prefix-to-conll-column.py -i conllx.lc.coarse -o conllx.lc.coarse.langprefix -j 2 -p "en:"
  python $wammar_utils/replace-words-in-conll-corpus.py -i conllx.lc.coarse.langprefix -o conllx.lc.coarse.langprefix.clusterid -d $word_clusters -p -c treebank_train_en_clustermap
  java -jar ParseOracleArcStdSwap.jar -t -1 -l 1 -c conllx.lc.coarse.langprefix.clusterid -i conllx.lc.coarse.langprefix.clusterid > $treebank_train_en_arcstdswap
  cp conllx.lc.coarse.langprefix.clusterid treebank_train_en_conllx_lc_coarse_langprefix_clusterid
  cat $treebank_train_en_arcstdswap  | sed -r 's/-ARC\(.*\)$/-ARC(default)/' > treebank_train_en_arcstdswap_nolabel

  python $wammar_utils/convert-conllu-to-conll06.py -i $treebank_train_es -o conllx
  python $wammar_utils/lowercase.py -i conllx -o conllx.lc
  python $wammar_utils/swap-conll-columns.py -i conllx.lc -o conllx.lc.coarse -j 4 -k 5
  python $wammar_utils/add-prefix-to-conll-column.py -i conllx.lc.coarse -o conllx.lc.coarse.langprefix -j 2 -p "es:"
  python $wammar_utils/replace-words-in-conll-corpus.py -i conllx.lc.coarse.langprefix -o conllx.lc.coarse.langprefix.clusterid -d $word_clusters -p -c treebank_train_es_clustermap
  java -jar ParseOracleArcStdSwap.jar -t -1 -l 1 -c conllx.lc.coarse.langprefix.clusterid -i conllx.lc.coarse.langprefix.clusterid > $treebank_train_es_arcstdswap
  cp conllx.lc.coarse.langprefix.clusterid treebank_train_es_conllx_lc_coarse_langprefix_clusterid
  cat $treebank_train_es_arcstdswap  | sed -r 's/-ARC\(.*\)$/-ARC(default)/' > treebank_train_es_arcstdswap_nolabel

  python $wammar_utils/convert-conllu-to-conll06.py -i $treebank_train_fi -o conllx
  python $wammar_utils/lowercase.py -i conllx -o conllx.lc
  python $wammar_utils/swap-conll-columns.py -i conllx.lc -o conllx.lc.coarse -j 4 -k 5
  python $wammar_utils/add-prefix-to-conll-column.py -i conllx.lc.coarse -o conllx.lc.coarse.langprefix -j 2 -p "fi:"
  python $wammar_utils/replace-words-in-conll-corpus.py -i conllx.lc.coarse.langprefix -o conllx.lc.coarse.langprefix.clusterid -d $word_clusters -p -c treebank_train_fi_clustermap
  java -jar ParseOracleArcStdSwap.jar -t -1 -l 1 -c conllx.lc.coarse.langprefix.clusterid -i conllx.lc.coarse.langprefix.clusterid > $treebank_train_fi_arcstdswap
  cp conllx.lc.coarse.langprefix.clusterid treebank_train_fi_conllx_lc_coarse_langprefix_clusterid
  cat $treebank_train_fi_arcstdswap  | sed -r 's/-ARC\(.*\)$/-ARC(default)/' > treebank_train_fi_arcstdswap_nolabel

  python $wammar_utils/convert-conllu-to-conll06.py -i $treebank_train_fr -o conllx
  python $wammar_utils/lowercase.py -i conllx -o conllx.lc
  python $wammar_utils/swap-conll-columns.py -i conllx.lc -o conllx.lc.coarse -j 4 -k 5
  python $wammar_utils/add-prefix-to-conll-column.py -i conllx.lc.coarse -o conllx.lc.coarse.langprefix -j 2 -p "fr:"
  python $wammar_utils/replace-words-in-conll-corpus.py -i conllx.lc.coarse.langprefix -o conllx.lc.coarse.langprefix.clusterid -d $word_clusters -p -c treebank_train_fr_clustermap
  java -jar ParseOracleArcStdSwap.jar -t -1 -l 1 -c conllx.lc.coarse.langprefix.clusterid -i conllx.lc.coarse.langprefix.clusterid > $treebank_train_fr_arcstdswap
  cp conllx.lc.coarse.langprefix.clusterid treebank_train_fr_conllx_lc_coarse_langprefix_clusterid
  cat $treebank_train_fr_arcstdswap  | sed -r 's/-ARC\(.*\)$/-ARC(default)/' > treebank_train_fr_arcstdswap_nolabel

  python $wammar_utils/convert-conllu-to-conll06.py -i $treebank_train_ga -o conllx
  python $wammar_utils/lowercase.py -i conllx -o conllx.lc
  python $wammar_utils/swap-conll-columns.py -i conllx.lc -o conllx.lc.coarse -j 4 -k 5
  python $wammar_utils/add-prefix-to-conll-column.py -i conllx.lc.coarse -o conllx.lc.coarse.langprefix -j 2 -p "ga:"
  python $wammar_utils/replace-words-in-conll-corpus.py -i conllx.lc.coarse.langprefix -o conllx.lc.coarse.langprefix.clusterid -d $word_clusters -p -c treebank_train_ga_clustermap
  java -jar ParseOracleArcStdSwap.jar -t -1 -l 1 -c conllx.lc.coarse.langprefix.clusterid -i conllx.lc.coarse.langprefix.clusterid > $treebank_train_ga_arcstdswap
  cp conllx.lc.coarse.langprefix.clusterid treebank_train_ga_conllx_lc_coarse_langprefix_clusterid
  cat $treebank_train_ga_arcstdswap  | sed -r 's/-ARC\(.*\)$/-ARC(default)/' > treebank_train_ga_arcstdswap_nolabel

  python $wammar_utils/convert-conllu-to-conll06.py -i $treebank_train_hu -o conllx
  python $wammar_utils/lowercase.py -i conllx -o conllx.lc
  python $wammar_utils/swap-conll-columns.py -i conllx.lc -o conllx.lc.coarse -j 4 -k 5
  python $wammar_utils/add-prefix-to-conll-column.py -i conllx.lc.coarse -o conllx.lc.coarse.langprefix -j 2 -p "hu:"
  python $wammar_utils/replace-words-in-conll-corpus.py -i conllx.lc.coarse.langprefix -o conllx.lc.coarse.langprefix.clusterid -d $word_clusters -p -c treebank_train_hu_clustermap
  java -jar ParseOracleArcStdSwap.jar -t -1 -l 1 -c conllx.lc.coarse.langprefix.clusterid -i conllx.lc.coarse.langprefix.clusterid > $treebank_train_hu_arcstdswap
  cp conllx.lc.coarse.langprefix.clusterid treebank_train_hu_conllx_lc_coarse_langprefix_clusterid
  cat $treebank_train_hu_arcstdswap  | sed -r 's/-ARC\(.*\)$/-ARC(default)/' > treebank_train_hu_arcstdswap_nolabel

  python $wammar_utils/convert-conllu-to-conll06.py -i $treebank_train_it -o conllx
  python $wammar_utils/lowercase.py -i conllx -o conllx.lc
  python $wammar_utils/swap-conll-columns.py -i conllx.lc -o conllx.lc.coarse -j 4 -k 5
  python $wammar_utils/add-prefix-to-conll-column.py -i conllx.lc.coarse -o conllx.lc.coarse.langprefix -j 2 -p "it:"
  python $wammar_utils/replace-words-in-conll-corpus.py -i conllx.lc.coarse.langprefix -o conllx.lc.coarse.langprefix.clusterid -d $word_clusters -p -c treebank_train_it_clustermap
  java -jar ParseOracleArcStdSwap.jar -t -1 -l 1 -c conllx.lc.coarse.langprefix.clusterid -i conllx.lc.coarse.langprefix.clusterid > $treebank_train_it_arcstdswap
  cp conllx.lc.coarse.langprefix.clusterid treebank_train_it_conllx_lc_coarse_langprefix_clusterid
  cat $treebank_train_it_arcstdswap  | sed -r 's/-ARC\(.*\)$/-ARC(default)/' > treebank_train_it_arcstdswap_nolabel

  python $wammar_utils/convert-conllu-to-conll06.py -i $treebank_train_sv -o conllx
  python $wammar_utils/lowercase.py -i conllx -o conllx.lc
  python $wammar_utils/swap-conll-columns.py -i conllx.lc -o conllx.lc.coarse -j 4 -k 5
  python $wammar_utils/add-prefix-to-conll-column.py -i conllx.lc.coarse -o conllx.lc.coarse.langprefix -j 2 -p "sv:"
  python $wammar_utils/replace-words-in-conll-corpus.py -i conllx.lc.coarse.langprefix -o conllx.lc.coarse.langprefix.clusterid -d $word_clusters -p -c treebank_train_sv_clustermap
  java -jar ParseOracleArcStdSwap.jar -t -1 -l 1 -c conllx.lc.coarse.langprefix.clusterid -i conllx.lc.coarse.langprefix.clusterid > $treebank_train_sv_arcstdswap
  cp conllx.lc.coarse.langprefix.clusterid treebank_train_sv_conllx_lc_coarse_langprefix_clusterid
  cat $treebank_train_sv_arcstdswap  | sed -r 's/-ARC\(.*\)$/-ARC(default)/' > treebank_train_sv_arcstdswap_nolabel

 ###########################
 # process dev treebanks

  python $wammar_utils/convert-conllu-to-conll06.py -i $treebank_dev_cs -o conllx
  python $wammar_utils/lowercase.py -i conllx -o conllx.lc
  python $wammar_utils/swap-conll-columns.py -i conllx.lc -o conllx.lc.coarse -j 4 -k 5
  python $wammar_utils/add-prefix-to-conll-column.py -i conllx.lc.coarse -o conllx.lc.coarse.langprefix -j 2 -p "cs:"
  python $wammar_utils/replace-words-in-conll-corpus.py -i conllx.lc.coarse.langprefix -o conllx.lc.coarse.langprefix.clusterid -d $word_clusters -p -c treebank_dev_cs_clustermap
  java -jar ParseOracleArcStdSwap.jar -t -1 -l 1 -c conllx.lc.coarse.langprefix.clusterid -i conllx.lc.coarse.langprefix.clusterid > $treebank_dev_cs_arcstdswap
  cp conllx.lc.coarse.langprefix.clusterid $treebank_dev_cs_conllx_lc_coarse_langprefix_clusterid
  cat $treebank_dev_cs_arcstdswap  | sed -r 's/-ARC\(.*\)$/-ARC(default)/' > treebank_dev_cs_arcstdswap_nolabel

  python $wammar_utils/convert-conllu-to-conll06.py -i $treebank_dev_de -o conllx
  python $wammar_utils/lowercase.py -i conllx -o conllx.lc
  python $wammar_utils/swap-conll-columns.py -i conllx.lc -o conllx.lc.coarse -j 4 -k 5
  python $wammar_utils/add-prefix-to-conll-column.py -i conllx.lc.coarse -o conllx.lc.coarse.langprefix -j 2 -p "de:"
  python $wammar_utils/replace-words-in-conll-corpus.py -i conllx.lc.coarse.langprefix -o conllx.lc.coarse.langprefix.clusterid -d $word_clusters -p -c treebank_dev_de_clustermap
  java -jar ParseOracleArcStdSwap.jar -t -1 -l 1 -c conllx.lc.coarse.langprefix.clusterid -i conllx.lc.coarse.langprefix.clusterid > $treebank_dev_de_arcstdswap
  cp conllx.lc.coarse.langprefix.clusterid $treebank_dev_de_conllx_lc_coarse_langprefix_clusterid
  cat $treebank_dev_de_arcstdswap  | sed -r 's/-ARC\(.*\)$/-ARC(default)/' > treebank_dev_de_arcstdswap_nolabel

  python $wammar_utils/convert-conllu-to-conll06.py -i $treebank_dev_en -o conllx
  python $wammar_utils/lowercase.py -i conllx -o conllx.lc
  python $wammar_utils/swap-conll-columns.py -i conllx.lc -o conllx.lc.coarse -j 4 -k 5
  python $wammar_utils/add-prefix-to-conll-column.py -i conllx.lc.coarse -o conllx.lc.coarse.langprefix -j 2 -p "en:"
  python $wammar_utils/replace-words-in-conll-corpus.py -i conllx.lc.coarse.langprefix -o conllx.lc.coarse.langprefix.clusterid -d $word_clusters -p -c treebank_dev_en_clustermap
  java -jar ParseOracleArcStdSwap.jar -t -1 -l 1 -c conllx.lc.coarse.langprefix.clusterid -i conllx.lc.coarse.langprefix.clusterid > $treebank_dev_en_arcstdswap
  cp conllx.lc.coarse.langprefix.clusterid $treebank_dev_en_conllx_lc_coarse_langprefix_clusterid
  cat $treebank_dev_en_arcstdswap  | sed -r 's/-ARC\(.*\)$/-ARC(default)/' > treebank_dev_en_arcstdswap_nolabel

  python $wammar_utils/convert-conllu-to-conll06.py -i $treebank_dev_es -o conllx
  python $wammar_utils/lowercase.py -i conllx -o conllx.lc
  python $wammar_utils/swap-conll-columns.py -i conllx.lc -o conllx.lc.coarse -j 4 -k 5
  python $wammar_utils/add-prefix-to-conll-column.py -i conllx.lc.coarse -o conllx.lc.coarse.langprefix -j 2 -p "es:"
  python $wammar_utils/replace-words-in-conll-corpus.py -i conllx.lc.coarse.langprefix -o conllx.lc.coarse.langprefix.clusterid -d $word_clusters -p -c treebank_dev_es_clustermap
  java -jar ParseOracleArcStdSwap.jar -t -1 -l 1 -c conllx.lc.coarse.langprefix.clusterid -i conllx.lc.coarse.langprefix.clusterid > $treebank_dev_es_arcstdswap
  cp conllx.lc.coarse.langprefix.clusterid $treebank_dev_es_conllx_lc_coarse_langprefix_clusterid
  cat $treebank_dev_es_arcstdswap  | sed -r 's/-ARC\(.*\)$/-ARC(default)/' > treebank_dev_es_arcstdswap_nolabel
 
  python $wammar_utils/convert-conllu-to-conll06.py -i $treebank_dev_fi -o conllx
  python $wammar_utils/lowercase.py -i conllx -o conllx.lc
  python $wammar_utils/swap-conll-columns.py -i conllx.lc -o conllx.lc.coarse -j 4 -k 5
  python $wammar_utils/add-prefix-to-conll-column.py -i conllx.lc.coarse -o conllx.lc.coarse.langprefix -j 2 -p "fi:"
  python $wammar_utils/replace-words-in-conll-corpus.py -i conllx.lc.coarse.langprefix -o conllx.lc.coarse.langprefix.clusterid -d $word_clusters -p -c treebank_dev_fi_clustermap
  java -jar ParseOracleArcStdSwap.jar -t -1 -l 1 -c conllx.lc.coarse.langprefix.clusterid -i conllx.lc.coarse.langprefix.clusterid > $treebank_dev_fi_arcstdswap
  cp conllx.lc.coarse.langprefix.clusterid $treebank_dev_fi_conllx_lc_coarse_langprefix_clusterid
  cat $treebank_dev_fi_arcstdswap  | sed -r 's/-ARC\(.*\)$/-ARC(default)/' > treebank_dev_fi_arcstdswap_nolabel
 
  python $wammar_utils/convert-conllu-to-conll06.py -i $treebank_dev_fr -o conllx
  python $wammar_utils/lowercase.py -i conllx -o conllx.lc
  python $wammar_utils/swap-conll-columns.py -i conllx.lc -o conllx.lc.coarse -j 4 -k 5
  python $wammar_utils/add-prefix-to-conll-column.py -i conllx.lc.coarse -o conllx.lc.coarse.langprefix -j 2 -p "fr:"
  python $wammar_utils/replace-words-in-conll-corpus.py -i conllx.lc.coarse.langprefix -o conllx.lc.coarse.langprefix.clusterid -d $word_clusters -p -c treebank_dev_fr_clustermap
  java -jar ParseOracleArcStdSwap.jar -t -1 -l 1 -c conllx.lc.coarse.langprefix.clusterid -i conllx.lc.coarse.langprefix.clusterid > $treebank_dev_fr_arcstdswap
  cp conllx.lc.coarse.langprefix.clusterid $treebank_dev_fr_conllx_lc_coarse_langprefix_clusterid
  cat $treebank_dev_fr_arcstdswap  | sed -r 's/-ARC\(.*\)$/-ARC(default)/' > treebank_dev_fr_arcstdswap_nolabel
 
  python $wammar_utils/convert-conllu-to-conll06.py -i $treebank_dev_ga -o conllx
  python $wammar_utils/lowercase.py -i conllx -o conllx.lc
  python $wammar_utils/swap-conll-columns.py -i conllx.lc -o conllx.lc.coarse -j 4 -k 5
  python $wammar_utils/add-prefix-to-conll-column.py -i conllx.lc.coarse -o conllx.lc.coarse.langprefix -j 2 -p "ga:"
  python $wammar_utils/replace-words-in-conll-corpus.py -i conllx.lc.coarse.langprefix -o conllx.lc.coarse.langprefix.clusterid -d $word_clusters -p -c treebank_dev_ga_clustermap
  java -jar ParseOracleArcStdSwap.jar -t -1 -l 1 -c conllx.lc.coarse.langprefix.clusterid -i conllx.lc.coarse.langprefix.clusterid > $treebank_dev_ga_arcstdswap
  cp conllx.lc.coarse.langprefix.clusterid $treebank_dev_ga_conllx_lc_coarse_langprefix_clusterid
  cat $treebank_dev_ga_arcstdswap  | sed -r 's/-ARC\(.*\)$/-ARC(default)/' > treebank_dev_ga_arcstdswap_nolabel
 
  python $wammar_utils/convert-conllu-to-conll06.py -i $treebank_dev_hu -o conllx
  python $wammar_utils/lowercase.py -i conllx -o conllx.lc
  python $wammar_utils/swap-conll-columns.py -i conllx.lc -o conllx.lc.coarse -j 4 -k 5
  python $wammar_utils/add-prefix-to-conll-column.py -i conllx.lc.coarse -o conllx.lc.coarse.langprefix -j 2 -p "hu:"
  python $wammar_utils/replace-words-in-conll-corpus.py -i conllx.lc.coarse.langprefix -o conllx.lc.coarse.langprefix.clusterid -d $word_clusters -p -c treebank_dev_hu_clustermap
  java -jar ParseOracleArcStdSwap.jar -t -1 -l 1 -c conllx.lc.coarse.langprefix.clusterid -i conllx.lc.coarse.langprefix.clusterid > $treebank_dev_hu_arcstdswap
  cp conllx.lc.coarse.langprefix.clusterid $treebank_dev_hu_conllx_lc_coarse_langprefix_clusterid
  cat $treebank_dev_hu_arcstdswap  | sed -r 's/-ARC\(.*\)$/-ARC(default)/' > treebank_dev_hu_arcstdswap_nolabel
 
  python $wammar_utils/convert-conllu-to-conll06.py -i $treebank_dev_it -o conllx
  python $wammar_utils/lowercase.py -i conllx -o conllx.lc
  python $wammar_utils/swap-conll-columns.py -i conllx.lc -o conllx.lc.coarse -j 4 -k 5
  python $wammar_utils/add-prefix-to-conll-column.py -i conllx.lc.coarse -o conllx.lc.coarse.langprefix -j 2 -p "it:"
  python $wammar_utils/replace-words-in-conll-corpus.py -i conllx.lc.coarse.langprefix -o conllx.lc.coarse.langprefix.clusterid -d $word_clusters -p -c treebank_dev_it_clustermap
  java -jar ParseOracleArcStdSwap.jar -t -1 -l 1 -c conllx.lc.coarse.langprefix.clusterid -i conllx.lc.coarse.langprefix.clusterid > $treebank_dev_it_arcstdswap
  cp conllx.lc.coarse.langprefix.clusterid $treebank_dev_it_conllx_lc_coarse_langprefix_clusterid
  cat $treebank_dev_it_arcstdswap  | sed -r 's/-ARC\(.*\)$/-ARC(default)/' > treebank_dev_it_arcstdswap_nolabel
 
  python $wammar_utils/convert-conllu-to-conll06.py -i $treebank_dev_sv -o conllx
  python $wammar_utils/lowercase.py -i conllx -o conllx.lc
  python $wammar_utils/swap-conll-columns.py -i conllx.lc -o conllx.lc.coarse -j 4 -k 5
  python $wammar_utils/add-prefix-to-conll-column.py -i conllx.lc.coarse -o conllx.lc.coarse.langprefix -j 2 -p "sv:"
  python $wammar_utils/replace-words-in-conll-corpus.py -i conllx.lc.coarse.langprefix -o conllx.lc.coarse.langprefix.clusterid -d $word_clusters -p -c treebank_dev_sv_clustermap
  java -jar ParseOracleArcStdSwap.jar -t -1 -l 1 -c conllx.lc.coarse.langprefix.clusterid -i conllx.lc.coarse.langprefix.clusterid > $treebank_dev_sv_arcstdswap
  cp conllx.lc.coarse.langprefix.clusterid $treebank_dev_sv_conllx_lc_coarse_langprefix_clusterid
  cat $treebank_dev_sv_arcstdswap  | sed -r 's/-ARC\(.*\)$/-ARC(default)/' > treebank_dev_sv_arcstdswap_nolabel
 
  python $wammar_utils/convert-conllu-to-conll06.py -i $treebank_test_cs -o conllx
  python $wammar_utils/lowercase.py -i conllx -o conllx.lc
  python $wammar_utils/swap-conll-columns.py -i conllx.lc -o conllx.lc.coarse -j 4 -k 5
  python $wammar_utils/add-prefix-to-conll-column.py -i conllx.lc.coarse -o conllx.lc.coarse.langprefix -j 2 -p "cs:"
  python $wammar_utils/replace-words-in-conll-corpus.py -i conllx.lc.coarse.langprefix -o conllx.lc.coarse.langprefix.clusterid -d $word_clusters -p -c treebank_test_cs_clustermap
  java -jar ParseOracleArcStdSwap.jar -t -1 -l 1 -c conllx.lc.coarse.langprefix.clusterid -i conllx.lc.coarse.langprefix.clusterid > $treebank_test_cs_arcstdswap
  cp conllx.lc.coarse.langprefix.clusterid $treebank_test_cs_conllx_lc_coarse_langprefix_clusterid
  cat $treebank_test_cs_arcstdswap  | sed -r 's/-ARC\(.*\)$/-ARC(default)/' > treebank_test_cs_arcstdswap_nolabel
 
  python $wammar_utils/convert-conllu-to-conll06.py -i $treebank_test_de -o conllx
  python $wammar_utils/lowercase.py -i conllx -o conllx.lc
  python $wammar_utils/swap-conll-columns.py -i conllx.lc -o conllx.lc.coarse -j 4 -k 5
  python $wammar_utils/add-prefix-to-conll-column.py -i conllx.lc.coarse -o conllx.lc.coarse.langprefix -j 2 -p "de:"
  python $wammar_utils/replace-words-in-conll-corpus.py -i conllx.lc.coarse.langprefix -o conllx.lc.coarse.langprefix.clusterid -d $word_clusters -p -c treebank_test_de_clustermap
  java -jar ParseOracleArcStdSwap.jar -t -1 -l 1 -c conllx.lc.coarse.langprefix.clusterid -i conllx.lc.coarse.langprefix.clusterid > $treebank_test_de_arcstdswap
  cp conllx.lc.coarse.langprefix.clusterid $treebank_test_de_conllx_lc_coarse_langprefix_clusterid
  cat $treebank_test_de_arcstdswap  | sed -r 's/-ARC\(.*\)$/-ARC(default)/' > treebank_test_de_arcstdswap_nolabel
 
  python $wammar_utils/convert-conllu-to-conll06.py -i $treebank_test_en -o conllx
  python $wammar_utils/lowercase.py -i conllx -o conllx.lc
  python $wammar_utils/swap-conll-columns.py -i conllx.lc -o conllx.lc.coarse -j 4 -k 5
  python $wammar_utils/add-prefix-to-conll-column.py -i conllx.lc.coarse -o conllx.lc.coarse.langprefix -j 2 -p "en:"
  python $wammar_utils/replace-words-in-conll-corpus.py -i conllx.lc.coarse.langprefix -o conllx.lc.coarse.langprefix.clusterid -d $word_clusters -p -c treebank_test_en_clustermap
  java -jar ParseOracleArcStdSwap.jar -t -1 -l 1 -c conllx.lc.coarse.langprefix.clusterid -i conllx.lc.coarse.langprefix.clusterid > $treebank_test_en_arcstdswap
  cp conllx.lc.coarse.langprefix.clusterid $treebank_test_en_conllx_lc_coarse_langprefix_clusterid
  cat $treebank_test_en_arcstdswap  | sed -r 's/-ARC\(.*\)$/-ARC(default)/' > treebank_test_en_arcstdswap_nolabel
 
  python $wammar_utils/convert-conllu-to-conll06.py -i $treebank_test_es -o conllx
  python $wammar_utils/lowercase.py -i conllx -o conllx.lc
  python $wammar_utils/swap-conll-columns.py -i conllx.lc -o conllx.lc.coarse -j 4 -k 5
  python $wammar_utils/add-prefix-to-conll-column.py -i conllx.lc.coarse -o conllx.lc.coarse.langprefix -j 2 -p "es:"
  python $wammar_utils/replace-words-in-conll-corpus.py -i conllx.lc.coarse.langprefix -o conllx.lc.coarse.langprefix.clusterid -d $word_clusters -p -c treebank_test_es_clustermap
  java -jar ParseOracleArcStdSwap.jar -t -1 -l 1 -c conllx.lc.coarse.langprefix.clusterid -i conllx.lc.coarse.langprefix.clusterid > $treebank_test_es_arcstdswap
  cp conllx.lc.coarse.langprefix.clusterid $treebank_test_es_conllx_lc_coarse_langprefix_clusterid
  cat $treebank_test_es_arcstdswap  | sed -r 's/-ARC\(.*\)$/-ARC(default)/' > treebank_test_es_arcstdswap_nolabel
 
  python $wammar_utils/convert-conllu-to-conll06.py -i $treebank_test_fi -o conllx
  python $wammar_utils/lowercase.py -i conllx -o conllx.lc
  python $wammar_utils/swap-conll-columns.py -i conllx.lc -o conllx.lc.coarse -j 4 -k 5
  python $wammar_utils/add-prefix-to-conll-column.py -i conllx.lc.coarse -o conllx.lc.coarse.langprefix -j 2 -p "fi:"
  python $wammar_utils/replace-words-in-conll-corpus.py -i conllx.lc.coarse.langprefix -o conllx.lc.coarse.langprefix.clusterid -d $word_clusters -p -c treebank_test_fi_clustermap
  java -jar ParseOracleArcStdSwap.jar -t -1 -l 1 -c conllx.lc.coarse.langprefix.clusterid -i conllx.lc.coarse.langprefix.clusterid > $treebank_test_fi_arcstdswap
  cp conllx.lc.coarse.langprefix.clusterid $treebank_test_fi_conllx_lc_coarse_langprefix_clusterid
  cat $treebank_test_fi_arcstdswap  | sed -r 's/-ARC\(.*\)$/-ARC(default)/' > treebank_test_fi_arcstdswap_nolabel
 
  python $wammar_utils/convert-conllu-to-conll06.py -i $treebank_test_fr -o conllx
  python $wammar_utils/lowercase.py -i conllx -o conllx.lc
  python $wammar_utils/swap-conll-columns.py -i conllx.lc -o conllx.lc.coarse -j 4 -k 5
  python $wammar_utils/add-prefix-to-conll-column.py -i conllx.lc.coarse -o conllx.lc.coarse.langprefix -j 2 -p "fr:"
  python $wammar_utils/replace-words-in-conll-corpus.py -i conllx.lc.coarse.langprefix -o conllx.lc.coarse.langprefix.clusterid -d $word_clusters -p -c treebank_test_fr_clustermap
  java -jar ParseOracleArcStdSwap.jar -t -1 -l 1 -c conllx.lc.coarse.langprefix.clusterid -i conllx.lc.coarse.langprefix.clusterid > $treebank_test_fr_arcstdswap
  cp conllx.lc.coarse.langprefix.clusterid $treebank_test_fr_conllx_lc_coarse_langprefix_clusterid
  cat $treebank_test_fr_arcstdswap  | sed -r 's/-ARC\(.*\)$/-ARC(default)/' > treebank_test_fr_arcstdswap_nolabel
 
  python $wammar_utils/convert-conllu-to-conll06.py -i $treebank_test_ga -o conllx
  python $wammar_utils/lowercase.py -i conllx -o conllx.lc
  python $wammar_utils/swap-conll-columns.py -i conllx.lc -o conllx.lc.coarse -j 4 -k 5
  python $wammar_utils/add-prefix-to-conll-column.py -i conllx.lc.coarse -o conllx.lc.coarse.langprefix -j 2 -p "ga:"
  python $wammar_utils/replace-words-in-conll-corpus.py -i conllx.lc.coarse.langprefix -o conllx.lc.coarse.langprefix.clusterid -d $word_clusters -p -c treebank_test_ga_clustermap
  java -jar ParseOracleArcStdSwap.jar -t -1 -l 1 -c conllx.lc.coarse.langprefix.clusterid -i conllx.lc.coarse.langprefix.clusterid > $treebank_test_ga_arcstdswap
  cp conllx.lc.coarse.langprefix.clusterid $treebank_test_ga_conllx_lc_coarse_langprefix_clusterid
  cat $treebank_test_ga_arcstdswap  | sed -r 's/-ARC\(.*\)$/-ARC(default)/' > treebank_test_ga_arcstdswap_nolabel
 
  python $wammar_utils/convert-conllu-to-conll06.py -i $treebank_test_hu -o conllx
  python $wammar_utils/lowercase.py -i conllx -o conllx.lc
  python $wammar_utils/swap-conll-columns.py -i conllx.lc -o conllx.lc.coarse -j 4 -k 5
  python $wammar_utils/add-prefix-to-conll-column.py -i conllx.lc.coarse -o conllx.lc.coarse.langprefix -j 2 -p "hu:"
  python $wammar_utils/replace-words-in-conll-corpus.py -i conllx.lc.coarse.langprefix -o conllx.lc.coarse.langprefix.clusterid -d $word_clusters -p -c treebank_test_hu_clustermap
  java -jar ParseOracleArcStdSwap.jar -t -1 -l 1 -c conllx.lc.coarse.langprefix.clusterid -i conllx.lc.coarse.langprefix.clusterid > $treebank_test_hu_arcstdswap
  cp conllx.lc.coarse.langprefix.clusterid $treebank_test_hu_conllx_lc_coarse_langprefix_clusterid
  cat $treebank_test_hu_arcstdswap  | sed -r 's/-ARC\(.*\)$/-ARC(default)/' > treebank_test_hu_arcstdswap_nolabel
 
  python $wammar_utils/convert-conllu-to-conll06.py -i $treebank_test_it -o conllx
  python $wammar_utils/lowercase.py -i conllx -o conllx.lc
  python $wammar_utils/swap-conll-columns.py -i conllx.lc -o conllx.lc.coarse -j 4 -k 5
  python $wammar_utils/add-prefix-to-conll-column.py -i conllx.lc.coarse -o conllx.lc.coarse.langprefix -j 2 -p "it:"
  python $wammar_utils/replace-words-in-conll-corpus.py -i conllx.lc.coarse.langprefix -o conllx.lc.coarse.langprefix.clusterid -d $word_clusters -p -c treebank_test_it_clustermap
  java -jar ParseOracleArcStdSwap.jar -t -1 -l 1 -c conllx.lc.coarse.langprefix.clusterid -i conllx.lc.coarse.langprefix.clusterid > $treebank_test_it_arcstdswap
  cp conllx.lc.coarse.langprefix.clusterid $treebank_test_it_conllx_lc_coarse_langprefix_clusterid
  cat $treebank_test_it_arcstdswap  | sed -r 's/-ARC\(.*\)$/-ARC(default)/' > treebank_test_it_arcstdswap_nolabel
 
  python $wammar_utils/convert-conllu-to-conll06.py -i $treebank_test_sv -o conllx
  python $wammar_utils/lowercase.py -i conllx -o conllx.lc
  python $wammar_utils/swap-conll-columns.py -i conllx.lc -o conllx.lc.coarse -j 4 -k 5
  python $wammar_utils/add-prefix-to-conll-column.py -i conllx.lc.coarse -o conllx.lc.coarse.langprefix -j 2 -p "sv:"
  python $wammar_utils/replace-words-in-conll-corpus.py -i conllx.lc.coarse.langprefix -o conllx.lc.coarse.langprefix.clusterid -d $word_clusters -p -c treebank_test_sv_clustermap
  java -jar ParseOracleArcStdSwap.jar -t -1 -l 1 -c conllx.lc.coarse.langprefix.clusterid -i conllx.lc.coarse.langprefix.clusterid > $treebank_test_sv_arcstdswap
  cp conllx.lc.coarse.langprefix.clusterid $treebank_test_sv_conllx_lc_coarse_langprefix_clusterid
  cat $treebank_test_sv_arcstdswap  | sed -r 's/-ARC\(.*\)$/-ARC(default)/' > treebank_test_sv_arcstdswap_nolabel

  # create global cluster map
  cat  *_clustermap > $global_clustermap
}

# Jiang Guo used a simple heuristic, which he calls PROJ, for inducing multilingual word embeddings. 
# First, he initialized English (monolingual) word embeddings using word2vec.
# Then he tuned the English embeddings towards the training objective of the English parser. 
# Then he computed the embeddings of words in other languages in the same (English) vector space by computing, 
# for each non-English word, the weighted average embedding of translationally-equivalent English words.
# Until Jiang makes the code for inducing these embeddings available, we can only compare to the
# final embeddings he obtained. 
# arrow #11
task PrepareJiangAcl15ProjEmbeddings
  :: jiangacl15_proj_en=@
  :: jiangacl15_proj_es=@
  :: jiangacl15_proj_fr=@
  :: jiangacl15_proj_de=@
  :: wammar_utils=@
   > embeddings
{
  
  python $wammar_utils/prefix.py -i $jiangacl15_proj_en -o jiangacl15_proj_en.langprefix -p "en:"
  python $wammar_utils/prefix.py -i $jiangacl15_proj_es -o jiangacl15_proj_es.langprefix -p "es:"
  python $wammar_utils/prefix.py -i $jiangacl15_proj_fr -o jiangacl15_proj_fr.langprefix -p "fr:"
  python $wammar_utils/prefix.py -i $jiangacl15_proj_de -o jiangacl15_proj_de.langprefix -p "de:"
  cat jiangacl15_proj_en.langprefix jiangacl15_proj_es.langprefix jiangacl15_proj_fr.langprefix jiangacl15_proj_de.langprefix >  $embeddings 
  rm jiangacl15_proj_*
}

# Using bilingual CCA (Faruqui and Dyer 2014), Jiang Guo obtained embeddings for three language pairs.
# This task just puts each of these embeddings in the same format as our home-grown multilingual embeddings.
# Note that these embeddings are only good for transferring from one source language to one target language.
# arrow #11(b)
task PrepareJiangAcl15CcaEmbeddings
   :: jiangacl15_cca_ende_en=@
   :: jiangacl15_cca_ende_de=@
   :: jiangacl15_cca_enfr_en=@
   :: jiangacl15_cca_enfr_fr=@
   :: jiangacl15_cca_enes_en=@
   :: jiangacl15_cca_enes_es=@
   :: wammar_utils=@
    > ende_embeddings
    > enfr_embeddings
    > enes_embeddings
{
  python $wammar_utils/prefix.py -i $jiangacl15_cca_ende_en -o jiangacl15_cca_ende_en.langprefix -p "en:"
  python $wammar_utils/prefix.py -i $jiangacl15_cca_ende_de -o jiangacl15_cca_ende_de.langprefix -p "de:"
  cat jiangacl15_cca_ende_en.langprefix jiangacl15_cca_ende_de.langprefix > $ende_embeddings
  python $wammar_utils/prefix.py -i $jiangacl15_cca_enfr_en -o jiangacl15_cca_enfr_en.langprefix -p "en:"
  python $wammar_utils/prefix.py -i $jiangacl15_cca_enfr_fr -o jiangacl15_cca_enfr_fr.langprefix -p "fr:"
  cat jiangacl15_cca_enfr_en.langprefix jiangacl15_cca_enfr_fr.langprefix > $enfr_embeddings
  python $wammar_utils/prefix.py -i $jiangacl15_cca_enes_en -o jiangacl15_cca_enes_en.langprefix -p "en:"
  python $wammar_utils/prefix.py -i $jiangacl15_cca_enes_es -o jiangacl15_cca_enes_es.langprefix -p "es:"
  cat jiangacl15_cca_enes_en.langprefix jiangacl15_cca_enes_es.langprefix > $enes_embeddings
  rm jiangacl15_cca_*
}

# First replace each word in the monolingual corpora with the string that identifies its multilingual word cluster.
# Then concatenate all monolingual corpora in the same file and train incremental_word2vec structured skipgram embeddings for
# the word clusters. Finally, for each cluster of size K words, repeat its embedding K times, once for each of the
# K words.
# arrow #7, #8, #9, and #10
task TrainEmbeddingsOfWordClusters
   :: corpus_cs=@
   :: corpus_de=@
   :: corpus_en=@
   :: corpus_es=@
   :: corpus_fi=@
   :: corpus_fr=@
   :: corpus_ga=@
   :: corpus_hu=@
   :: corpus_it=@
   :: corpus_sv=@
   :: wammar_utils=@
   :: incremental_word2vec=@
   :: cdec=@
    < word_clusters=$word_clusters@MakeMultilingualWordClusters
    > cluster_embeddings
    > augmented_word_clusters
   :: pretrained_dim=@
{
  python $wammar_utils/replace-words-in-monolingual-corpus.py -d $word_clusters -l "cs:" -i $corpus_cs -o corpus_cs.langprefix -od word_clusters_cs
  python $wammar_utils/replace-words-in-monolingual-corpus.py -d word_clusters_cs -l "de:" -i $corpus_de -o corpus_de.langprefix -od word_clusters_cs+de
  python $wammar_utils/replace-words-in-monolingual-corpus.py -d word_clusters_cs+de -l "en:" -i $corpus_en -o corpus_en.langprefix -od word_clusters_cs+de+en
  python $wammar_utils/replace-words-in-monolingual-corpus.py -d word_clusters_cs+de+en -l "es:" -i $corpus_es -o corpus_es.langprefix -od word_clusters_cs+de+en+es
  python $wammar_utils/replace-words-in-monolingual-corpus.py -d word_clusters_cs+de+en+es -l "fi:" -i $corpus_fi -o corpus_fi.langprefix -od word_clusters_cs+de+en+es+fi
  python $wammar_utils/replace-words-in-monolingual-corpus.py -d word_clusters_cs+de+en+es+fi -l "fr:" -i $corpus_fr -o corpus_fr.langprefix -od word_clusters_cs+de+en+es+fi+fr
  python $wammar_utils/replace-words-in-monolingual-corpus.py -d word_clusters_cs+de+en+es+fi+fr -l "ga:" -i $corpus_ga -o corpus_ga.langprefix -od word_clusters_cs+de+en+es+fi+fr+ga
  python $wammar_utils/replace-words-in-monolingual-corpus.py -d word_clusters_cs+de+en+es+fi+fr+ga -l "hu:" -i $corpus_hu -o corpus_hu.langprefix -od word_clusters_cs+de+en+es+fi+fr+ga+hu
  python $wammar_utils/replace-words-in-monolingual-corpus.py -d word_clusters_cs+de+en+es+fi+fr+ga+hu -l "it:" -i $corpus_it -o corpus_it.langprefix -od word_clusters_cs+de+en+es+fi+fr+ga+hu+it
  python $wammar_utils/replace-words-in-monolingual-corpus.py -d word_clusters_cs+de+en+es+fi+fr+ga+hu+it -l "sv:" -i $corpus_sv -o corpus_sv.langprefix -od augmented_word_clusters
  cat corpus_cs.langprefix corpus_de.langprefix corpus_en.langprefix corpus_es.langprefix corpus_fi.langprefix corpus_fr.langprefix corpus_ga.langprefix corpus_hu.langprefix corpus_it.langprefix corpus_sv.langprefix > corpus.langprefix

  # train cluster embeddings for all languages
  $incremental_word2vec/word2vec -train corpus.langprefix -min-count 40 -output cluster_embeddings -threads 32 -size $pretrained_dim -iter 20

}

# combine all bilingual dictionaries made available by the tasks MakeBilingualDictionaries*
# and compute the transitive closure of translationally equivalent words across languages. 
# For example, if both "dog ||| chien", "dog ||| male" and "dog ||| hund" appear in *.en-fr,
# in *.en-fr, and in *en-de, the resulting cluster may look like:
# en:dog_|_fr:chien_|_fr:male_|_de:hund, which is written in four separate lines in the output
# file as follows:
# en:dog ||| en:dog_|_fr:chien_|_fr:male_|_de:hund
# fr:chien ||| en:dog_|_fr:chien_|_fr:male_|_de:hund
# fr:male ||| en:dog_|_fr:chien_|_fr:male_|_de:hund
# de:hund ||| en:dog_|_fr:chien_|_fr:male_|_de:hund
# Everything should be lowercased in the output file. To avoid catastorphically large closures,
# we filter out some items from the bilingual dictionaries.
# arrow #4, #5, #18
task MakeMultilingualWordClusters
   < wiktionary_bi_dicts=$wiktionary_bi_dicts@MakeBilingualDictionariesFromWiktionary
   < parallel_bi_dicts=$parallel_bi_dicts@MakeBilingualDictionariesFromBitext
   < jiangacl15_bi_dicts=$jiangacl15_bi_dicts@MakeBilingualDictionariesFromJiangAcl15
  :: wammar_utils=@
#   > punctuation_clusters
   > word_clusters
{
  python $wammar_utils/map-words-to-transitive-closures.py -i $wiktionary_bi_dicts/wiktionary.* $parallel_bi_dicts/parallel.* $jiangacl15_bi_dicts/jiangacl15.* -o word_clusters
#  python $wammar_utils/generate-crosslingual-punctuation-mappings.py -o $punctuation_clusters
#  cat $punctuation_clusters word_clusters_without_punc > $word_clusters
#  rm word_clusters_without_punc
}

# read the wiktionary translations dump, and extract word pair translations for each language pair
# (there's a total of 10! language pairs), and write a bilingual dictionary file for each language pair.
# the output directory wiktionary_bi_dicts should contain files titled similar to wiktionary.en-fr which
# contain lines such as "dog ||| chien". Bilingual dictionaries should be lowercased.
# arrow #1
task MakeBilingualDictionariesFromWiktionary
  :: wiktionary_dump=@
  :: wammar_utils=@
  >  wiktionary_bi_dicts
{
  python $wammar_utils/wiktionary-multilingual-to-bilingual-dictionaries.py -rawfile $wiktionary_dump -outdir $wiktionary_bi_dicts -langs "czech|german|english|spanish|finnish|french|irish|hungarian|italian|swedish" -lowercase
}

# Inputs are the alignment files Jiang Guo used to induce word embeddings. This is nice because we can compare 
# his multilingual embeddings based on the same word pairs. This task just does some filtering and reformats
# the bilingual dictionary files to be consistent with other sources of bilingual dictionaries. The output file
# jiangacl15.fr-en should contain "chien ||| dog"
# arrow #19
task MakeBilingualDictionariesFromJiangAcl15
  :: jiangacl15_alignments_de_en=@
  :: jiangacl15_alignments_es_en=@
  :: jiangacl15_alignments_fr_en=@
  :: wammar_utils=@
   > jiangacl15_bi_dicts
{
  mkdir $jiangacl15_bi_dicts
  python $wammar_utils/convert-to-one-target-per-line.py -i $jiangacl15_alignments_de_en -o $jiangacl15_bi_dicts/jiangacl15.de-en
  python $wammar_utils/convert-to-one-target-per-line.py -i $jiangacl15_alignments_es_en -o $jiangacl15_bi_dicts/jiangacl15.es-en
  python $wammar_utils/convert-to-one-target-per-line.py -i $jiangacl15_alignments_fr_en -o $jiangacl15_bi_dicts/jiangacl15.fr-en
}

# read in parallel data (tokenized and lowercased). Process the dump files necessary for extracting
# wikipedia parallel titles, and append them to the parallel data. Then align the aggregate 
# (parallel+wikipedia) to obtain high quality word pair translations. 
# The output file parallel_bi_dicts/parallel.en-fi should contain lines that look like "dog ||| chien"
# arrow #6 and #2
task MakeBilingualDictionariesFromBitext
  :: bitext_cs_en=@
  :: bitext_de_en=@
  :: bitext_es_en=@
  :: bitext_fi_en=@
  :: bitext_fr_en=@
  :: bitext_ga_en=@
  :: bitext_hu_en=@
  :: bitext_it_en=@
  :: bitext_sv_en=@
  :: wikipedia_parallel_titles=@
  :: wikipedia_base_cs=@
  :: wikipedia_base_de=@
  :: wikipedia_base_es=@
  :: wikipedia_base_fi=@
  :: wikipedia_base_fr=@
  :: wikipedia_base_ga=@
  :: wikipedia_base_hu=@
  :: wikipedia_base_it=@
  :: wikipedia_base_sv=@
  :: wammar_utils=@
  :: fast_align=@
  :: cdec=@
   > parallel_bi_dicts
{
  mkdir $parallel_bi_dicts
  
  # extract parallel titles from wikipedia
  $wikipedia_parallel_titles/build-corpus.sh en $wikipedia_base_cs > wikipedia_titles.cs-en
  $wikipedia_parallel_titles/build-corpus.sh en $wikipedia_base_de > wikipedia_titles.de-en
  $wikipedia_parallel_titles/build-corpus.sh en $wikipedia_base_es > wikipedia_titles.es-en
  $wikipedia_parallel_titles/build-corpus.sh en $wikipedia_base_fi > wikipedia_titles.fi-en
  $wikipedia_parallel_titles/build-corpus.sh en $wikipedia_base_fr > wikipedia_titles.fr-en
  $wikipedia_parallel_titles/build-corpus.sh en $wikipedia_base_ga > wikipedia_titles.ga-en
  $wikipedia_parallel_titles/build-corpus.sh en $wikipedia_base_hu > wikipedia_titles.hu-en
  $wikipedia_parallel_titles/build-corpus.sh en $wikipedia_base_it > wikipedia_titles.it-en
  $wikipedia_parallel_titles/build-corpus.sh en $wikipedia_base_sv > wikipedia_titles.sv-en

  # tokenize and lowercase
  python $cdec/corpus/tokenize-parallel.py wikipedia_titles.cs-en >wikipedia_titles.tok.cs-en
  python $wammar_utils/lowercase.py -i wikipedia_titles.tok.cs-en -o wikipedia_titles.tok.lc.cs-en
  python $cdec/corpus/tokenize-parallel.py wikipedia_titles.de-en >wikipedia_titles.tok.de-en
  python $wammar_utils/lowercase.py -i wikipedia_titles.tok.de-en -o wikipedia_titles.tok.lc.de-en
  python $cdec/corpus/tokenize-parallel.py wikipedia_titles.es-en >wikipedia_titles.tok.es-en
  python $wammar_utils/lowercase.py -i wikipedia_titles.tok.es-en -o wikipedia_titles.tok.lc.es-en
  python $cdec/corpus/tokenize-parallel.py wikipedia_titles.fi-en >wikipedia_titles.tok.fi-en
  python $wammar_utils/lowercase.py -i wikipedia_titles.tok.fi-en -o wikipedia_titles.tok.lc.fi-en
  python $cdec/corpus/tokenize-parallel.py wikipedia_titles.fr-en >wikipedia_titles.tok.fr-en
  python $wammar_utils/lowercase.py -i wikipedia_titles.tok.fr-en -o wikipedia_titles.tok.lc.fr-en
  python $cdec/corpus/tokenize-parallel.py wikipedia_titles.ga-en >wikipedia_titles.tok.ga-en
  python $wammar_utils/lowercase.py -i wikipedia_titles.tok.ga-en -o wikipedia_titles.tok.lc.ga-en
  python $cdec/corpus/tokenize-parallel.py wikipedia_titles.hu-en >wikipedia_titles.tok.hu-en
  python $wammar_utils/lowercase.py -i wikipedia_titles.tok.hu-en -o wikipedia_titles.tok.lc.hu-en
  python $cdec/corpus/tokenize-parallel.py wikipedia_titles.it-en >wikipedia_titles.tok.it-en
  python $wammar_utils/lowercase.py -i wikipedia_titles.tok.it-en -o wikipedia_titles.tok.lc.it-en
  python $cdec/corpus/tokenize-parallel.py wikipedia_titles.sv-en >wikipedia_titles.tok.sv-en
  python $wammar_utils/lowercase.py -i wikipedia_titles.tok.sv-en -o wikipedia_titles.tok.lc.sv-en

  # aggregate parallel data of each language
  cat wikipedia_titles.tok.lc.cs-en $bitext_cs_en > aggregate.cs-en
  cat wikipedia_titles.tok.lc.de-en $bitext_de_en > aggregate.de-en
  cat wikipedia_titles.tok.lc.es-en $bitext_es_en > aggregate.es-en
  cat wikipedia_titles.tok.lc.fi-en $bitext_fi_en > aggregate.fi-en
  cat wikipedia_titles.tok.lc.fr-en $bitext_fr_en > aggregate.fr-en
  cat wikipedia_titles.tok.lc.ga-en $bitext_ga_en > aggregate.ga-en
  cat wikipedia_titles.tok.lc.hu-en $bitext_hu_en > aggregate.hu-en
  cat wikipedia_titles.tok.lc.it-en $bitext_it_en > aggregate.it-en
  cat wikipedia_titles.tok.lc.sv-en $bitext_sv_en > aggregate.sv-en

  # filter then align the aggregate parallel data, and filter parameters
  $cdec/corpus/filter-length.pl aggregate.cs-en > aggregate.filtered.cs-en
  $fast_align/fast_align -i aggregate.filtered.cs-en -v -d -o -c params.cs-en >/dev/null
  $fast_align/fast_align -i aggregate.filtered.cs-en -v -d -o -r -c params.en-cs >/dev/null
  python $wammar_utils/filter-word-alignment-parameters.py -t 0.05 -if params.cs-en -ir params.en-cs -d $parallel_bi_dicts/parallel.cs-en

  $cdec/corpus/filter-length.pl aggregate.de-en > aggregate.filtered.de-en
  $fast_align/fast_align -i aggregate.filtered.de-en -v -d -o -c params.de-en >/dev/null
  $fast_align/fast_align -i aggregate.filtered.de-en -v -d -o -r -c params.en-de >/dev/null
  python $wammar_utils/filter-word-alignment-parameters.py -t 0.05 -if params.de-en -ir params.en-de -d $parallel_bi_dicts/parallel.de-en

  $cdec/corpus/filter-length.pl aggregate.es-en > aggregate.filtered.es-en
  $fast_align/fast_align -i aggregate.filtered.es-en -v -d -o -c params.es-en >/dev/null
  $fast_align/fast_align -i aggregate.filtered.es-en -v -d -o -r -c params.en-es >/dev/null
  python $wammar_utils/filter-word-alignment-parameters.py -t 0.05 -if params.es-en -ir params.en-es -d $parallel_bi_dicts/parallel.es-en

  $cdec/corpus/filter-length.pl aggregate.fi-en > aggregate.filtered.fi-en
  $fast_align/fast_align -i aggregate.filtered.fi-en -v -d -o -c params.fi-en >/dev/null
  $fast_align/fast_align -i aggregate.filtered.fi-en -v -d -o -r -c params.en-fi >/dev/null
  python $wammar_utils/filter-word-alignment-parameters.py -t 0.05 -if params.fi-en -ir params.en-fi -d $parallel_bi_dicts/parallel.fi-en

  $cdec/corpus/filter-length.pl aggregate.fr-en > aggregate.filtered.fr-en
  $fast_align/fast_align -i aggregate.filtered.fr-en -v -d -o -c params.fr-en >/dev/null
  $fast_align/fast_align -i aggregate.filtered.fr-en -v -d -o -r -c params.en-fr >/dev/null
  python $wammar_utils/filter-word-alignment-parameters.py -t 0.05 -if params.fr-en -ir params.en-fr -d $parallel_bi_dicts/parallel.fr-en

  $cdec/corpus/filter-length.pl aggregate.ga-en > aggregate.filtered.ga-en
  $fast_align/fast_align -i aggregate.filtered.ga-en -v -d -o -c params.ga-en >/dev/null
  $fast_align/fast_align -i aggregate.filtered.ga-en -v -d -o -r -c params.en-ga >/dev/null
  python $wammar_utils/filter-word-alignment-parameters.py -t 0.05 -if params.ga-en -ir params.en-ga -d $parallel_bi_dicts/parallel.ga-en

  $cdec/corpus/filter-length.pl aggregate.hu-en > aggregate.filtered.hu-en
  $fast_align/fast_align -i aggregate.filtered.hu-en -v -d -o -c params.hu-en >/dev/null
  $fast_align/fast_align -i aggregate.filtered.hu-en -v -d -o -r -c params.en-hu >/dev/null
  python $wammar_utils/filter-word-alignment-parameters.py -t 0.05 -if params.hu-en -ir params.en-hu -d $parallel_bi_dicts/parallel.hu-en

  $cdec/corpus/filter-length.pl aggregate.it-en > aggregate.filtered.it-en
  $fast_align/fast_align -i aggregate.filtered.it-en -v -d -o -c params.it-en >/dev/null
  $fast_align/fast_align -i aggregate.filtered.it-en -v -d -o -r -c params.en-it >/dev/null
  python $wammar_utils/filter-word-alignment-parameters.py -t 0.05 -if params.it-en -ir params.en-it -d $parallel_bi_dicts/parallel.it-en

  $cdec/corpus/filter-length.pl aggregate.sv-en > aggregate.filtered.sv-en
  $fast_align/fast_align -i aggregate.filtered.sv-en -v -d -o -c params.sv-en >/dev/null
  $fast_align/fast_align -i aggregate.filtered.sv-en -v -d -o -r -c params.en-sv >/dev/null
  python $wammar_utils/filter-word-alignment-parameters.py -t 0.05 -if params.sv-en -ir params.en-sv -d $parallel_bi_dicts/parallel.sv-en

  # remove intermediate files
  rm aggregate.filtered.cs-en aggregate.cs-en wikipedia_titles.tok.lc.cs-en wikipedia_titles.tok.cs-en wikipedia_titles.cs-en params.cs-en params.en-cs
  rm aggregate.filtered.de-en aggregate.de-en wikipedia_titles.tok.lc.de-en wikipedia_titles.tok.de-en wikipedia_titles.de-en params.de-en params.en-de
  rm aggregate.filtered.es-en aggregate.es-en wikipedia_titles.tok.lc.es-en wikipedia_titles.tok.es-en wikipedia_titles.es-en params.es-en params.en-es
  rm aggregate.filtered.fi-en aggregate.fi-en wikipedia_titles.tok.lc.fi-en wikipedia_titles.tok.fi-en wikipedia_titles.fi-en params.fi-en params.en-fi
  rm aggregate.filtered.fr-en aggregate.fr-en wikipedia_titles.tok.lc.fr-en wikipedia_titles.tok.fr-en wikipedia_titles.fr-en params.fr-en params.en-fr
  rm aggregate.filtered.ga-en aggregate.ga-en wikipedia_titles.tok.lc.ga-en wikipedia_titles.tok.ga-en wikipedia_titles.ga-en params.ga-en params.en-ga
  rm aggregate.filtered.hu-en aggregate.hu-en wikipedia_titles.tok.lc.hu-en wikipedia_titles.tok.hu-en wikipedia_titles.hu-en params.hu-en params.en-hu
  rm aggregate.filtered.it-en aggregate.it-en wikipedia_titles.tok.lc.it-en wikipedia_titles.tok.it-en wikipedia_titles.it-en params.it-en params.en-it
  rm aggregate.filtered.sv-en aggregate.sv-en wikipedia_titles.tok.lc.sv-en wikipedia_titles.tok.sv-en wikipedia_titles.sv-en params.sv-en params.en-sv
}

// TODO(wammar): speed-up the character-level embeddings thing
// TODO(wammar): consider using adagrad instead of sgd
#include <cstdlib>
#include <algorithm>
#include <sstream>
#include <iostream>
#include <vector>
#include <limits>
#include <cmath>
#include <chrono>
#include <ctime>

#include <unordered_map>
#include <unordered_set>

#include <execinfo.h>
#include <unistd.h>
#include <signal.h>
#include <stdlib.h>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/program_options.hpp>

#include "cnn/training.h"
#include "cnn/cnn.h"
#include "cnn/expr.h"
#include "cnn/nodes.h"
#include "cnn/lstm.h"
#include "cnn/rnn.h"
#include "c2.h"

cpyp::Corpus corpus;

enum TypologyMode {
  // the typology embedding is linearly combined with the stack, buffer and action stacks,
  // all together feeding into a nonlinear layer that represents parser state
  TYPOLOGY_MODE_LINEAR = 0,

  // the traditional parser state is first computed then is linearly combined with the typology
  // embedding via another nonlinear layer to obtain a language-specific parser state
  TYPOLOGY_MODE_CASCADE = 1,

  // the traditional parser state is further hadamard-multiplied by a linear transformation of
  // the typology embedding
  TYPOLOGY_MODE_HADAMARD = 2,

  // the typology embedding is appended to all traditional stack LSTM inputs and, 
  // similarly to TYPOLOGY_MODE_LINEAR, is also linearly combined with the stack, 
  // buffer and action stacks to compute the parser state.
  TYPOLOGY_MODE_LINEAR_LEXICAL = 10,

  // the typology embedding is appended to all traditional stack LSTM inputs and, 
  // similarly to TYPOLOGY_MODE_CASCADE, is also linearly combined with the traditional
  // parser state, producing a language-specific parser state
  TYPOLOGY_MODE_CASCADE_LEXICAL = 11,

  // the typology embedding is appended to all traditional stack LSTM inputs and, 
  // similarly to TYPOLOGY_MODE_HADAMARD, is also hadamard-multiplied by the traditional
  // parser state
  TYPOLOGY_MODE_HADAMARD_LEXICAL = 12,

  // reserve no parameters for individual languages
  TYPOLOGY_MODE_NONE = 100
};

// OOV counters
unsigned brown_non_oov_count = 0, brown_oov_count = 0, brown2_non_oov_count = 0, brown2_oov_count = 0, pretrained_oov_count = 0, pretrained_non_oov_count = 0, learned_non_oov_count = 0, learned_oov_count = 0;

volatile bool requested_stop = false;
unsigned EPOCHS = 0;
unsigned LAYERS = 2;
unsigned INPUT_DIM = 40;
unsigned HIDDEN_DIM = 60;
unsigned ACTION_DIM = 36;
unsigned PRETRAINED_DIM = 50;
unsigned LSTM_INPUT_DIM = 60;
unsigned POS_DIM = 10;
unsigned REL_DIM = 8;
unsigned BROWN_DIM = 20;
unsigned BROWN2_DIM = 100;
unsigned OBSERVED_TYPOLOGY_DIM = 0;
unsigned COMPRESSED_TYPOLOGY_DIM = 10;
unsigned TYPOLOGY_MODE = TYPOLOGY_MODE_NONE;

// This is the size of the concatenated forward+backward character LSTM embeddings (must be even).
unsigned LSTM_CHAR_OUTPUT_DIM = 2 * 50;
bool USE_SPELLING = false;

bool USE_POS = false;

double DROPOUT = 0.0;
double BLOCK_DROPOUT_WORD_EMBEDDING = 0.0;
double BLOCK_DROPOUT_SPELL_EMBEDDING = 0.0;
double BLOCK_DROPOUT_PRETRAINED_EMBEDDING = 0.0;
double BLOCK_DROPOUT_FINE_POS_EMBEDDING = 0.0;
double DROPOUT_FINE_POS_EMBEDDING = 0.0;
double BLOCK_DROPOUT_TYPOLOGY_EMBEDDING = 0.0;

constexpr const char* ROOT_SYMBOL = "ROOT";
constexpr const char* UNK_BROWN = "UNK_BROWN";
constexpr const unsigned kUNK_BROWN = 0;
unsigned kROOT_SYMBOL = 0;
unsigned kUNK_SYMBOL = 0;
unsigned kROOT_CHAR_SYMBOL = 0;
unsigned kUNK_CHAR_SYMBOL = 0;
unsigned ACTION_SIZE = 0;
unsigned VOCAB_SIZE = 0;
unsigned CHAR_SIZE = 0; 
unsigned POS_SIZE = 0;
unsigned BROWN_CLUSTERS_COUNT = 0;
unsigned BROWN2_CLUSTERS_COUNT = 0;
string SCORE_FILENAME = "";

using namespace cnn::expr;
using namespace cnn;
using namespace std;
using namespace cpyp;
namespace po = boost::program_options;

vector<unsigned> possible_actions;

void InitCommandLine(int argc, char** argv, po::variables_map* conf) {
  po::options_description opts("Configuration options");
  opts.add_options()
    // TREEBANKS
    ("training_data,T", po::value<string>(), "List of Transitions - Training corpus")
    ("dev_data,d", po::value<string>(), "Development corpus")
    // MULTILINGUAL PARSING
    ("brown_clusters", po::value<string>(), "The paths file produced by Percy Liang's brown_clusters tool.")
    ("brown2_clusters", po::value<string>(), "The paths file produced by Percy Liang's brown_clusters tool.")
    ("typological_properties,y", po::value<string>(), "load typological properties of various "
     "languages specified in this file. Each line represents one language with space delimited "
     "fields. The first field is the 2-letter ISO name of a language. Each of the remaininng "
     "fieldsis the numeric value of a particular typological property such as 'SVO'. All lines "
     "have the same number of columns. As a result of specifying this, the representation of each "
     "word is augmented with an embedding of the language to which the word belongs. The "
     "two-letter prefix of the surface form identifies the language (e.g., 'en:good' and 'fr:bon').")
    ("typology_mode", po::value<unsigned>()->default_value(10), "This option determines how "
     "typological embeddings are used (values: 0, 1, 2, 10, 11, 12). This option is only valid "
     "when --typological_properties is also specified.")
    // NETWORK ARCHITECTURE PARAMETERS
    ("layers", po::value<unsigned>()->default_value(2), "number of LSTM layers")
    ("action_dim", po::value<unsigned>()->default_value(16), "action embedding size")
    ("input_dim", po::value<unsigned>()->default_value(32), "input embedding size")
    ("hidden_dim", po::value<unsigned>()->default_value(64), "hidden dimension")
    ("pretrained_dim", po::value<unsigned>()->default_value(50), "pretrained input dimension")
    ("pos_dim", po::value<unsigned>()->default_value(12), "POS dimension")
    ("rel_dim", po::value<unsigned>()->default_value(10), "relation dimension")
    ("lstm_input_dim", po::value<unsigned>()->default_value(60), "LSTM input dimension")
    // TRAINING VS. DECODING PARAMETERS
    ("train,t", "Should training be run?")
    ("parsing_model,m", po::value<string>(), "Load saved parsing_model from this file")
    ("server", "Should run the parser as a server which reads input sentences from STDIN and writes the predictiosn to STDOUT?")
    // DROPOUT PARAMETERS
    ("dropout", po::value<double>()->default_value(0.0),
     "dropout coefficient for individual elements in the token embedding, "
     "defaults at 0.0 (i.e., never dropout), must be in [0, 1]")
    ("block_dropout_word_embedding", po::value<double>()->default_value(0.0), 
     "dropout coefficient for the entire *learned* word embedding, "
     "defaults at 0.0 (i.e., never dropout), must be in [0, 1]")
    ("block_dropout_spell_embedding", po::value<double>()->default_value(0.0),
     "dropout coefficient for the entire (learned) character-based embedding, "
     "defaults at 0.0 (i.e., never dropout), must be in [0, 1]")
    ("block_dropout_pretrained_embedding", po::value<double>()->default_value(0.0),
     "dropout coefficient for the entire *pretrained* word embedding, "
     "defaults at 0.0 (i.e., never dropout), must be in [0, 1]")
    ("block_dropout_fine_pos_embedding", po::value<double>()->default_value(0.0),
     "dropout coefficient for the fine POS embedding, "
     "defaults at 0.0 (i.e., never dropout), must be in [0, 1]")
    ("dropout_fine_pos_embedding", po::value<double>()->default_value(0.0),
     "dropout coefficient for the fine POS embedding, "
     "defaults at 0.0 (i.e., never dropout), must be in [0, 1]")
    ("block_dropout_typology_embedding", po::value<double>()->default_value(0.0),
     "dropout coefficient for the entire typology embedding, "
     "defaults at 0.0 (i.e., never dropout), must be in [0, 1]")
    // OPTIONAL TOKEN-LEVEL REPRESENTATION PARAMETERS
    ("use_spelling,S", "Use spelling model")
    ("use_pos_tags,P", "make POS tags visible to parser")
    ("coarse_only", "Only use coarse POS tags. This option is only valid when use_pos_tags is set.")
    ("pretrained,w", po::value<string>(), "Pretrained word embeddings")
    // LEARNING PARAMETERS
    ("beam_size,b", po::value<unsigned>()->default_value(1), "beam size") // TODO(wammar): implement
    ("epochs", po::value<unsigned>()->default_value(0), "number of epochs used for training")
    ("unk_strategy,o", po::value<unsigned>()->default_value(1), "Unknown word strategy: \
      1 = singletons become UNK with probability unk_prob")
    ("unk_prob,u", po::value<double>()->default_value(0.2), "Probably with which to replace "
     "singletons with UNK in training data")
    ("score_file", po::value<string>(), "Write the parsing_model UAS score to this file")
    // HELP
    ("help,h", "Help");
  po::options_description dcmdline_options;
  dcmdline_options.add(opts);
  po::store(parse_command_line(argc, argv, dcmdline_options), *conf);
  if (conf->count("help")) {
    cerr << dcmdline_options << endl;
    exit(1);
  }
  if (conf->count("training_data") == 0) {
    cerr << "Please specify --traing_data (-T): this is required to determine the vocabulary mapping, even if the parser is used in prediction mode.\n";
    exit(1);
  }
}

struct ParserBuilder {
  LSTMBuilder tagging_forward_lstm, tagging_backward_lstm;
  LSTMBuilder stack_lstm; // (layers, input, hidden, trainer)
  LSTMBuilder buffer_lstm;
  LSTMBuilder action_lstm;
  LookupParameters* p_w = 0; // word embeddings
  LookupParameters* p_tagging_w = 0; // word embeddings (for tagger)
  LookupParameters* p_t = 0; // pretrained word embeddings (not updated)
  LookupParameters* p_observed_typology = 0; // typological properties (not updated)
  LookupParameters* p_a = 0; // input action embeddings
  LookupParameters* p_r = 0; // relation embeddings
  LookupParameters* p_p = 0; // pos tag embeddings
  LookupParameters* p_brown_embeddings = 0; // brown cluster embeddings
  LookupParameters* p_brown2_embeddings = 0; // brown2 cluster embeddings
  LookupParameters* p_tagging_brown_embeddings = 0; // brown cluster embeddings (tagger)
  LookupParameters* p_tagging_brown2_embeddings = 0; // brown2 cluster embeddings (tagger)
  Parameters* p_pbias = 0; // parser state bias
  Parameters* p_A = 0; // action lstm to parser state
  Parameters* p_B = 0; // buffer lstm to parser state
  Parameters* p_S = 0; // stack lstm to parser state
  Parameters* p_compressed_typology_to_parser_state = 0; // typology to parser state
  Parameters* p_H = 0; // head matrix for composition function
  Parameters* p_D = 0; // dependency matrix for composition function
  Parameters* p_R = 0; // relation matrix for composition function
  Parameters* p_word2l = 0; // lookup word embedding to LSTM input
  Parameters* p_tagging_word2l = 0; // lookup word embedding to the tagging bidirectional LSTM input
  Parameters* p_spell2l = 0; // character-based spell embedding to LSTM input
  Parameters* p_p2l = 0; // POS to LSTM input
  Parameters* p_coarse_p2l = 0; // coarse POS to LSTM input
  Parameters* p_pos2state = 0; // POS to parser state
  Parameters* p_t2l = 0; // pretrained word embeddings to LSTM input
  Parameters* p_tagging_t2l = 0; // pretrained word embeddings to the tagging bidirectional LSTM input
  Parameters* p_brown2l = 0; // Brown cluster embedding to LSTM input
  Parameters* p_brown22l = 0; // Brown2 cluster embedding to LSTM input
  Parameters* p_tagging_brown2l = 0; // Brown cluster embedding to LSTM input (tagger)
  Parameters* p_tagging_brown22l = 0; // Brown2 cluster embedding to LSTM input (tagger)
  Parameters* p_compressed_typology_to_lstm_input = 0; // compressed typology to LSTM input
  Parameters* p_tagging_compressed_typology_to_lstm_input = 0; // compressed typology to LSTM input (for the POS tagger)
  Parameters* p_observed_to_compressed_typology = 0; // from the binary observed typology vector to the hidden compressed typology embedding
  Parameters* p_ib = 0; // LSTM input bias
  Parameters* p_tagging_ib = 0; // LSTM input bias (for the tagger)
  Parameters* p_tagging_sos = 0; // token representation of the initial input to the forward LSTM used for tagging 
  Parameters* p_tagging_eos = 0; // token representation of the initial input to the backward LSTM used for tagging
  Parameters* p_cbias = 0; // composition function bias
  Parameters* p_p2a = 0;   // parser state to action
  Parameters* p_tagging_bi2pos = 0; // bidirectional LSTM to POS (tagger)
  Parameters* p_tagging_pos_bias = 0;  // pos bias (tagger)
  Parameters* p_action_start = 0;  // action bias
  Parameters* p_abias = 0;  // action bias
  Parameters* p_buffer_guard = 0;  // end of buffer
  Parameters* p_stack_guard = 0;  // end of stack

  Parameters* p_start_of_word = 0;// -->dummy <s> symbol
  Parameters* p_end_of_word = 0; // --> dummy </s> symbol
  LookupParameters* p_char_emb = 0; // --> mapping of characters to vectors 
  
  Parameters* p_pretrained_unk = 0;

  LSTMBuilder fw_char_lstm;
  LSTMBuilder bw_char_lstm;

  explicit ParserBuilder(Model* parsing_model) :
  tagging_forward_lstm(LAYERS, LSTM_INPUT_DIM, HIDDEN_DIM, parsing_model),
  tagging_backward_lstm(LAYERS, LSTM_INPUT_DIM, HIDDEN_DIM, parsing_model),
    stack_lstm(LAYERS, LSTM_INPUT_DIM, HIDDEN_DIM, parsing_model),
    buffer_lstm(LAYERS, LSTM_INPUT_DIM, HIDDEN_DIM, parsing_model),
    action_lstm(LAYERS, ACTION_DIM + COMPRESSED_TYPOLOGY_DIM, HIDDEN_DIM, parsing_model),
    p_w(parsing_model->add_lookup_parameters(VOCAB_SIZE, Dim({INPUT_DIM, 1}))),
  p_tagging_w(parsing_model->add_lookup_parameters(VOCAB_SIZE, Dim({INPUT_DIM, 1}))),
    p_a(parsing_model->add_lookup_parameters(ACTION_SIZE, Dim({ACTION_DIM, 1}))),
    p_r(parsing_model->add_lookup_parameters(ACTION_SIZE, Dim({REL_DIM, 1}))),
    p_pbias(parsing_model->add_parameters(Dim({HIDDEN_DIM, 1}))),
    p_A(parsing_model->add_parameters(Dim({HIDDEN_DIM, HIDDEN_DIM}))),
    p_B(parsing_model->add_parameters(Dim({HIDDEN_DIM, HIDDEN_DIM}))),
    p_S(parsing_model->add_parameters(Dim({HIDDEN_DIM, HIDDEN_DIM}))),
    p_H(parsing_model->add_parameters(Dim({LSTM_INPUT_DIM, LSTM_INPUT_DIM}))),
    p_D(parsing_model->add_parameters(Dim({LSTM_INPUT_DIM, LSTM_INPUT_DIM}))),
    p_R(parsing_model->add_parameters(Dim({LSTM_INPUT_DIM, REL_DIM}))),
    p_word2l(parsing_model->add_parameters(Dim({LSTM_INPUT_DIM, INPUT_DIM}))),
  p_tagging_word2l(parsing_model->add_parameters(Dim({LSTM_INPUT_DIM, INPUT_DIM}))),
    p_spell2l(parsing_model->add_parameters(Dim({LSTM_INPUT_DIM, LSTM_CHAR_OUTPUT_DIM}))),
    p_ib(parsing_model->add_parameters(Dim({LSTM_INPUT_DIM, 1}))),
  p_tagging_ib(parsing_model->add_parameters(Dim({LSTM_INPUT_DIM, 1}))),
  p_tagging_sos(parsing_model->add_parameters(Dim({LSTM_INPUT_DIM, 1}))),
  p_tagging_eos(parsing_model->add_parameters(Dim({LSTM_INPUT_DIM, 1}))),
    p_cbias(parsing_model->add_parameters(Dim({LSTM_INPUT_DIM, 1}))),
    p_p2a(parsing_model->add_parameters(Dim({ACTION_SIZE, HIDDEN_DIM}))),
  p_tagging_bi2pos(parsing_model->add_parameters(Dim({POS_SIZE, 2 * HIDDEN_DIM}))),
  p_tagging_pos_bias(parsing_model->add_parameters(Dim({POS_SIZE, 1}))),
    p_action_start(parsing_model->add_parameters(Dim({ACTION_DIM, 1}))),
    p_abias(parsing_model->add_parameters(Dim({ACTION_SIZE, 1}))),

    p_buffer_guard(parsing_model->add_parameters(Dim({LSTM_INPUT_DIM, 1}))),
    p_stack_guard(parsing_model->add_parameters(Dim({LSTM_INPUT_DIM, 1}))),

    p_start_of_word(parsing_model->add_parameters(Dim({LSTM_INPUT_DIM, 1}))),
    p_end_of_word(parsing_model->add_parameters(Dim({LSTM_INPUT_DIM, 1}))), 

    fw_char_lstm(LAYERS, LSTM_INPUT_DIM, LSTM_CHAR_OUTPUT_DIM/2, parsing_model),
    bw_char_lstm(LAYERS, LSTM_INPUT_DIM, LSTM_CHAR_OUTPUT_DIM/2, parsing_model) {
    
    if (USE_POS) {
      p_p = parsing_model->add_lookup_parameters(POS_SIZE, Dim({POS_DIM, 1}));
      p_p2l = parsing_model->add_parameters(Dim({LSTM_INPUT_DIM, POS_DIM}));
      p_coarse_p2l = parsing_model->add_parameters(Dim({LSTM_INPUT_DIM, POS_DIM}));

      // use pos of the stack back as input to parser state
      // TODO(wammar): clean this mess
      p_pos2state = nullptr; // parsing_model->add_parameters(Dim(HIDDEN_DIM, POS_DIM));
    }

    if (corpus.brown_clusters.size() > 0) {
      // parser
      p_brown_embeddings = parsing_model->add_lookup_parameters(BROWN_CLUSTERS_COUNT, Dim({BROWN_DIM, 1}));
      p_brown2l = parsing_model->add_parameters(Dim({LSTM_INPUT_DIM, BROWN_DIM}));
      // tagger
      p_tagging_brown_embeddings = parsing_model->add_lookup_parameters(BROWN_CLUSTERS_COUNT, Dim({BROWN_DIM, 1}));
      p_tagging_brown2l = parsing_model->add_parameters(Dim({LSTM_INPUT_DIM, BROWN_DIM}));
    }
  
    if (corpus.brown2_clusters.size() > 0) {
      // parser
      p_brown2_embeddings = parsing_model->add_lookup_parameters(BROWN2_CLUSTERS_COUNT, Dim({BROWN2_DIM, 1}));
      p_brown22l = parsing_model->add_parameters(Dim({LSTM_INPUT_DIM, BROWN2_DIM}));
      // tagger
      p_tagging_brown2_embeddings = parsing_model->add_lookup_parameters(BROWN2_CLUSTERS_COUNT, Dim({BROWN2_DIM, 1}));
      p_tagging_brown22l = parsing_model->add_parameters(Dim({LSTM_INPUT_DIM, BROWN2_DIM}));
    }

    if (corpus.pretrained.size() > 0) {
      p_t = parsing_model->add_lookup_parameters(VOCAB_SIZE, Dim({PRETRAINED_DIM, 1}));
      for (auto it : corpus.pretrained)
        p_t->Initialize(it.first, it.second);
      p_t2l = parsing_model->add_parameters(Dim({LSTM_INPUT_DIM, PRETRAINED_DIM}));
      p_tagging_t2l = parsing_model->add_parameters(Dim({LSTM_INPUT_DIM, PRETRAINED_DIM}));
      p_pretrained_unk = parsing_model->add_parameters(Dim({PRETRAINED_DIM, 1}));
    } else {
      p_t = nullptr;
      p_t2l = nullptr;
      p_tagging_t2l = nullptr;
    }
    
    p_char_emb = parsing_model->add_lookup_parameters(CHAR_SIZE, Dim({LSTM_INPUT_DIM, 1}));
    
    if (TYPOLOGY_MODE == TYPOLOGY_MODE_HADAMARD_LEXICAL ||
        TYPOLOGY_MODE == TYPOLOGY_MODE_LINEAR_LEXICAL ||
        TYPOLOGY_MODE == TYPOLOGY_MODE_CASCADE_LEXICAL) {
      p_compressed_typology_to_lstm_input = parsing_model->add_parameters(Dim({LSTM_INPUT_DIM, COMPRESSED_TYPOLOGY_DIM}));
      p_tagging_compressed_typology_to_lstm_input = parsing_model->add_parameters(Dim({LSTM_INPUT_DIM, COMPRESSED_TYPOLOGY_DIM}));
    }
    
    // load typological properties (p_observed_typology) 
    // and create typology to parser state parameters (p_compressed_typology_to_parser_state)
    if (corpus.typological_properties_map.size() > 0) {
      // one element is reserved for UNK.
      unsigned languages_count = corpus.typological_properties_map.size() + 1;
      p_observed_typology = parsing_model->add_lookup_parameters(languages_count, Dim({OBSERVED_TYPOLOGY_DIM, 1}));
      for (auto it : corpus.typological_properties_map) {
        p_observed_typology->Initialize(it.first, it.second);
      }
      
      p_compressed_typology_to_parser_state = parsing_model->add_parameters(Dim({HIDDEN_DIM, COMPRESSED_TYPOLOGY_DIM}));
      p_observed_to_compressed_typology = parsing_model->add_parameters(Dim({COMPRESSED_TYPOLOGY_DIM, OBSERVED_TYPOLOGY_DIM}));
      
      cerr << "p_observed_typology was " << p_observed_typology << endl << endl;
    } else {
      p_observed_typology = nullptr;
      p_compressed_typology_to_parser_state = nullptr;
      p_observed_to_compressed_typology = nullptr;

      cerr << "p_observed_typology was (null): " << p_observed_typology << endl << endl;
    }
  }

  static bool IsActionForbidden(const string& a, unsigned bsize, unsigned ssize, vector<int> stacki) {
    if (a[1]=='W' && ssize<3) return true;

    if (a[1]=='W') {

      int top=stacki[stacki.size()-1];
      int sec=stacki[stacki.size()-2];

      if (sec>top) return true;
    }

    bool is_shift = (a[0] == 'S' && a[1]=='H');
    bool is_reduce = !is_shift;
    if (is_shift && bsize == 1) return true;
    if (is_reduce && ssize < 3) return true;
    if (bsize == 2 && // ROOT is the only thing remaining on buffer
        ssize > 2 && // there is more than a single element on the stack
        is_shift) return true;
    // only attach left to ROOT
    if (bsize == 1 && ssize == 3 && a[0] == 'R') return true;
    return false;
  }

  static unordered_map<int,int> compute_heads(unsigned sent_len, const vector<unsigned>& actions, const vector<string>& setOfActions, unordered_map<int,string>* pr = nullptr) {
    unordered_map<int,int> heads;
    unordered_map<int,string> r;
    unordered_map<int,string>& rels = (pr ? *pr : r);
    for(unsigned i=0;i<sent_len;i++) { heads[i]=-1; rels[i]="ERROR"; }
    vector<int> bufferi(sent_len + 1, 0), stacki(1, -999);
    for (unsigned i = 0; i < sent_len; ++i)
      bufferi[sent_len - i] = i;
    bufferi[0] = -999;
    for (auto action: actions) { // loop over transitions for sentence
      const string& actionString=setOfActions[action];
      const char ac = actionString[0];
      const char ac2 = actionString[1];
      if (ac =='S' && ac2=='H') {  // SHIFT
        assert(bufferi.size() > 1); // dummy symbol means > 1 (not >= 1)
        stacki.push_back(bufferi.back());
        bufferi.pop_back();
      } 
      else if (ac=='S' && ac2=='W') {
        assert(stacki.size() > 2);

        //	std::cout<<"SWAP"<<"\n";
        unsigned ii = 0, jj = 0;
        jj=stacki.back();
        stacki.pop_back();

        ii=stacki.back();
        stacki.pop_back();

        bufferi.push_back(ii);

        stacki.push_back(jj);
      }

      else { // LEFT or RIGHT
        assert(stacki.size() > 2); // dummy symbol means > 2 (not >= 2)
        assert(ac == 'L' || ac == 'R');
        unsigned depi = 0, headi = 0;
        (ac == 'R' ? depi : headi) = stacki.back();
        stacki.pop_back();
        (ac == 'R' ? headi : depi) = stacki.back();
        stacki.pop_back();
        stacki.push_back(headi);
        heads[depi] = headi;
        rels[depi] = actionString;
      }
    }
    assert(bufferi.size() == 1);
    //assert(stacki.size() == 2);
    return heads;
  }


  // given the first character of a UTF8 block, find out how wide it is
  // see http://en.wikipedia.org/wiki/UTF-8 for more info
  inline unsigned int UTF8Len(unsigned char x) {
    if (x < 0x80) return 1;
    else if ((x >> 5) == 0x06) return 2;
    else if ((x >> 4) == 0x0e) return 3;
    else if ((x >> 3) == 0x1e) return 4;
    else if ((x >> 2) == 0x3e) return 5;
    else if ((x >> 1) == 0x7e) return 6;
    else return 0;
  }

  // if build_training_graph == false, this runs greedy decoding
  // the chosen tags are inserted into the "results" vector (in training just returns the reference)
  void log_prob_tagger(ComputationGraph& hg,
		       vector<TokenInfo>& sent,
		       const vector<unsigned>& pos_tag_set,
		       double *right,
		       bool build_training_graph) {
    // Initialize the bidirectional LSTM.
    tagging_forward_lstm.new_graph(hg);
    tagging_backward_lstm.new_graph(hg);
    tagging_forward_lstm.start_new_sequence();
    tagging_backward_lstm.start_new_sequence();

    // variables in the computation graph representing parameters
    Expression pretrained_unk;
    if (corpus.pretrained.size() > 0) {
      pretrained_unk = parameter(hg, p_pretrained_unk);
    }
    Expression compressed_typology_to_parser_state, observed_to_compressed_typology;
    if (corpus.typological_properties_map.size() > 0) {
      compressed_typology_to_parser_state = parameter(hg, p_compressed_typology_to_parser_state);
      observed_to_compressed_typology = parameter(hg, p_observed_to_compressed_typology);
    }
    Expression tagging_word2l = parameter(hg, p_tagging_word2l);
    Expression tagging_t2l;
    if (corpus.pretrained.size() > 0) {
      tagging_t2l = parameter(hg, p_tagging_t2l);
    }
    Expression tagging_compressed_typology_to_lstm_input;
    Expression compressed_typology;
    if (TYPOLOGY_MODE == TYPOLOGY_MODE_HADAMARD_LEXICAL ||
        TYPOLOGY_MODE == TYPOLOGY_MODE_LINEAR_LEXICAL ||
        TYPOLOGY_MODE == TYPOLOGY_MODE_CASCADE_LEXICAL) {
      tagging_compressed_typology_to_lstm_input = parameter(hg, p_tagging_compressed_typology_to_lstm_input);
      // .. also use the compressed typological embedding
      Expression observed_typology = const_lookup(hg, p_observed_typology, sent[0].lang_id);
      vector<float> observed_typology_vector = as_vector(hg.incremental_forward());
      compressed_typology = tanh(observed_to_compressed_typology * observed_typology);
      if (build_training_graph && BLOCK_DROPOUT_TYPOLOGY_EMBEDDING > 0.0) {
	compressed_typology = block_dropout(compressed_typology, BLOCK_DROPOUT_TYPOLOGY_EMBEDDING);
      }
    } else {
      vector<float> zeros(COMPRESSED_TYPOLOGY_DIM, 0.0);  // set x_values to change the inputs to the network
      compressed_typology = input(hg, {COMPRESSED_TYPOLOGY_DIM}, &zeros);
    }
    // layer between bidirecationl lstm and output layer
    Expression tagging_pos_bias = parameter(hg, p_tagging_pos_bias); // bias
    Expression tagging_bi2pos = parameter(hg, p_tagging_bi2pos); // weight matrix
    // bias for the token representation
    Expression tagging_ib = parameter(hg, p_tagging_ib);
    // start of sentence and end of sentence token representations
    Expression tagging_sos = parameter(hg, p_tagging_sos);
    Expression tagging_eos = parameter(hg, p_tagging_eos);
    // brown to input
    Expression tagging_brown2l;
    Expression tagging_brown22l;
    if (p_tagging_brown2l) { tagging_brown2l = parameter(hg, p_tagging_brown2l); }
    if (p_tagging_brown22l) { tagging_brown22l = parameter(hg, p_tagging_brown22l); }

    hg.incremental_forward();

    // compute token embeddings, which will be used as input to the bidirectional LSTM
    vector<Expression> token_embeddings(sent.size());
    for (unsigned i = 0; i < sent.size(); ++i) {
    
      // initialize the embedding of this token with bias parameters
      Expression i_i = tagging_ib;

      // .. add the (learned) word embedding
      Expression word_embedding;
      if (sent[i].training_oov) {
        // OOV words are replaced with a special UNK symbol.
        word_embedding = lookup(hg, p_tagging_w, kUNK_SYMBOL);
      } else {
        // lookup the learned embedding of a regular word.
        assert(sent[i].word_id < VOCAB_SIZE);
        word_embedding = lookup(hg, p_tagging_w, sent[i].word_id);
      }
      if (build_training_graph || BLOCK_DROPOUT_WORD_EMBEDDING == 0.0) {
        word_embedding = block_dropout(word_embedding, BLOCK_DROPOUT_WORD_EMBEDDING);
      } else if (!build_training_graph && BLOCK_DROPOUT_WORD_EMBEDDING == 1.0) {
        word_embedding = 0.0 * word_embedding;
      }
      i_i = affine_transform({i_i, tagging_word2l, word_embedding});

      // .. also use brown cluster embeddings
      if (corpus.brown_clusters.size() > 0) {
        // by default, assign the unkown brown cluster id, then update it if this word actually appears in the brown clusters file
        unsigned brown_cluster_id = corpus.brown_clusters[kUNK_BROWN];
        if (corpus.brown_clusters.count(sent[i].word_id) > 0) {
          brown_cluster_id = corpus.brown_clusters[sent[i].word_id];
        }
        // lookup the embedding of this brown cluster id
        Expression brown_embedding = lookup(hg, p_tagging_brown_embeddings, brown_cluster_id);
        i_i = affine_transform({i_i, tagging_brown2l, brown_embedding});
      }

      // .. also use brown2 cluster embeddings
      if (corpus.brown2_clusters.size() > 0) {
        // by default, assign the unkown brown cluster id, then update it if this word actually appears in the brown clusters file
        unsigned brown2_cluster_id = corpus.brown2_clusters[kUNK_BROWN];
        if (corpus.brown2_clusters.count(sent[i].word_id) > 0) {
          brown2_cluster_id = corpus.brown2_clusters[sent[i].word_id];
        }
        // lookup the embedding of this brown cluster id
        Expression brown2_embedding = lookup(hg, p_tagging_brown2_embeddings, brown2_cluster_id);
        i_i = affine_transform({i_i, tagging_brown22l, brown2_embedding});
      }

      // .. also use (pretrained) word embeddings if available
      if (p_t) {
        Expression t;
        if (corpus.pretrained.count(sent[i].word_id) != 0) {
          t = const_lookup(hg, p_t, sent[i].word_id);
        } else {
          t = pretrained_unk;
        }
        if (build_training_graph || BLOCK_DROPOUT_PRETRAINED_EMBEDDING == 0.0) {
          t = block_dropout(t, BLOCK_DROPOUT_PRETRAINED_EMBEDDING);
        } else if (!build_training_graph && BLOCK_DROPOUT_PRETRAINED_EMBEDDING == 1.0) {
          t = 0.0 * t;
        }
        i_i = affine_transform({i_i, tagging_t2l, t});
      }

      // .. also use the compressed typological embedding
      if (TYPOLOGY_MODE == TYPOLOGY_MODE_HADAMARD_LEXICAL ||
          TYPOLOGY_MODE == TYPOLOGY_MODE_LINEAR_LEXICAL ||
          TYPOLOGY_MODE == TYPOLOGY_MODE_CASCADE_LEXICAL) {
        Expression dropped_out_compressed_typology;
        if (build_training_graph || BLOCK_DROPOUT_TYPOLOGY_EMBEDDING == 0.0) {
          dropped_out_compressed_typology = block_dropout(compressed_typology, BLOCK_DROPOUT_TYPOLOGY_EMBEDDING);
        } else if (!build_training_graph && BLOCK_DROPOUT_TYPOLOGY_EMBEDDING == 1.0) {
          dropped_out_compressed_typology = 0.0 * compressed_typology;
        }
        i_i = affine_transform({i_i, tagging_compressed_typology_to_lstm_input, dropped_out_compressed_typology});
      }
      
      // .. nonlinearity
      i_i = tanh(i_i);

      // .. dropout at training time only
      if (build_training_graph) {
        i_i = dropout(i_i, DROPOUT);
      }

      // token representation is complete
      token_embeddings[i] = i_i;
    }

    // compute bidirectional lstm states
    tagging_forward_lstm.add_input(tagging_sos);
    tagging_backward_lstm.add_input(tagging_eos);
    vector<Expression> forward_lstm_outputs;
    vector<Expression> backward_lstm_outputs;
    for (unsigned i = 0; i < sent.size(); ++i) {
      // compute next forward output (represents token i)
      tagging_forward_lstm.add_input(token_embeddings[i]);
      forward_lstm_outputs.push_back(tagging_forward_lstm.back());
      // compute next backward output (represnets token |sent|-1-i)
      tagging_backward_lstm.add_input(token_embeddings[sent.size()-1-i]);
      backward_lstm_outputs.push_back(tagging_backward_lstm.back());
    }
    // reverse outputs of the backward lstm
    std::reverse(std::begin(backward_lstm_outputs), std::end(backward_lstm_outputs));
    
    // learn/decode next POS tag
    vector<Expression> negative_log_probs;
    for (unsigned i = 0; i < sent.size(); ++i) {

      // pos_scores = tagging_pos_bias + tagging_bi2pos * bi_lstm_output
      Expression bi_lstm_output = concatenate({forward_lstm_outputs[i], backward_lstm_outputs[i]});
      Expression pos_scores = affine_transform({tagging_pos_bias, tagging_bi2pos, bi_lstm_output});
      Expression pos_log_probs = log_softmax(pos_scores);
      vector<float> pos_log_probs_vector = as_vector(hg.incremental_forward());

      // predict
      unsigned most_likely_pos_id = 0;
      for (unsigned i = 1; i < pos_log_probs_vector.size(); ++i) {
	if (corpus.coarse_pos_vocab.count(i) == 0) { continue; }
	if (corpus.coarse_pos_vocab.count(most_likely_pos_id) == 0 || pos_log_probs_vector[i] > pos_log_probs_vector[most_likely_pos_id]) {
	  most_likely_pos_id = i;
	}
      }
      sent[i].predicted_coarse_pos_id = most_likely_pos_id;
      if (build_training_graph && most_likely_pos_id == sent[i].coarse_pos_id) { (*right)++; }

      // learn
      if (build_training_graph) {
	Expression negative_log_prob = -pick(pos_log_probs, most_likely_pos_id);
	negative_log_probs.push_back(negative_log_prob);
      }

      // report bug
      for (auto nonpositive : pos_log_probs_vector) { 
	if (nonpositive >= 0.01) { 
	  cerr << "ERROR: log_softmax = " << nonpositive << endl; 
	  assert(nonpositive <= 0.01); 
	} 
      }
      hg.incremental_forward();
    }

    // final cost for this sentence.
    Expression tot_neglogprob = sum(negative_log_probs);
    assert(tot_neglogprob.pg != nullptr);
    cerr << "tot_neglogprob = " << hg.incremental_forward() << endl;
  }

  // *** if correct_actions is empty, this runs greedy decoding ***
  // the chosen parse actions are inserted into the "results" vector (in training just returns the reference)
  void log_prob_parser(ComputationGraph& hg,
                       const vector<TokenInfo>& sent,
                       const vector<unsigned>& correct_actions,
                       const vector<string>& setOfActions,
                       double *right,
                       vector<unsigned> &results) {
    //for (unsigned i = 0; i < sent.size(); ++i) cerr << ' ' << corpus.intToWords.find(sent[i].word_id)->second;
    //cerr << endl;
  
    // Make sure the output vector is empty before inserting parse actions into it.
    assert(results.size() == 0);
    const bool build_training_graph = correct_actions.size() > 0;
  
    // initialize LSTMs for the stack, buffer, and actions.
    stack_lstm.new_graph(hg);
    buffer_lstm.new_graph(hg);
    action_lstm.new_graph(hg);
    stack_lstm.start_new_sequence();
    buffer_lstm.start_new_sequence();
    action_lstm.start_new_sequence();
    
    // variables in the computation graph representing the parameters
    Expression pbias = parameter(hg, p_pbias);
    Expression pretrained_unk;
    if (corpus.pretrained.size() > 0) {
      pretrained_unk = parameter(hg, p_pretrained_unk);
    }
    Expression H = parameter(hg, p_H);
    Expression D = parameter(hg, p_D);
    Expression R = parameter(hg, p_R);
    Expression cbias = parameter(hg, p_cbias);
    Expression S = parameter(hg, p_S);
    Expression B = parameter(hg, p_B);
    Expression A = parameter(hg, p_A);
    Expression compressed_typology_to_parser_state, observed_to_compressed_typology;
    if (corpus.typological_properties_map.size() > 0) {
      compressed_typology_to_parser_state = parameter(hg, p_compressed_typology_to_parser_state);
      observed_to_compressed_typology = parameter(hg, p_observed_to_compressed_typology);
    }
    Expression ib = parameter(hg, p_ib);
    Expression word2l = parameter(hg, p_word2l);
    Expression spell2l = parameter(hg, p_spell2l);
    Expression p2l, coarse_p2l;
    if (USE_POS) {
      p2l = parameter(hg, p_p2l);
      coarse_p2l = parameter(hg, p_coarse_p2l);
    }
    Expression t2l;
    if (p_t2l) { t2l = parameter(hg, p_t2l); }
    Expression brown2l;
    Expression brown22l;
    if (p_brown2l) { brown2l = parameter(hg, p_brown2l); }
    if (p_brown22l) { brown22l = parameter(hg, p_brown22l); }
    Expression compressed_typology_to_lstm_input;
    if (p_compressed_typology_to_lstm_input) { compressed_typology_to_lstm_input = parameter(hg, p_compressed_typology_to_lstm_input); }
    Expression p2a = parameter(hg, p_p2a);
    Expression abias = parameter(hg, p_abias);
    Expression action_start = parameter(hg, p_action_start);
    hg.incremental_forward();
    
    Expression compressed_typology;
    if (TYPOLOGY_MODE == TYPOLOGY_MODE_HADAMARD_LEXICAL ||
        TYPOLOGY_MODE == TYPOLOGY_MODE_LINEAR_LEXICAL ||
        TYPOLOGY_MODE == TYPOLOGY_MODE_CASCADE_LEXICAL) {
      // .. also use the compressed typological embedding
      Expression observed_typology = const_lookup(hg, p_observed_typology, sent[0].lang_id);
      vector<float> observed_typology_vector = as_vector(hg.incremental_forward());
      compressed_typology = tanh(observed_to_compressed_typology * observed_typology);
      if (build_training_graph && BLOCK_DROPOUT_TYPOLOGY_EMBEDDING > 0.0) {
	compressed_typology = block_dropout(compressed_typology, BLOCK_DROPOUT_TYPOLOGY_EMBEDDING);
      }
      // TODO(wammar): visualize the language embeddings here
      //vector<float> compressed_typology_vector = as_vector(hg.incremental_forward());
      //cerr << "compressed_typology_vector = "; 
      // for (auto& compressed_typology_element : compressed_typology_vector) {
      //   cerr << compressed_typology_element << " "; 
      //   cerr << endl;
      // }
    } else {
      vector<float> zeros(COMPRESSED_TYPOLOGY_DIM, 0.0);  // set x_values to change the inputs to the network
      compressed_typology = input(hg, {COMPRESSED_TYPOLOGY_DIM}, &zeros);
    }
    hg.incremental_forward();

    Expression action_input = concatenate({action_start, compressed_typology});
    hg.incremental_forward();
    action_lstm.add_input(action_input);
    hg.incremental_forward();

    // variables representing token embeddings (possibly including POS info)
    vector<Expression> buffer(sent.size() + 1);  
    // position of the words in the sentence
    vector<int> bufferi(sent.size() + 1);  
    // precompute buffer representation from left to right
    
    Expression word_end = parameter(hg, p_end_of_word);
    Expression word_start = parameter(hg, p_start_of_word); 
    
    if (USE_SPELLING) {
      fw_char_lstm.new_graph(hg);
      bw_char_lstm.new_graph(hg);
    }
          
    // Scan tokens of the sentence.
    for (unsigned i = 0; i < sent.size(); ++i) {

    
      // initialize the embedding of this token with bias parameters
      const TokenInfo& tokenInfo = sent[i];
      Expression i_i = ib;

      // .. add the (learned) word embedding
      Expression word_embedding;
      if (tokenInfo.training_oov) {
        // OOV words are replaced with a special UNK symbol.
        word_embedding = lookup(hg, p_w, kUNK_SYMBOL);
        learned_oov_count += 1;
      } else {
        // lookup the learned embedding of a regular word.
        assert(tokenInfo.word_id < VOCAB_SIZE);
        word_embedding = lookup(hg, p_w, tokenInfo.word_id);
        learned_non_oov_count += 1;
      }
      if (build_training_graph || BLOCK_DROPOUT_WORD_EMBEDDING == 0.0) {
        word_embedding = block_dropout(word_embedding, BLOCK_DROPOUT_WORD_EMBEDDING);
      } else if (!build_training_graph && BLOCK_DROPOUT_WORD_EMBEDDING == 1.0) {
        word_embedding = 0.0 * word_embedding;
      }
      i_i = affine_transform({i_i, word2l, word_embedding});

      // .. add the (learned) character-based spell embedding of this word
      if (USE_SPELLING) {
        Expression spell_embedding;
        
        // Get the surface form string.
        vector<unsigned> &char_ids = corpus.wordIntsToCharInts[tokenInfo.word_id];
        assert(char_ids.size() > 0);
        
        // encode this token using both left-to-right and right-to-left character LSTM
        fw_char_lstm.start_new_sequence();
        bw_char_lstm.start_new_sequence();
        fw_char_lstm.add_input(word_start);
        bw_char_lstm.add_input(word_end);
        unsigned sequence_length = char_ids.size();
        for (unsigned i = 0; i < sequence_length; ++i) {
          unsigned fw_char_id = char_ids[i];
	  // skip OOV characters (e.g., dev set is japanese while training set is english)
          if (corpus.training_char_vocab.count(fw_char_id) > 0) {
	    Expression fw_char_emb = lookup(hg, p_char_emb, fw_char_id);
	    fw_char_lstm.add_input(fw_char_emb);
	  }
          unsigned bw_char_id = char_ids[sequence_length - i - 1];
	  // skip OOV characters (e.g., dev set is japanese while training set is english)
	  if (corpus.training_char_vocab.count(bw_char_id) > 0) {
	    Expression bw_char_emb = lookup(hg, p_char_emb, bw_char_id);
	    bw_char_lstm.add_input(bw_char_emb);
	  }
	}
        fw_char_lstm.add_input(word_end);
        bw_char_lstm.add_input(word_start);
        Expression fw_i = fw_char_lstm.back();
        Expression bw_i = bw_char_lstm.back();

        // concatenate left-to-right and right-to-left character-based encoding
        vector<Expression> tt = {fw_i, bw_i};
        spell_embedding = concatenate(tt); //and this goes into the buffer...

        // use block dropout to stochastically zero out the spell embedding.
        if (build_training_graph || BLOCK_DROPOUT_SPELL_EMBEDDING == 0.0) {
          spell_embedding = block_dropout(spell_embedding, BLOCK_DROPOUT_SPELL_EMBEDDING);
        } else if (!build_training_graph && BLOCK_DROPOUT_SPELL_EMBEDDING == 1.0) {
          spell_embedding = 0.0 * spell_embedding;
        }
        
        // actually add the spell embedding
        i_i = affine_transform({i_i, spell2l, spell_embedding});
      }

      // .. also use (learned) POS embeddings
      if (USE_POS) {

	// use coarse pos embeddings if it was observed in training
	if (corpus.training_pos_vocab.count(tokenInfo.coarse_pos_id)) {
	  Expression coarse_p = lookup(hg, p_p, tokenInfo.coarse_pos_id);
	  i_i = affine_transform({i_i, coarse_p2l, coarse_p});
	}
	
	// use fine grained pos embeddings if it was observed in training
	if (corpus.training_pos_vocab.count(tokenInfo.pos_id)) {
	  // TODO[wammar]: check the block dropout implementation here
	  // Use block dropout with the fine grained POS tag so that the parser can make predictions even when they are missing
	  Expression p = lookup(hg, p_p, tokenInfo.pos_id);
	  if (tokenInfo.pos_id == 0) {
	    // if the fine grained POS tag is not specified, do not use it.
	    p = 0.0 * p;
	  }
	  if (build_training_graph || BLOCK_DROPOUT_FINE_POS_EMBEDDING == 0.0) {
	    p = block_dropout(p, BLOCK_DROPOUT_FINE_POS_EMBEDDING);
	  } else if (!build_training_graph && BLOCK_DROPOUT_FINE_POS_EMBEDDING == 1.0) {
	    p = 0.0 * p;
	  }
	  // Use dropout with the fine grained POS tag if specified
	  if (build_training_graph) {
	    p = dropout(p, DROPOUT_FINE_POS_EMBEDDING);
	  }
	  i_i = affine_transform({i_i, p2l, p});
	}
      }

      // .. also use brown cluster embeddings
      if (corpus.brown_clusters.size() > 0) {
        // by default, assign the unkown brown cluster id, then update it if this word actually appears in the brown clusters file
        unsigned brown_cluster_id = corpus.brown_clusters[kUNK_BROWN];
        if (corpus.brown_clusters.count(tokenInfo.word_id) > 0) {
          brown_cluster_id = corpus.brown_clusters[tokenInfo.word_id];
          brown_non_oov_count += 1;
        } else {
          brown_oov_count += 1;
        }
        //cerr << "tokenInfo.word_id = " << tokenInfo.word_id << ", brown_cluster_id=" << brown_cluster_id << " surface=\t" << corpus.intToWords[tokenInfo.word_id];
        // lookup the embedding of this brown cluster id
        Expression brown_embedding = lookup(hg, p_brown_embeddings, brown_cluster_id);
        i_i = affine_transform({i_i, brown2l, brown_embedding});
        //        cerr << "done." << endl;
      }

      // .. also use brown2 cluster embeddings
      if (corpus.brown2_clusters.size() > 0) {
        // by default, assign the unkown brown cluster id, then update it if this word actually appears in the brown clusters file
        unsigned brown2_cluster_id = corpus.brown2_clusters[kUNK_BROWN];
        if (corpus.brown2_clusters.count(tokenInfo.word_id) > 0) {
          brown2_cluster_id = corpus.brown2_clusters[tokenInfo.word_id];
          brown2_non_oov_count += 1;
        } else {
          brown2_oov_count += 1;
        }
        //cerr << "tokenInfo.word_id = " << tokenInfo.word_id << ", brown2_cluster_id=" << brown2_cluster_id << " surface=\t" << corpus.intToWords[tokenInfo.word_id];
        // lookup the embedding of this brown cluster id
        Expression brown2_embedding = lookup(hg, p_brown2_embeddings, brown2_cluster_id);
        i_i = affine_transform({i_i, brown22l, brown2_embedding});
        //        cerr << "done." << endl;
      }

      // .. also use (pretrained) word embeddings if available
      if (p_t) {
        Expression t;
        if (corpus.pretrained.count(tokenInfo.word_id) != 0) {
          t = const_lookup(hg, p_t, tokenInfo.word_id);
          pretrained_non_oov_count += 1;
        } else {
          t = pretrained_unk;
          pretrained_oov_count += 1;
        }
        if (build_training_graph || BLOCK_DROPOUT_PRETRAINED_EMBEDDING == 0.0) {
          t = block_dropout(t, BLOCK_DROPOUT_PRETRAINED_EMBEDDING);
        } else if (!build_training_graph && BLOCK_DROPOUT_PRETRAINED_EMBEDDING == 1.0) {
          t = 0.0 * t;
        }
        i_i = affine_transform({i_i, t2l, t});
      }

      // .. also use the compressed typological embedding
      if (TYPOLOGY_MODE == TYPOLOGY_MODE_HADAMARD_LEXICAL ||
          TYPOLOGY_MODE == TYPOLOGY_MODE_LINEAR_LEXICAL ||
          TYPOLOGY_MODE == TYPOLOGY_MODE_CASCADE_LEXICAL) {
        Expression dropped_out_compressed_typology;
        if (build_training_graph || BLOCK_DROPOUT_TYPOLOGY_EMBEDDING == 0.0) {
          dropped_out_compressed_typology = block_dropout(compressed_typology, BLOCK_DROPOUT_TYPOLOGY_EMBEDDING);
        } else if (!build_training_graph && BLOCK_DROPOUT_TYPOLOGY_EMBEDDING == 1.0) {
          dropped_out_compressed_typology = 0.0 * compressed_typology;
        }
        i_i = affine_transform({i_i, compressed_typology_to_lstm_input, dropped_out_compressed_typology});
      }
      
      // .. nonlinearity
      i_i = tanh(i_i);

      // .. dropout at training time only
      if (build_training_graph) {
        i_i = dropout(i_i, DROPOUT);
      }

      buffer[sent.size() - i] = i_i;
      bufferi[sent.size() - i] = i;

    }

    // dummy symbol to represent the empty buffer
    buffer[0] = parameter(hg, p_buffer_guard);
    bufferi[0] = -999;
    for (auto& b : buffer)
      buffer_lstm.add_input(b);
    
    vector<Expression> stack;  // variables representing subtree embeddings
    vector<int> stacki; // position of words in the sentence of head of subtree
    stack.push_back(parameter(hg, p_stack_guard));
    stacki.push_back(-999); // not used for anything
    // drive dummy symbol on stack through LSTM
    stack_lstm.add_input(stack.back());
    vector<Expression> negative_log_probs;
    unsigned action_count = 0;  // incremented at each prediction
    while(stack.size() > 2 || buffer.size() > 1) {

      // get list of possible actions for the current parser state
      vector<unsigned> current_valid_actions;
      for (auto a: possible_actions) {
        if (IsActionForbidden(setOfActions[a], buffer.size(), stack.size(), stacki))
          continue;
        current_valid_actions.push_back(a);
      }

      // p_t = pbias + S * slstm + B * blstm + A * almst + (Y * observed_typological_properties_vector)
      Expression p_t;
      if (p_observed_typology && 
          // bad condition.
          stacki.back() != -999 &&
          corpus.typological_properties_map.size() > 0 &&
          // bad condition.
          corpus.typological_properties_map.count(sent[stacki.back()].lang_id)) {
        Expression observed_typology = const_lookup(hg, p_observed_typology, sent[stacki.back()].lang_id);

        // TODO(wammar): clean this mess
        // TODO(wammar): try also adding POS tag of the next word on buffer and the next word in stack.
        //unsigned pos_id = sent[stacki.back()].pos_id;
        //Expression p = lookup(hg, p_p, pos_id);
        
        switch (TYPOLOGY_MODE) {
        case TYPOLOGY_MODE_LINEAR:
        case TYPOLOGY_MODE_LINEAR_LEXICAL:
          p_t = affine_transform({pbias, 
                // TODO(wammar): clean this mess
                //                pos2state, p,
                S, stack_lstm.back(), 
                B, buffer_lstm.back(), 
                A, action_lstm.back(),
                compressed_typology_to_parser_state, tanh(observed_to_compressed_typology * observed_typology)});
          break;
        case TYPOLOGY_MODE_CASCADE:
        case TYPOLOGY_MODE_CASCADE_LEXICAL:
          p_t = affine_transform({pbias, 
                // TODO(wammar): clean this mess
                //                pos2state, p
                S, stack_lstm.back(), 
                B, buffer_lstm.back(), 
                A, action_lstm.back()});
          p_t = tanh(p_t);
          p_t = affine_transform({p_t, 
                compressed_typology_to_parser_state, tanh(observed_to_compressed_typology * observed_typology)});
          break;
        case TYPOLOGY_MODE_HADAMARD:
        case TYPOLOGY_MODE_HADAMARD_LEXICAL:
          p_t = affine_transform({pbias, 
                // TODO(wammar): clean this mess
                //                pos2state, p
                S, stack_lstm.back(), 
                B, buffer_lstm.back(), 
                A, action_lstm.back()});
          p_t = tanh(p_t);
          p_t = 
            cwise_multiply(p_t, 
                           tanh(compressed_typology_to_parser_state * tanh(observed_to_compressed_typology * observed_typology)));
          break;
	case TYPOLOGY_MODE_NONE:
	  p_t = affine_transform({pbias, S, stack_lstm.back(), B, buffer_lstm.back(), A, action_lstm.back()});
	  break;
	default:
          assert(false);
          exit(1);
          break;
        }
        hg.incremental_forward();
	
      } else {
	p_t = affine_transform({pbias, S, stack_lstm.back(), B, buffer_lstm.back(), A, action_lstm.back()});

      }
      Expression nlp_t = tanh(p_t);
      // r_t = abias + p2a * nlp
      Expression r_t = affine_transform({abias, p2a, nlp_t});
      hg.incremental_forward();
      
      // TODO(wammar): clean this mess
      Expression adiste;
      vector<float> adist;
      
      // allow invalid actions!
      adiste = log_softmax(r_t);
      adist = as_vector(hg.incremental_forward());
      // disallow invalid actions!
      //adiste = log_softmax(r_t, current_valid_actions);        
      //adist = as_vector(hg.incremental_forward());
      
      // report bug
      for (auto nonpositive : adist) { 
	if (nonpositive >= 0.01) { 
	  cerr << "ERROR: log_softmax = " << nonpositive << endl; 
	  assert(nonpositive <= 0.01); 
	} 
      }
    
      unsigned best_a = current_valid_actions[0];
      double best_score = adist[best_a];
      for (unsigned i = 1; i < current_valid_actions.size(); ++i) {
        if (adist[current_valid_actions[i]] > best_score) {
          best_score = adist[current_valid_actions[i]];
          best_a = current_valid_actions[i];
        }
      }
      unsigned action = best_a;
      if (build_training_graph) {  // if we have reference actions (for training) use the reference action
        action = correct_actions[action_count];
        if (best_a == action) { (*right)++; }
      }
      ++action_count;
      // action_log_prob = pick(adist, action)
      
      Expression action_negative_log_prob = -pick(adiste, action);

      //cerr << "action_negative_log_prob = " << as_scalar(hg.incremental_forward()) << endl;
      hg.incremental_forward();
      negative_log_probs.push_back(action_negative_log_prob);
      results.push_back(action);

      // add current action to action LSTM
      Expression actione = lookup(hg, p_a, action);
      action_lstm.add_input(concatenate({actione, compressed_typology}));

      // get relation embedding from action (TODO: convert to relation from action?)
      Expression relation = lookup(hg, p_r, action);

      // do action
      const string& actionString=setOfActions[action];
      const char ac = actionString[0];
      const char ac2 = actionString[1];

      if (ac =='S' && ac2=='H') {  // SHIFT
        assert(buffer.size() > 1); // dummy symbol means > 1 (not >= 1)
        stack.push_back(buffer.back());
        stack_lstm.add_input(buffer.back());
        buffer.pop_back();
        buffer_lstm.rewind_one_step();
        stacki.push_back(bufferi.back());
        bufferi.pop_back();
      } 
      else if (ac=='S' && ac2=='W'){ //SWAP
        assert(stack.size() > 2); // dummy symbol means > 2 (not >= 2)

        //std::cout<<"SWAP: "<<"stack.size:"<<stack.size()<<"\n";

        Expression toki, tokj;
        unsigned ii = 0, jj = 0;
        tokj=stack.back();
        jj=stacki.back();
        stack.pop_back();
        stacki.pop_back();

        toki=stack.back();
        ii=stacki.back();
        stack.pop_back();
        stacki.pop_back();

        buffer.push_back(toki);
        bufferi.push_back(ii);

        stack_lstm.rewind_one_step();
        stack_lstm.rewind_one_step();


        buffer_lstm.add_input(buffer.back());

        stack.push_back(tokj);
        stacki.push_back(jj);

        stack_lstm.add_input(stack.back());

        //stack_lstm.rewind_one_step();
        //buffer_lstm.rewind_one_step();
      }
      else { // LEFT or RIGHT
        assert(stack.size() > 2); // dummy symbol means > 2 (not >= 2)
        assert(ac == 'L' || ac == 'R');
        Expression dep, head;
        unsigned depi = 0, headi = 0;
        (ac == 'R' ? dep : head) = stack.back();
        (ac == 'R' ? depi : headi) = stacki.back();
        stack.pop_back();
        stacki.pop_back();
        (ac == 'R' ? head : dep) = stack.back();
        (ac == 'R' ? headi : depi) = stacki.back();
        stack.pop_back();
        stacki.pop_back();
        // composed = cbias + H * head + D * dep + R * relation
        Expression composed = affine_transform({cbias, H, head, D, dep, R, relation});
        Expression nlcomposed = tanh(composed);
        stack_lstm.rewind_one_step();
        stack_lstm.rewind_one_step();
        stack_lstm.add_input(nlcomposed);
        stack.push_back(nlcomposed);
        stacki.push_back(headi);
      }
    }
    // TODO(wammar): continue copying for the tagger from here
    
    assert(stack.size() == 2); // guard symbol, root
    assert(stacki.size() == 2);
    assert(buffer.size() == 1); // guard symbol
    assert(bufferi.size() == 1);
    Expression tot_neglogprob = sum(negative_log_probs);
    //cerr << "tot_neglogprob = " << as_scalar(hg.incremental_forward()) << endl;
    hg.incremental_forward();
    assert(tot_neglogprob.pg != nullptr);
    assert(results.size() > 0);
  } // end of log_prob_parse(...)
};

void signal_callback_handler(int /* signum */) {
  if (requested_stop) {
    cerr << "\nReceived SIGINT again, quitting.\n";
    _exit(1);
  }
  cerr << "\nReceived SIGINT terminating optimization early...\n";
  requested_stop = true;
}

unsigned compute_correct(const unordered_map<int,int>& ref, const unordered_map<int,int>& hyp, unsigned len) {
  unsigned res = 0;
  for (unsigned i = 0; i < len; ++i) {
    auto ri = ref.find(i);
    auto hi = hyp.find(i);
    assert(ri != ref.end());
    assert(hi != hyp.end());
    if (ri->second == hi->second) ++res;
  }
  return res;
}

void output_conll(const vector<TokenInfo>& sentence,
                  const unordered_map<int,int>& hyp, 
		  const unordered_map<int,string>& rel_hyp) {
  for (unsigned i = 0; i < (sentence.size()-1); ++i) {
    auto index = i + 1;
    string wit = corpus.intToWords.at(sentence[i].word_id);
    string pit = corpus.intToPos.at(sentence[i].pos_id);
    assert(hyp.find(i) != hyp.end());
    auto hyp_head = hyp.find(i)->second + 1;
    if (hyp_head == (int)sentence.size()) hyp_head = 0;
    auto hyp_rel_it = rel_hyp.find(i);
    assert(hyp_rel_it != rel_hyp.end());
    auto hyp_rel = hyp_rel_it->second;
    size_t first_char_in_rel = hyp_rel.find('(') + 1;
    size_t last_char_in_rel = hyp_rel.rfind(')') - 1;
    hyp_rel = hyp_rel.substr(first_char_in_rel, last_char_in_rel - first_char_in_rel + 1);
    cout << index << '\t'       // 1. ID 
         << wit << '\t'         // 2. FORM
         << "_" << '\t'         // 3. LEMMA 
         << "_" << '\t'         // 4. CPOSTAG 
         << pit << '\t'         // 5. POSTAG
         << "_" << '\t'         // 6. FEATS
         << hyp_head << '\t'    // 7. HEAD
         << hyp_rel << '\t'     // 8. DEPREL
         << "_" << '\t'         // 9. PHEAD
         << "_" << endl;        // 10. PDEPREL
  }
  cout << endl;
}

void load_typological_properties_map(std::string& filename) {
  // If the command line spcifies a file with typological properties of each 
  // language of interest, load them.
  if (filename.size() != 0) {
    cerr << "Loading typological properties from " << filename << "\n";
    ifstream in(filename.c_str());
    string line;
    string language_2letter_iso;
    while (getline(in, line)) {
      istringstream lin(line);
      // comments start with #
      if (line[0] == '#') continue;
      // The first field is a language's 2-letter ISO name.
      lin >> language_2letter_iso;
      if (language_2letter_iso.size() != 2) {
        cerr << "ERROR: the language ids in the typological properties file must be of length two. "
             << "The following language identifier is not: " << language_2letter_iso << endl;
      }
      assert(language_2letter_iso.size() == 2);
      // While processing the first language, we don't know how many properties
      // there are.
      vector<float> v;
      if (OBSERVED_TYPOLOGY_DIM == 0) {
        vector<float> property_values;
        float buffer;
        // So we just keep reading property values into a vector.
        while (!lin.eof()) {
          lin >> buffer;
          property_values.push_back(buffer);
        }
        // Now we know the number of typological properties. All subsequent 
        // languages are expected to have the same number.
        OBSERVED_TYPOLOGY_DIM = property_values.size();
        assert(OBSERVED_TYPOLOGY_DIM > 0);
        // Copy the property values into the observed language vector v.
        v.resize(OBSERVED_TYPOLOGY_DIM);
        for (unsigned i = 0; i < OBSERVED_TYPOLOGY_DIM; ++i) {
          v[i] = property_values[i];
        }
      } else {
        // Read each field and fill the corresponding dimension in the observed
        // language vector v.
        v.resize(OBSERVED_TYPOLOGY_DIM);
        for (unsigned i = 0; i < OBSERVED_TYPOLOGY_DIM; ++i) {
          float buffer;
          lin >> buffer;
          v[i] = buffer;
        }
      }
      // Associate the vector of typological property values with this language.
      unsigned language_id = corpus.get_or_add_lang(language_2letter_iso);
      if (language_id == 0) {
        cerr << "ERROR: corpus.get_or_add_lang(" << language_2letter_iso << ") returns "
             << language_id << endl;
      }
      assert(language_id != 0);
      corpus.typological_properties_map[language_id] = v;
    }
  }
}

// each line in this file consists of three tab-delimited fields: brown cluster bitstring, word, and frequency.
void load_brown_clusters(std::string& filename) {
  // If the command line spcifies a file with a cluster map, load it.
  if (filename.size() == 0) {
    return;
  }
  cerr << "Loading the brown clusters from " << filename << "\n";
  ifstream in(filename.c_str());
  string line;
  string cluster_bitstring, word, frequency;
  unordered_map<string, int> cluster_bitstring_to_id;
  cluster_bitstring_to_id[UNK_BROWN] = 0;
  assert(kUNK_BROWN == 0);
  while (getline(in, line)) {
    istringstream lin(line);
    // read language, cluster, and surface form
    lin >> cluster_bitstring; assert(cluster_bitstring.size() > 0);
    lin >> word; assert(word.size() > 0);
    lin >> frequency; assert(frequency.size() > 0);

    // get the cluster id (and add a new one if necessary)
    if (cluster_bitstring_to_id.count(cluster_bitstring) == 0) {
      cluster_bitstring_to_id[cluster_bitstring] = cluster_bitstring_to_id.size();
    }
    int cluster_id = cluster_bitstring_to_id[cluster_bitstring];

    // get word id
    int word_id = corpus.get_or_add_word(word);

    // Associate this word id with this brown cluster id
    corpus.brown_clusters[word_id] = cluster_id;
  }
  BROWN_CLUSTERS_COUNT = cluster_bitstring_to_id.size() + 1;
}

// each line in this file consists of three tab-delimited fields: brown cluster bitstring, word, and frequency.
void load_brown2_clusters(std::string& filename) {
  // If the command line spcifies a file with a cluster map, load it.
  if (filename.size() == 0) {
    return;
  }
  cerr << "Loading the brown2 clusters from " << filename << "\n";
  ifstream in(filename.c_str());
  string line;
  string cluster_bitstring, word, frequency;
  unordered_map<string, int> cluster_bitstring_to_id;
  cluster_bitstring_to_id[UNK_BROWN] = 0;
  assert(kUNK_BROWN == 0);
  while (getline(in, line)) {
    istringstream lin(line);
    // read language, cluster, and surface form
    lin >> cluster_bitstring; assert(cluster_bitstring.size() > 0);
    lin >> word; assert(word.size() > 0);
    lin >> frequency; assert(frequency.size() > 0);

    // get the cluster id (and add a new one if necessary)
    if (cluster_bitstring_to_id.count(cluster_bitstring) == 0) {
      cluster_bitstring_to_id[cluster_bitstring] = cluster_bitstring_to_id.size();
    }
    int cluster_id = cluster_bitstring_to_id[cluster_bitstring];

    // get word id
    int word_id = corpus.get_or_add_word(word);

    // Associate this word id with this brown cluster id
    corpus.brown2_clusters[word_id] = cluster_id;
  }
  BROWN2_CLUSTERS_COUNT = cluster_bitstring_to_id.size() + 1;
}

int main(int argc, char** argv) {
  //unsigned random_seed = 0; // the seed is read from the clock
  unsigned random_seed = 11; // the seed is fixed everytime (to reproduce stuff)
  cnn::Initialize(argc, argv, random_seed);

  cerr << "COMMAND:"; 
  for (unsigned i = 0; i < static_cast<unsigned>(argc); ++i) cerr << ' ' << argv[i];
  cerr << endl;
  unsigned status_every_i_iterations = 100;

  po::variables_map conf;
  InitCommandLine(argc, argv, &conf);
  USE_POS = conf.count("use_pos_tags");

  corpus.USE_SPELLING = USE_SPELLING = conf.count("use_spelling");

  corpus.COARSE_ONLY = conf.count("coarse_only");

  if (conf.count("dropout")) { 
    // intialization should happen in only one place. right now, DROPOUT is initialized AND the conf parameter is defaulted.
    DROPOUT = conf["dropout"].as<double>();
  }
  if (conf.count("block_dropout_word_embedding")) {
    BLOCK_DROPOUT_WORD_EMBEDDING = conf["block_dropout_word_embedding"].as<double>();
  }
  if (conf.count("block_dropout_spell_embedding")) {
    BLOCK_DROPOUT_SPELL_EMBEDDING = conf["block_dropout_spell_embedding"].as<double>();
  }
  if (conf.count("block_dropout_pretrained_embedding")) {
    BLOCK_DROPOUT_PRETRAINED_EMBEDDING = conf["block_dropout_pretrained_embedding"].as<double>();
  }
  if (conf.count("block_dropout_fine_pos_embedding")) {
    BLOCK_DROPOUT_FINE_POS_EMBEDDING = conf["block_dropout_fine_pos_embedding"].as<double>();
  }
  if (conf.count("dropout_fine_pos_embedding")) {
    DROPOUT_FINE_POS_EMBEDDING = conf["dropout_fine_pos_embedding"].as<double>();
  }
  if (conf.count("block_dropout_typology_embedding")) {
    BLOCK_DROPOUT_TYPOLOGY_EMBEDDING = conf["block_dropout_typology_embedding"].as<double>();
  }

  EPOCHS = conf["epochs"].as<unsigned>();
  
  LAYERS = conf["layers"].as<unsigned>();
  INPUT_DIM = conf["input_dim"].as<unsigned>();
  PRETRAINED_DIM = conf["pretrained_dim"].as<unsigned>();
  HIDDEN_DIM = conf["hidden_dim"].as<unsigned>();
  ACTION_DIM = conf["action_dim"].as<unsigned>();
  LSTM_INPUT_DIM = conf["lstm_input_dim"].as<unsigned>();
  POS_DIM = conf["pos_dim"].as<unsigned>();
  REL_DIM = conf["rel_dim"].as<unsigned>();

  if (conf.count("score_file")) {
    SCORE_FILENAME = conf["score_file"].as<string>().c_str();
    cerr << "UAS score will be written to " << SCORE_FILENAME << ".";
  }

  // When the "typological-properties" command line option is specified, each word should
  // be formatted as "<language>:<surface_form>" where <language> is the 2-letter
  // ISO code for the language. For example: "en:dog" and "fr:chen".
  if (conf.count("typological_properties") > 0 && 
      conf["typological_properties"].as<string>().size() > 0) { 
    corpus.set_use_language_prefix(true); 
    string filename = conf["typological_properties"].as<string>();
    load_typological_properties_map(filename);
    TYPOLOGY_MODE = conf["typology_mode"].as<unsigned>();
  }

  const unsigned beam_size = conf["beam_size"].as<unsigned>();
  const unsigned unk_strategy = conf["unk_strategy"].as<unsigned>();
  cerr << "Unknown word strategy: ";
  if (unk_strategy == 1) {
    cerr << "STOCHASTIC REPLACEMENT\n";
  } else {
    abort();
  }
  const double unk_prob = conf["unk_prob"].as<double>();
  assert(unk_prob >= 0.); assert(unk_prob <= 1.);
  ostringstream os;
  os << "parser_" << (USE_POS ? "pos" : "nopos")
     << '_' << LAYERS
     << '_' << INPUT_DIM
     << '_' << HIDDEN_DIM
     << '_' << ACTION_DIM
     << '_' << LSTM_INPUT_DIM
     << '_' << POS_DIM
     << '_' << REL_DIM
     << "-pid" << getpid() << ".params";
  int best_correct_heads = 0;
  const string fname = os.str();
  if (conf.count("train")) { cerr << "Writing parameters to file: " << fname << endl; }
  bool softlinkCreated = false;

  // encode special symbols
  kUNK_SYMBOL = corpus.get_or_add_word(cpyp::Corpus::UNK);
  kROOT_SYMBOL = corpus.get_or_add_word(ROOT_SYMBOL);
  kUNK_CHAR_SYMBOL = corpus.get_or_add_char(cpyp::Corpus::UNK);
  kROOT_CHAR_SYMBOL = corpus.get_or_add_char(ROOT_SYMBOL);

  // Load pretrained embeddings
  if (conf.count("pretrained")) {
    corpus.pretrained[kUNK_SYMBOL] = vector<float>(PRETRAINED_DIM, 0);
    cerr << "Loading from " << conf["pretrained"].as<string>() << " with" << PRETRAINED_DIM << " dimensions\n";
    ifstream in(conf["pretrained"].as<string>().c_str());
    string line;
    getline(in, line);
    vector<float> v(PRETRAINED_DIM, 0);
    string word;
    while (getline(in, line)) {
      istringstream lin(line);
      lin >> word;
      for (unsigned i = 0; i < PRETRAINED_DIM; ++i) lin >> v[i];
      unsigned id = corpus.get_or_add_word(word);
      corpus.pretrained[id] = v;
    }
  }

  // Load training treebank followed by dev treebank. 
  corpus.load_correct_actions(conf["training_data"].as<string>());	
  assert(corpus.sentences_count > 0);
  VOCAB_SIZE = corpus.maxWord + 1;
  CHAR_SIZE = corpus.maxChars + 1;
  
  // Determine the number of unique words, actions, POS tags, and characters.
  POS_SIZE = corpus.maxPos + 10;
  ACTION_SIZE = corpus.actions_count + 1;
  possible_actions.resize(corpus.actions_count);
  for (unsigned i = 0; i < corpus.actions_count; ++i)
    possible_actions[i] = i;

  // Load brown clusters
  if (conf.count("brown_clusters") > 0 &&
      conf["brown_clusters"].as<string>().size() > 0) {
    string filename = conf["brown_clusters"].as<string>();
    load_brown_clusters(filename);
    cerr << "Number of brown clusters used (including UNK_BROWN) = " << BROWN_CLUSTERS_COUNT << endl;
    cerr << "Number of words for which we have brown clusters = " << corpus.brown_clusters.size() << endl;
  }
  
  // Load brown2 clusters
  if (conf.count("brown2_clusters") > 0 &&
      conf["brown2_clusters"].as<string>().size() > 0) {
    string filename = conf["brown2_clusters"].as<string>();
    load_brown2_clusters(filename);
    cerr << "Number of brown2 clusters used (including UNK_BROWN) = " << BROWN2_CLUSTERS_COUNT << endl;
    cerr << "Number of words for which we have brown2 clusters = " << corpus.brown2_clusters.size() << endl;
  }
  
  // Find out which words in the training vocabulary are singletons.
  set<unsigned> singletons;
  {
    // compute the singletons in the parser's training data
    unordered_map<unsigned, unsigned> counts;
    for (auto sent : corpus.sentences)
      for (auto tokenInfo : sent.second) { 
	counts[tokenInfo.word_id]++; 
      }
    for (auto wc : counts)
      if (wc.second == 1) singletons.insert(wc.first);
  }

  // Log some stats about the corpus.
  cerr << "Number of words in training vocab: " << corpus.training_vocab.size() << endl;
  cerr << "Total number of words: " << VOCAB_SIZE << endl;
  cerr << "Number of UTF8 chars: " << corpus.maxChars << endl;
  
  // Initialize the parser.
  Model parsing_model;
  ParserBuilder parser(&parsing_model);
  if (conf.count("parsing_model")) {
    cerr << "reading the parsing_model from " << conf["parsing_model"].as<string>().c_str() << "...";
    ifstream in(conf["parsing_model"].as<string>().c_str());
    boost::archive::text_iarchive ia(in);
    ia >> parsing_model;
    cerr << "done." << endl;
  }

  // we are only allowed to read the dev data after initializing the parsing_model
  corpus.load_correct_actionsDev(conf["dev_data"].as<string>());
  
  // TRAINING
  if (conf.count("train")) {
    signal(SIGINT, signal_callback_handler);
    SimpleSGDTrainer adam(&parsing_model);
    //AdamTrainer adam(&parsing_model);
    adam.eta_decay = 0.08;
    cerr << "Training started."<<"\n";
    vector<unsigned> order(corpus.sentences_count);
    unordered_map<int, vector<unsigned>> order_per_lang;
    for (unsigned i = 0; i < corpus.sentences_count; ++i) {
      order[i] = i;
      if (corpus.use_language_prefix) {
        order_per_lang[corpus.sentences[i][0].lang_id].push_back(i);
      } else {
        order_per_lang[-1].push_back(i);
        assert(order_per_lang.size() == 1 && 
               "Language may be specified for all sentences or none, but not for some.");
      }
    }
    unsigned min_sents_per_lang = corpus.sentences_count;
    for (auto& pair : order_per_lang) {
      min_sents_per_lang = min(min_sents_per_lang, 
                               static_cast<unsigned>(pair.second.size()));
    }
    double tot_seen = 0;
    status_every_i_iterations = min(status_every_i_iterations, corpus.sentences_count);
    unsigned si = min_sents_per_lang;
    cerr << "NUMBER OF TRAINING SENTENCES: " << corpus.sentences_count << endl;
    cerr << "epoch size: " << min_sents_per_lang << "(when the training data consists of multiple languages, this is the minimum number of sentences in the same language)" << endl;
    unsigned trs = 0;
    double right = 0;
    double llh = 0;
    double sum_of_gradient_norms = 0;
    bool first = true;
    int iter = -1;

    unsigned epoch_count = 0;
    while(!requested_stop) {
      ++iter;
      // Work for a few iterations, then report status.
      for (unsigned sii = 0; sii < status_every_i_iterations; ++sii) {
        
        // Shuffle the order in which we process sentences after a full epoch.
        if (si == min_sents_per_lang) {
          si = 0;
          if (first) { 
            first = false; 
          } else { 
            adam.update_epoch(); 
            epoch_count += 1;
          }
          if (epoch_count == EPOCHS) {
            requested_stop = true;
            break;
          }
          cerr << "**SHUFFLE\n";
          random_shuffle(order.begin(), order.end());
          cerr << "order[0] = " << order[0] << endl;
          order_per_lang.clear();
          for (const int &i : order) {
            if (corpus.use_language_prefix) {
              order_per_lang[corpus.sentences[i][0].lang_id].push_back(i);
            } else {
              order_per_lang[-1].push_back(i);
            }
          }
        }

        // Every iteration, we process one sentence per language.
        tot_seen += order_per_lang.size();

        // define one computation graph per multilingual sentence group.
        
        // Go over the languages one at a time.
        for(auto const &order_per_lang_iter : order_per_lang) {

          // Find out which sentence id is next by indexing into the current language's
          // shuffled sentence ids. Languages which have fewer sentences repeat their
          // sentences in the same order until next shuffle.
          int sent_id = order_per_lang_iter.second[si % order_per_lang_iter.second.size()];
          vector<TokenInfo>& sentence = corpus.sentences[sent_id];
          
          // Mark some singletons as OOVs while training so that the parsing_model knows how to deal with real OOVs in the dev set.
          // This overrides the property TokenInfo.training_oov of tokens in the training treebank.
          if (unk_strategy == 1) {
            for (auto &tokenInfo : sentence) {
              tokenInfo.training_oov = (singletons.count(tokenInfo.word_id) && cnn::rand01() < unk_prob);
            }
          }
          const vector<unsigned>& actions = corpus.correct_act_sent[sent_id];

          vector<unsigned> dummy_actions;
          ComputationGraph hg;
          parser.log_prob_parser(hg, sentence, actions, corpus.actions, &right, dummy_actions);
          double lp = as_scalar(hg.incremental_forward());
          if (std::isnan(lp)) { 
	    cerr << "WARNING: log_prob = nan" << endl;
	    continue; 
	  }
	  assert(!std::isnan(lp));
          hg.backward();
          assert(lp >= -0.1);
          llh += lp;
          trs += actions.size();
        }
        
        double gradient_l2_norm = parsing_model.gradient_l2_norm();
        //cerr << "gradient_l2_norm =" << gradient_l2_norm << endl;
        if (std::isnan(gradient_l2_norm)) {
          cerr << "WARNING: gradient_l2_norm is nan. there's a bug somewhere in CNN which gave rise to this. As a workaround, I'm going to reset the gradient." << endl;
          parsing_model.reset_gradient();
        }
        sum_of_gradient_norms += gradient_l2_norm;
        adam.update(1.0);
        
        ++si;
      }
      adam.status();
      cerr << "update #" << iter << " (epoch " << (epoch_count) << ")\t"  
           << "llh: " << llh 
           << " sum_of_gradient_norms: " << sum_of_gradient_norms
           << " ppl: " << exp(llh / trs) 
           << " err: " << (trs - right) / trs << endl;
      llh = trs = right = sum_of_gradient_norms = 0;

      static int logc = 0;
      ++logc;
      if ((logc < 100 && logc % 5 == 1) || 
          (logc >= 100 && logc % 25 == 1)) { // report on dev set
        unsigned dev_size = corpus.sentencesDev_count;
        // dev_size = 100;
        double llh = 0;
        double trs = 0;
        double right = 0;
        double correct_heads = 0;
        double total_heads = 0;
        auto t_start = std::chrono::high_resolution_clock::now();
        cerr << "train OOV rates:" << endl;
        cerr << "brown OOV rate is " << brown_oov_count << " / " << (brown_oov_count + brown_non_oov_count) << endl;
        cerr << "brown2 OOV rate is " << brown2_oov_count << " / " << (brown2_oov_count + brown2_non_oov_count) << endl;
        cerr << "pretrained OOV rate is " << pretrained_oov_count << " / " << (pretrained_oov_count + pretrained_non_oov_count) << endl;
        cerr << "learned OOV rate is " << learned_oov_count << " / " << (learned_oov_count + learned_non_oov_count) << endl;
        brown_non_oov_count = 0, brown_oov_count = 0, brown2_non_oov_count = 0, brown2_oov_count = 0, pretrained_oov_count = 0, pretrained_non_oov_count = 0, learned_non_oov_count = 0, learned_oov_count = 0;

        for (unsigned sii = 0; sii < dev_size; ++sii) {
          const vector<TokenInfo>& sentence = corpus.sentencesDev[sii];
          const vector<unsigned>& actions = corpus.correct_act_sentDev[sii];

          ComputationGraph hg;
          vector<unsigned> pred;
          parser.log_prob_parser(hg, sentence, vector<unsigned>(), corpus.actions, &right, pred);
          double lp = 0;
          //vector<unsigned> pred = parser.log_prob_parser_beam(hg,sentence,sentencePos,corpus.actions,beam_size,&lp);
          llh -= lp;
          trs += actions.size();
          unordered_map<int, string> rel_ref, rel_hyp;
	  unordered_map<int,int> ref = parser.compute_heads(sentence.size(), actions, corpus.actions, &rel_ref);
          unordered_map<int,int> hyp = parser.compute_heads(sentence.size(), pred, corpus.actions, &rel_hyp);
          //output_conll(sentence, corpus.intToWords, ref, hyp);
          correct_heads += compute_correct(ref, hyp, sentence.size() - 1);
          total_heads += sentence.size() - 1;
        }

        cerr << "dev OOV rates:" << endl;
        cerr << "brown OOV rate is " << brown_oov_count << " / " << (brown_oov_count + brown_non_oov_count) << endl;
        cerr << "brown2 OOV rate is " << brown2_oov_count << " / " << (brown2_oov_count + brown2_non_oov_count) << endl;
        cerr << "pretrained OOV rate is " << pretrained_oov_count << " / " << (pretrained_oov_count + pretrained_non_oov_count) << endl;
        cerr << "learned OOV rate is " << learned_oov_count << " / " << (learned_oov_count + learned_non_oov_count) << endl;
        brown_non_oov_count = 0, brown_oov_count = 0, brown2_non_oov_count = 0, brown2_oov_count = 0, pretrained_oov_count = 0, pretrained_non_oov_count = 0, learned_non_oov_count = 0, learned_oov_count = 0;

        auto t_end = std::chrono::high_resolution_clock::now();
        cerr << "  **dev (iter=" << iter << " epoch=" << (epoch_count + (1.0 * si) / min_sents_per_lang) << ")\tllh=" << llh << " ppl: " << exp(llh / trs) << " err: " << (trs - right) / trs << " uas: " << (correct_heads / total_heads) << "\t[" << dev_size << " sents in " << std::chrono::duration<double, std::milli>(t_end-t_start).count() << " ms]" << endl;
        if (correct_heads > best_correct_heads) {
          best_correct_heads = correct_heads;
          ofstream out(fname);
          boost::archive::text_oarchive oa(out);
          oa << parsing_model;
          // Create a soft link to the most recent parsing_model in order to make it
          // easier to refer to it in a shell script.
          if (!softlinkCreated) {
            string softlink = " latest_model";
            if (system((string("rm -f ") + softlink).c_str()) == 0 && 
                system((string("ln -s ") + fname + softlink).c_str()) == 0) {
              cerr << "Created " << softlink << " as a soft link to " << fname 
                   << " for convenience." << endl;
            }
            softlinkCreated = true;
          }
	  out.close();
	  // write the parsing_model's UAS performance to file
	  if (SCORE_FILENAME != "") {
	    ofstream score_file(SCORE_FILENAME);
	    score_file << (correct_heads / total_heads);
	    score_file.close();
	  }
	}
      }
    }

    // reload the best parsing_model
    cerr << "reloading the best parsing_model from file latest_model...";
    ifstream in("latest_model");
    boost::archive::text_iarchive ia(in);
    ia >> parsing_model;
    cerr << "done." << endl;
    
  } // should do training?

  // keep the parser running in the background to serve other processes.
  if (conf.count("server")) {
    while (true) {
      // prompt for input
      cerr << "Type in a sequence of tokens, e.g., `en:he-pron en:is-verb en:stupid-adj', then press ENTER" << endl;
      
      // read input
      string input_sentence;
      getline(cin, input_sentence);

      // interpret input
      if (input_sentence == "QUIT" || input_sentence == "EXIT") { break; }
      istringstream input_sentence_stream(input_sentence);
      vector<TokenInfo> sentence;
      while (true) { 
	string lang_word_pos;
	input_sentence_stream >> lang_word_pos;
	cerr << "lang_word_pos = " << lang_word_pos << endl;
	if (lang_word_pos.size() == 0) { break; }
	TokenInfo current_token;
	corpus.ReadTokenInfo(lang_word_pos, current_token);
	current_token.training_oov = (corpus.training_vocab.count(current_token.word_id) == 0);
	sentence.push_back(current_token);
      }
      TokenInfo root_token;
      corpus.ReadTokenInfo("ROOT-ROOT", root_token);
      root_token.training_oov = (corpus.training_vocab.count(root_token.word_id) == 0);
      sentence.push_back(root_token);
      
      // parse!
      double right = 0;
      ComputationGraph cg;
      vector<unsigned> pred;
      auto t_start = std::chrono::high_resolution_clock::now();
      parser.log_prob_parser(cg, sentence, vector<unsigned>(), corpus.actions, &right, pred);
      auto t_end = std::chrono::high_resolution_clock::now();

      // compute heads
      unordered_map<int, string> rel_hyp;
      unordered_map<int,int> hyp = parser.compute_heads(sentence.size(), pred, corpus.actions, &rel_hyp);
      output_conll(sentence, hyp, rel_hyp);

      // print output
      cerr << std::chrono::duration<double, std::milli>(t_end-t_start).count() << " ms" << endl;
    }
    cerr << "The language-universal dependency parsing service has been terminated." << endl;
  } else { // do test evaluation

    double llh = 0;
    double trs = 0;
    double right = 0;
    double correct_heads = 0;
    double total_heads = 0;
    auto t_start = std::chrono::high_resolution_clock::now();
    unsigned corpus_size = corpus.sentencesDev_count;
    for (unsigned sii = 0; sii < corpus_size; ++sii) {
      const vector<TokenInfo>& sentence = corpus.sentencesDev[sii];
      const vector<unsigned>& actions=corpus.correct_act_sentDev[sii];
      ComputationGraph cg;
      double lp = 0;
      vector<unsigned> pred;
      if (beam_size == 1) {
        parser.log_prob_parser(cg, sentence, vector<unsigned>(), corpus.actions, &right, pred);
      } else {
        assert(false && "beam search not implemented");
      }
      llh -= lp;
      trs += actions.size();
      unordered_map<int, string> rel_ref, rel_hyp;
      unordered_map<int,int> ref = parser.compute_heads(sentence.size(), actions, corpus.actions, &rel_ref);
      unordered_map<int,int> hyp = parser.compute_heads(sentence.size(), pred, corpus.actions, &rel_hyp);
      output_conll(sentence, hyp, rel_hyp);
      correct_heads += compute_correct(ref, hyp, sentence.size() - 1);
      total_heads += sentence.size() - 1;
    }
    auto t_end = std::chrono::high_resolution_clock::now();
    cerr << "TEST llh=" << llh << " ppl: " << exp(llh / trs) << " err: " << (trs - right) / trs << " uas: " << (correct_heads / total_heads) << "\t[" << corpus_size << " sents in " << std::chrono::duration<double, std::milli>(t_end-t_start).count() << " ms]" << endl;

  }
}

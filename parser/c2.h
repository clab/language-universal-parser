#ifndef CPYPDICT_H_
#define CPYPDICT_H_

#include <string>
#include <iostream>
#include <cassert>
#include <fstream>
#include <sstream>
#include <vector>
#include <set>
#include <unordered_map>
#include <functional>
#include <vector>
#include <map>
#include <string>

using namespace std;

namespace cpyp {

  struct TokenInfo {
    unsigned word_id = 0;
    unsigned lang_id = 0;
    unsigned pos_id = 0; // fine-grained
    unsigned coarse_pos_id = 0;
    unsigned predicted_pos_id = 0; // fine-grained
    unsigned predicted_coarse_pos_id = 0;
    bool training_oov = true;
  };

class Corpus {
public: 

   bool USE_SPELLING = false;
   bool COARSE_ONLY = false;
   bool PREDICT_ATTACHMENTS_ONLY = false;

   map<int, vector<TokenInfo>> sentences;
   map<int, vector<TokenInfo>> sentencesDev;

   map<int, vector<unsigned>> correct_act_sent;
   map<int, vector<unsigned>> correct_act_sentDev;
   
   unordered_map<unsigned, vector<float>> pretrained;
   unordered_map<unsigned, vector<float>> typological_properties_map;
   unordered_map<unsigned, unsigned> brown_clusters;
   unordered_map<unsigned, unsigned> brown2_clusters;

   set<unsigned> training_vocab; // words available in the training corpus
   set<unsigned> training_pos_vocab; // pos available in the training corpus
   set<unsigned> coarse_pos_vocab; // coarse pos available in any corpus
   set<unsigned> fine_pos_vocab; // coarse pos available in any corpus
   set<unsigned> training_char_vocab; // chars available in the training corpus
   
   unsigned sentences_count = 0;
   unsigned sentencesDev_count = 0;
   unsigned actions_count = 0;
   
   unsigned maxWord = 0;
   unsigned maxPos = 1;
   unsigned maxLang = 0;

   map<string, unsigned> wordsToInt;
   map<unsigned, string> intToWords;
   map<unsigned, vector<unsigned> > wordIntsToCharInts;
   vector<string> actions;

   map<string, unsigned> posToInt;
   map<unsigned, string> intToPos;

   int maxChars;
   map<string, unsigned> charsToInt;
   map<unsigned, string> intToChars;

   map<string, unsigned> langToInt;
   map<unsigned, string> intToLang;
   bool use_language_prefix;

   // String literals
   static constexpr const char* UNK = "UNK";
   static constexpr const char* BAD0 = "<BAD0>";

 public:
  Corpus() {
    maxWord = 0;
    maxPos = 0;
    maxChars=0;

    // always add the UNK language.
    use_language_prefix = false;
    langToInt["UNK"] = 0; // unknown language
    intToLang[0] = "UNK";
    maxLang = 1;
  }

  inline void set_use_language_prefix(bool use) {
    use_language_prefix = use;
  }
 
  inline bool get_use_language_prefix() {
    return use_language_prefix;
  }
  
  inline unsigned UTF8Len(unsigned char x) {
    if (x < 0x80) return 1;
    else if ((x >> 5) == 0x06) return 2;
    else if ((x >> 4) == 0x0e) return 3;
    else if ((x >> 3) == 0x1e) return 4;
    else if ((x >> 2) == 0x3e) return 5;
    else if ((x >> 1) == 0x7e) return 6;
    else return 0;
  }

  void ReadTokenInfo(string lang_word_pos, TokenInfo &current_token) {
    // remove the trailing comma if need be.
    if (lang_word_pos[lang_word_pos.size() - 1] == ',') { 
      lang_word_pos = lang_word_pos.substr(0, lang_word_pos.size() - 1);
    }
    
    // identify the POS.
    size_t posIndex = lang_word_pos.rfind('-');
    if (posIndex == string::npos) {
      cerr << "lang_word_pos = " << lang_word_pos << endl;
      assert(false && "FATAL: Bad treebank format. I can't find the dash between a word and a POS tag.");
    }
    string pos = lang_word_pos.substr(posIndex + 1);
    unsigned pos_id = COARSE_ONLY? 0 : get_or_add_pos(pos);
    size_t size_of_coarse_pos_substring = pos.find(':');
    string coarse_pos = 
      size_of_coarse_pos_substring == string::npos?
      pos + "_c" :
      pos.substr(0, size_of_coarse_pos_substring);
    unsigned coarse_pos_id = get_or_add_pos(coarse_pos);
    coarse_pos_vocab.insert(coarse_pos_id);
    if (!COARSE_ONLY) { fine_pos_vocab.insert(pos_id); }
    string lang_word = lang_word_pos.substr(0, posIndex);
    //cerr << "lang_word_pos = " << lang_word_pos << endl;
    //cerr << "coarse_pos = " << coarse_pos << endl;
    //cerr << "coarse_pos_id = " << coarse_pos_id << endl;
    //cerr << "pos = " << pos << endl;
    //cerr << "pos_id = " << pos_id << endl << endl;
   
    // identify the language.
    unsigned lang_id = 0;
    string lang;
    if (use_language_prefix) {
      // Each word must be formatted as "en:with" or "fr:avec"
      // The only exception here is "ROOT".
      if (lang_word != "ROOT" && (lang_word.size() < 3 || lang_word[2] != ':')) {
	cerr << "lang_word = " << lang_word << endl;
	assert(false && "Language typology is provided but the 2-letter langauge prefix is missing from the current token (lang_word).");
      }
      lang = (lang_word == "ROOT")? "__" : lang_word.substr(0,2);
      lang_id = get_or_add_lang(lang);
    }
    
    // Identify the "word" for which we tune embeddings by default.
    unsigned word_id = get_or_add_word(lang_word);
    
    // Identify the "surface form", which we use to estimate char-based word embeddings. 
    unsigned kROOT = get_or_add_word("ROOT");
    if (wordIntsToCharInts.count(word_id) == 0) {
      if (word_id == kROOT) {
	unsigned special_char = get_or_add_char("ROOT");
	wordIntsToCharInts[kROOT].push_back(special_char);
      } else {
	string surface_form = (use_language_prefix)? lang_word.substr(3) : lang_word;
	
	// Add utf8_characters to charsToInt and intToChars if need be.
	unsigned j = 0;
	while(j < surface_form.length()) {
	  string utf8_char = "";
	  for (unsigned h = j; h < j + UTF8Len(surface_form[j]); h++) {
	    utf8_char += surface_form[h];
	  }
	  j += UTF8Len(surface_form[j]);
	  
	  unsigned char_id = get_or_add_char(utf8_char);
	  wordIntsToCharInts[word_id].push_back(char_id);
	}
      }
    }
    
    // Add this token details to the sentence.
    current_token.word_id = word_id;
    current_token.pos_id = pos_id;
    current_token.coarse_pos_id = coarse_pos_id;
    current_token.lang_id = lang_id;
  }

  inline void load_correct_actions(string file){
    
    ifstream actionsFile(file);
    //correct_act_sent=new vector<vector<unsigned>>();
    string lineS;
    
    int count=-1;
    int sentence=-1;
    bool initial=false;
    bool first=true;
    get_or_add_word(BAD0);
    get_or_add_word(UNK);
    get_or_add_char(BAD0);
    assert(maxPos == 0);
    assert(maxLang > 0);
    maxPos = 1;
    
    vector<TokenInfo> current_sent;
    while (getline(actionsFile, lineS)){
      ReplaceStringInPlace(lineS, "-RRB-", "_RRB_");
      ReplaceStringInPlace(lineS, "-LRB-", "_LRB_");
      if (lineS.empty()) {
	count = 0;
	if (!first) {
	  sentences[sentence] = current_sent;
	}
	
	sentences_count = ++sentence;
	
	initial = true;
	current_sent.clear();
      } else if (count == 0) {
	first = false;
	//stack and buffer, for now, leave it like this.
	count = 1;
	if (initial) {
	  // the initial line in each sentence may look like:
	  // [][the-det, cat-noun, is-verb, on-adp, the-det, mat-noun, ,-punct, ROOT-ROOT]
	  // first, get rid of the square brackets.
	  lineS = lineS.substr(3, lineS.size() - 4);
	  // read the initial line, token by token "the-det," "cat-noun," ...
	  istringstream iss(lineS);
	  do {
	    string lang_word_pos;
	    iss >> lang_word_pos;
	    if (lang_word_pos.size() == 0) { continue; }
	    
	    TokenInfo tokenInfo;
	    ReadTokenInfo(lang_word_pos, tokenInfo);
	    tokenInfo.training_oov = false; // because this is the training data.
	    training_vocab.insert(tokenInfo.word_id);
	    current_sent.push_back(tokenInfo);
	  } while(iss);
	}
	initial = false;
      }
      else if (count == 1){
        // find the action string
	size_t open_bracket_position = lineS.find('(');
	string actionString;
	if (PREDICT_ATTACHMENTS_ONLY && open_bracket_position != string::npos) {
	  actionString = lineS.substr(0, open_bracket_position);
	  // string label = lineS.substr(open_bracket_position, lineS.length() - open_bracket_position - 1); /*unused*/
	} else {
	  actionString = lineS;
	}

	// add the index of this action to the vector of correct actions for this sentence
	auto actionIter = find(actions.begin(), actions.end(), actionString);
	if (actionIter == actions.end()) {
	  actions.push_back(actionString);
	  cerr << "adding " << actionString << "to the list of possible actions" << endl;
	  actionIter = find(actions.begin(), actions.end(), actionString);
	  assert(actionIter != actions.end());
	}
	unsigned actionIndex = distance(actions.begin(), actionIter);
	correct_act_sent[sentence].push_back(actionIndex);
	
	count = 0;
      }
    }
    
    // Add the last sentence.
    if (current_sent.size() > 0) {
      sentences[sentence] = current_sent;
      sentences_count = ++sentence;
    }
    
    actionsFile.close();
    
    // add all pos ids and char ids available now to the correspodning training vocab
    for (auto pos: intToPos) {
      training_pos_vocab.insert(pos.first);
    }
    for (auto c: intToChars) {
      training_char_vocab.insert(c.first);
    }
    
    cerr << "done" << "\n";
    for (auto a: actions) {
      cerr << a << "\n";
    }
    actions_count = actions.size();
    cerr << "actions_count:" << actions_count << "\n";
    cerr << "maxWord:" << maxWord << "\n";
    for (unsigned i = 0; i < maxPos; i++) {
      cerr << i << ":" << intToPos[i] << "\n";
    }
    actions_count = actions.size();
    
  }
  
  inline string lookup_lang(unsigned id) {
    if (id < maxLang) {
      return intToLang[id];
    } else {
      return intToLang[0];
    }
  }
  
  inline unsigned get_or_add_lang(const string& lang) {
    unsigned&	id = langToInt[lang];
    if (id == 0) {
      id		   = maxLang++;
      intToLang[id]  = lang;
    }
    return id;
  }
  
  inline unsigned get_or_add_word(const string& word) {
    unsigned&	id = wordsToInt[word];
    if (id == 0) {
      id		   = maxWord++;
      intToWords[id] = word;
    }
    return id;
  }
  
  inline unsigned get_or_add_pos(const string& pos) {
    unsigned&	id = posToInt[pos];
    if (id == 0) {
      id		   = maxPos++;
      intToPos[id]   = pos;
    }
    return id;
  }
  
  inline unsigned get_or_add_char(const string& utf8_char) {
    unsigned& id = charsToInt[utf8_char];
    if (id == 0) {
      id		   = maxChars++;
      intToChars[id] = utf8_char;
    }
    return id;
  }
 
inline void load_correct_actionsDev(string file) {
  if (training_vocab.size() == 0) {
    assert(false && "FATAL: load_correct_actions() MUST be called before load_correct_actionsDev() because otherwise we can't tell if a word in the dev treebank is OOV");
  }

  ifstream actionsFile(file);
  string lineS;

  assert(maxPos > 1);
  assert(maxWord > 3);
  int count = -1;
  int sentence_id = -1;
  bool initial = false;
  bool first = true;
  vector<TokenInfo> current_sent;
  while (getline(actionsFile, lineS)) {
    ReplaceStringInPlace(lineS, "-RRB-", "_RRB_");
    ReplaceStringInPlace(lineS, "-LRB-", "_LRB_");
    if (lineS.empty()) {
      // an empty line marks the end of a sentence.
      count = 0;
      if (!first) {
        sentencesDev[sentence_id] = current_sent;
      }
      
      sentencesDev_count = ++sentence_id;
      
      initial = true;
      current_sent.clear();
    } else if (count == 0) {
      first = false;
      //stack and buffer, for now, leave it like this.
      count = 1;
      if (initial) {
        // the initial line in each sentence may look like:
        // [][the-det, cat-noun, is-verb, on-adp, the-det, mat-noun, ,-punct, ROOT-ROOT]
        // first, get rid of the square brackets.
        lineS = lineS.substr(3, lineS.size() - 4);
        // read the initial line, token by token "the-det," "cat-noun," ...
        istringstream iss(lineS);
        do {

          // Read token.
          string lang_word_pos;
          iss >> lang_word_pos;
          if (lang_word_pos.size() == 0) { continue; }

          TokenInfo tokenInfo;
          ReadTokenInfo(lang_word_pos, tokenInfo);
          // it's an OOV if it didn't appear in the training treebank.
          tokenInfo.training_oov = (training_vocab.count(tokenInfo.word_id) == 0);
          current_sent.push_back(tokenInfo);

        } while(iss);
      }
      initial = false;
    } else if (count == 1) {
      size_t open_bracket_position = lineS.find('(');
      string actionString;
      if (PREDICT_ATTACHMENTS_ONLY && open_bracket_position != string::npos) {
	actionString = lineS.substr(0, open_bracket_position);
	// string label = lineS.substr(open_bracket_position, lineS.length() - open_bracket_position - 1); /*unused*/
      } else {
	actionString = lineS;
      }
      auto actionIter = find(actions.begin(), actions.end(), actionString);
      if (actionIter != actions.end()) {
        unsigned actionIndex = distance(actions.begin(), actionIter);
        correct_act_sentDev[sentence_id].push_back(actionIndex);
      } else {
	cerr << "new actionString in dev set: " << actionString << endl;
	assert(false);
        // TODO: right now, new actions which haven't been observed in training
        // are not added to correct_act_sentDev. This may be a problem if the
        // training data is little.
      }
      count=0;
    }
  }

  // Add the last sentence.
  if (current_sent.size() > 0) {
    sentencesDev[sentence_id] = current_sent;
    sentencesDev_count = ++sentence_id;
  }
  
  actionsFile.close();
}

void ReplaceStringInPlace(string& subject, const string& search,
                          const string& replace) {
    size_t pos = 0;
    while ((pos = subject.find(search, pos)) != string::npos) {
         subject.replace(pos, search.length(), replace);
         pos += replace.length();
    }
}

string ReplaceStringAndReturnNew(string str, const string& from,
                                      const string& to) {
  size_t pos = 0;
  while ( (pos = str.find(from, pos)) != string::npos) {
    str.replace(pos, from.length(), to);
    pos += to.length();
  }
  return str;
}
 
};

} // namespace

#endif

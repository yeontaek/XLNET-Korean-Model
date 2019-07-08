# XLNET-Korean-Model
This is a repository of Korean XLNET model with SentencePiece tokenizer.

* XLNET-Base, Korean Model : 12-layer, 768-hidden, 12-heads
* XLNET-Large, Korean Model: 24-layer, 1024-hidden, 16-heads

## SentencePiece tokenizer 학습
 1억 8천만 문장(위키피디아, 뉴스 데이터)을 활용하여 32,000개의 vocabulary (subwords)를 학습하였습니다. XLNET에서는 model type을 unigram을 사용하였는데,  해당 github에 문의 결과 다른 모델 타입도 문제가 없을 것이라 하여 본 모델에서는 bpe type으로 사전을 구축하였습니다.(https://github.com/zihangdai/xlnet/issues/22)
 
 
```python
import sentencepiece as spm
RAW_DATA_FPATH = "total_corpus_20190605.txt"

MODEL_PREFIX = "sp10m.cased.v3"
VOC_SIZE = 32000
COVERAGE = 1.0
SPM_COMMAND = ('--input={} '
               '--model_prefix={} '
               '--vocab_size={} '
               '--character_coverage={} '
               '--shuffle_input_sentence=true ' 
               '--model_type=bpe '
               '--control_symbols=<cls>,<sep>,<pad>,<mask>,<eod> '
	             '--user_defined_symbols=<eop>,.,(,),",-,–,£,€ ').format(
               RAW_DATA_FPATH,
               MODEL_PREFIX,
               VOC_SIZE,
               COVERAGE)

#sentencePiece 학습 시작
spm.SentencePieceTrainer.Train(SPM_COMMAND)
```   
<br>

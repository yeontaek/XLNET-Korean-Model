# XLNET-Korean-Model
This is a repository of Korean XLNET model with SentencePiece tokenizer.

* XLNET-Base, Korean Model : 12-layer, 768-hidden, 12-heads
* XLNET-Large, Korean Model: 24-layer, 1024-hidden, 16-heads

## SentencePiece tokenizer 학습
 1억 8천만 문장(위키피디아, 뉴스 데이터)을 활용하여 32,000개의 vocabulary (subwords)를 학습하였습니다. XLNET에서는 model type을 unigram을 사용하였습니다. 그러나 해당 github에 model type 관련 문의 결과 다른 모델 타입을 사용해도 문제가 없을 것이라는 의견이 있어 본 모델에서는 bpe type으로 사전을 구축하였습니다.(https://github.com/zihangdai/xlnet/issues/22)
 
 
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


## 사전 학습 데이터 준비  
data_utils.py를 사용하여 <code>.tfrecord</code> 파일 형식으로 변환하였습니다. BERT 모델과 마찬가지로 학습 데이터의 구성은 한 줄에 한 문장씩 구성하고 Document 사이에는 빈 줄을 삽입할 것을 권장하고 있습니다. 선택사항으로 특정 문장의 끝에 \<eop\> 토큰을 삽입하여 해당 문장이 단락 끝임을 표시하는 토큰을 넣을 수 있습니다. 그러나 본 모델에서는 해당 토큰을 사용하지 않고 BERT Model 학습시 사용했던 wiki 문서를 동일하게 사용했습니다. 
 
~~~
라 토스카(La Tosca)는 1887년에 프랑스 극작가 사르두가 배우 사라 베르나르를 위해 만든 작품이다.
1887년 파리에서 처음 상연되었다.
1990년 베르나르를 주인공으로 미국 뉴욕에서 재상연되었다.
1800년 6월 중순의 이탈리아 로마를 배경으로 하며, 당시의 시대적 상황 하에서 이야기가 전개된다.
1900년, 사르두의 연극은 푸치니의 오페라 토스카로 새롭게 각색되었다.
베르디는 사드루의 각본에서 "갑작스런 종결" 부분을 수정할 것을 권하지만, 사르루는 이를 거절한다.
후에, 푸치니 또한 사르두의 각본에서 "갑작스런 종결부분"을 수정할 것을 제안하지만 끝내 사르두를 설득하지 못했다.

2008년 하계 올림픽의 복싱 남자 라이트급 종목은 8월 11일일부터 8월 24일까지 중화인민공화국의 베이징에 있는 베이징 노동자 체육관에서 열렸다.
27개국에서 27명의 선수가 참가하였다.
2008년 하계 올림픽 복싱 남자 라이트급 경기는 개최 도시인 베이징에 있는 베이징 노동자 체육관에서 경기가 열렸다.

도리데 시는 일본 이바라키현의 남부에 있는 시이다.
간토 평야에 위치하고 도네 강과 고카이 강에 접하고 있다.
이 때문인지 일찍이 수해가 많았다.
현재에도 시 남서부의 대지를 제외하면 시역이 많은 부분이 침수의 위험성이 있다.
그러나 최근에 도네 강, 고카이 강 등의 제방의 고기능화에 의해 하천의 범람에 의한 침수 피해는 거의 없어졌다.
한편 집중호우에 의해 시내의 저지 등에서는 도로가 일부 침수하는 등의 피해가 일어난다.
~~~

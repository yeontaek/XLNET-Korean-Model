# XLNET-Korean-Model

* XLNET-Base, Korean Model : 12-layer, 768-hidden, 12-heads
* XLNET-Large, Korean Model: 24-layer, 1024-hidden, 16-heads

## SentencePiece tokenizer 학습
 1억 8천만 문장(위키피디아, 뉴스 데이터)을 활용하여 32,000개의 vocabulary (subwords)를 학습하였습니다. XLNET에서는 model type을 unigram을 사용하였습니다. 그러나 model type 관련 문의 결과 다른 모델 타입을 사용해도 문제가 없을 것이라는 의견이 있어 본 모델에서는 bpe type으로 사전을 구축하였습니다.(https://github.com/zihangdai/xlnet/issues/22)
 <br>
 <br>
 추가로 <code>character_coverage</code>도 기존 0.99995에서 1.0로 변경하여 학습을 진행했습니다. 이는 BERT Model에서 0.9995보다 1.0으로 구축하였을 경우 더 좋은 성능을 얻어 XLNET에서도 동일하게 적용하였습니다. 
 
 
```python
import sentencepiece as spm
RAW_DATA_FPATH = "corpus.txt"

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
data_utils.py를 사용하여 <code>.tfrecord</code> 파일 형식으로 변환하였습니다. BERT 모델과 마찬가지로 학습 데이터의 구성은 한 줄에 한 문장씩 구성하고 Document 사이에는 빈 줄을 삽입할 것을 권장하고 있습니다. XLNET에서는 선택사항으로 특정 문장의 끝에 \<eop\> 토큰을 삽입하여 해당 문장이 단락 끝임을 표시하는 토큰을 넣을 수 있습니다. 그러나 본 모델에서는 해당 토큰을 사용하지 않고 한국어 BERT Model 학습시 사용했던 wiki 문서를 동일하게 사용했습니다. 
 
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

## Pretraining with XLNet
학습 데이터는 **한국어 위키데이터(2019.01 dump file, 약 350만 문장)** 을 사용하여 학습을 진행하였으며, 모델의 하이퍼파라미터는 base model에 맞춰서 수정하였습니다.(https://github.com/zihangdai/xlnet/issues/137)
<br>
<br>
학습 step은 논문과 동일하게 50만 step을 진행하였습니다.

```python
python train.py
  —record_info_dir=$DATA/tfrecords \
  —train_batch_size=2048 \
  —seq_len=512 \
  —reuse_len=256 \
  —mem_len=384 \
  —perm_size=256 \
  —n_layer=12 \
  —d_model=768 \
  —d_embed=768 \
  —n_head=12 \
  —d_head=64 \
  —d_inner=3072 \
  —untie_r=True \
  —mask_alpha=6 \
  —mask_beta=1 \
  —num_predict=85
```   
<br>

## KorQuAD Task   
XLNET Model 성능 평가을 위해 한국어 SQuAD Task [KorQuAD](https://korquad.github.io/)로 평가를 진행하였습니다. XLNET github에는 SQuAD 2.0에 대한 평가 코드가 있어 이를 KorQuAD Task에 맞춰서 수정했습니다. 아래 KorQuAD 관련 Flag를 하나 추가하였으며, 관련 코드는 github에 첨부하였습니다. 
```python
flags.DEFINE_bool("korquad", default= True, help="True when using Korquad, False if not")
```  



## 성능 평가  
XLNET Model 성능 평가는 한국어 SQuAD Task [KorQuAD](https://korquad.github.io/)로 평가하였습니다. 성능 결과는 아래와 같습니다.   

| Model | F1(Dev Set 기준) |
|:---:|:---:|
| BiDAF (single) | 83% |
| DocQA (single) | 85.91% |
| BERT-Base, Multilingual Cased (single) | 89.9% |
| **BERT-Base, Korean Model(our model)** | 87.8% |
| **BERT-Large, Korean Model(our model)** | 00% |
| **XLNET-Base, Korean Model(our model)** | 00% |


<br>


## 기타
해당 모델은 한국어 Pre-training 모델 연구와 실험을 위한 한국어 XLNET Model입니다. 모델 사용시 반드시 출처를 밝혀주시길 바랍니다.
<br>
<br>
한국어 XLNET Model 학습이나 모델 사용 관련하여 궁금한 사항이 있으시면 oh31400@naver.com 메일로 문의 부탁립니다. 




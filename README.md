# XLNET-Korean-Model

* XLNET-Small, Korean Model : 6-layer, 6-heads

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
<code>data_utils.py</code>를 사용하여 <code>.tfrecord</code> 파일 형식으로 변환하였습니다. BERT 모델과 마찬가지로 학습 데이터의 구성은 한 줄에 한 문장씩 구성하고 Document 사이에는 빈 줄을 삽입할 것을 권장하고 있습니다. XLNET에서는 선택사항으로 특정 문장의 끝에 \<eop\> 토큰을 삽입하여 해당 문장이 단락 끝임을 표시하는 토큰을 넣을 수 있습니다. 그러나 본 모델에서는 해당 토큰을 사용하지 않고 한국어 BERT Model 학습시 사용했던 wiki 문서를 동일하게 사용했습니다. 
 
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
<br>

## Pretraining with XLNet
학습 데이터는 **한국어 위키데이터(2019.01 dump file, 약 350만 문장)** 을 사용하여 학습을 진행하였으며, 모델의 하이퍼파라미터는 서버 스펙에 맞춰 수정하였습니다.
<br>
<br>
총 학습 step은 논문과 동일하게 50만 step 진행하였습니다.

```
python train.py
  -—record_info_dir=$DATA/tfrecords \
  -—train_batch_size=64 \
  -—seq_len=512 \
  -—reuse_len=256 \
  -—mem_len=384 \
  -—perm_size=256 \
  -—n_layer=6 \
  -—d_model=768 \
  -—d_embed=768 \
  -—n_head=6 \
  -—d_head=64 \
  -—d_inner=3072 \
  -—untie_r=True \
  -—mask_alpha=6 \
  -—mask_beta=1 \
  -—num_predict=85 \
  --train_steps=500000 \
  --save_steps=100000
```   
<br>

## KorQuAD Task   
XLNET Model 성능 평가을 위해 한국어 SQuAD Task [KorQuAD](https://korquad.github.io/)로 평가를 진행하였습니다. XLNET github에는 SQuAD 2.0에 대한 평가 코드가 있어 이를 KorQuAD에 맞춰서 수정했습니다. 아래 KorQuAD 관련 Flag를 하나 추가하였으며, 관련 코드는 <code>run_korquad.py</code>를 참고하시길 바랍니다. 

```python
flags.DEFINE_bool("use_korquad", default= True, help="True when using Korquad, False if not")
```  
<br>

**1. KorQuAD 사전 데이터 준비** <br>
XLNET은 BERT와는 다르게 KorQuAD train 데이터를 이용해 tfrecord 파일을 만드는 전처리 과정이 필요합니다. 아래 스크립트를 실행하게 되면 <code>sp10m.cased.v3.model.0.slen-512.qlen-64.train.tf_record</code> 파일이 생성됩니다. 

```
SQUAD_DIR = data/squad
OUTPUT_DIR = data/output

python run_korquad.py \
  --use_tpu=False \
  --do_prepro=True \
  --spiece_model_file=${INIT_CKPT_DIR}/sp10m.cased.v3.model \
  --train_file=${SQUAD_DIR}/KorQuAD_v1.0_train.json \
  --output_dir=${OUTPUT_DIR} \
  --uncased=False \
  --max_seq_length=512 \
```
**2. KorQuAD Fine-tuning** <br>
TPU 환경에서 Fine-tuning을 진행했으며, 학습 파라미터는 아래 코드를 참고하시면 되겠습니다. 

* --init_checkpoint = XLNET pretrained 모델의 경로 
* --output_dir = 전처리 과정에서 만든 tfrecord 파일 경로
* --model_dir = Fine-tuning 과정에서 저장될 checkpoint 경로

```python
CONFIG_PATH = BUCKET_PATH + "/xlnet_model/config.json"
INIT_DIR = BUCKET_PATH+"/xlnet_model/model.ckpt"
OUTPUT_DIR = BUCKET_PATH+ "/output_file"
TF_DIR = BUCKET_PATH +"/tfrecord/"
TPU_NAME= "node-3"

# RUN Pre-Training
RUN_CMD = ("python3 run_korquad.py "
           "--use_tpu=True "
           "--tpu={} "
           "--num_hosts=1 "
           "--num_core_per_host=8 "
           
           "--model_config_path={} "
           "--spiece_model_file=sentence_model/sp10m.cased.v3.model "
           
           "--output_dir={} "
           "--init_checkpoint={} "
           "--model_dir={} "
           
           "--train_file=squad_data/KorQuAD_v1.0_train.json "
           "--predict_file=squad_data/KorQuAD_v1.0_dev.json "
           
           "--uncased=False "
           "--max_seq_length=512 "
           "--do_train=True "
           "--train_batch_size=48 "
           "--do_predict=True "
           "--predict_batch_size=32 "
           "--learning_rate=3e-5 "
           "--adam_epsilon=1e-6 "
           "--iterations=1000 "
           "--save_steps=1000 "
           "--train_steps=8000 "
           "--warmup_steps=1000 ")

RUN_CMD = RUN_CMD.format(TPU_NAME,
                         CONFIG_PATH,
			 TF_DIR,
                         INIT_DIR,
                         OUTPUT_DIR)
os.system(RUN_CMD)

```

<br>
<br>


## Step별 성능 평가  

| step | F1(KorQuAD Dev Set 기준) |
|:---:|:---:|
| **20만 step** | **83.21%** |
| 30만 step | 82.08% |
| 40만 step | 82.13% |
| 50만 step | 81.38% |


<br>

## 성능 평가  

| Model | F1(KorQuAD Dev Set 기준) |
|:---:|:---:|
| BERT-Base, Multilingual Cased (single) | 89.9% |
| **BERT-Base, Korean Model(our model)** | 87.8% |
| **XLNET-Base, Korean Model(our model)** | 83.21% |

<br>



## 기타
해당 모델은 한국어 Pre-training 모델 연구와 실험을 위한 한국어 XLNET Model입니다. 모델 사용시 반드시 출처를 밝혀주시길 바랍니다.
<br>
<br>
한국어 XLNET Model 학습이나 모델 사용 관련하여 궁금한 사항이 있으시면 oh31400@naver.com 메일로 문의 부탁립니다. 




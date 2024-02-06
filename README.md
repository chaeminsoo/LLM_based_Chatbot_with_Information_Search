# Chat-KoRani

<p align="center"><img width="150" alt="image" src="https://github.com/chaeminsoo/QnA_GPT/assets/79351899/7d65925a-e6e7-42ac-9c50-785d5646f1f0"></p>


## Contents
[1. Abstract](#1-abstract)  
[2. Model](#2-model)  
[3. Train Data](#3-train-data)  
[4. Service Architecture](#4-service-architecture)  
[5. Usage](#5-usage)  
[6. Details](#6-details)


## 1. Abstract

정보 탐색이 가능한 LLM 기반 Chatbot입니다.  
정보 탐색에는 RAG(Retrieval Augmented Generation) 기법이 적용되었습니다.  
일반적인 대화는 모델이 자체적으로 수행합니다.

검색을 요청하면, Chatbot이 스스로 구글에 검색하여 결과를 얻고, 이를 바탕으로 답변을 생성합니다.  
답변에 사용된 지식의 출처를 제공하기 때문에, 사용자가 직접 확인할 수 있습니다.

오픈 소스 LLM을 "한국어 대화"에 맞게 직접 Fine-tuning하여 사용했습니다.  
따라서, 한국어로 사용할 것을 권장합니다.


## 2. Model

<p align="center"><img width="150" alt="image" src="https://github.com/chaeminsoo/QnA_GPT/assets/79351899/b9a78bf5-0d80-435a-ba14-e288e8886f99"></p>

- EleutherAI의 polyglot-ko-5.8b를 기반으로 Instruction Tuning한 한국어 언어 모델
    - 네이버 지식인과 한국어로 번역된 ShareGPT 데이터로 구성된 약 10만개의 Instruction 데이터를 학습
- QLoRA 기법을 사용해 Fine-tuning 진행
    - LoRA 대비 약 30% 메모리 절약 
    - 전체 파라미터의 약 0.1195%만을 train
- DeepSpeed를 사용하여 Train 단계 최적화
    - Out of Memory Error 해결
    - 더 큰 Batch Size로 학습 허용
    - Training time 단축

<br>

※ 동원 가능한 GPU 및 메모리가 부족했기 때문에, 최대한 이를 절약하는 방식을 사용

## 3. Train Data

- 네이버 지식인 데이터 21155개 ([beomi님의 데이터 사용](https://huggingface.co/datasets/beomi/KoAlpaca-v1.1a))
- 한국어로 번역된 ShareGPT 데이터 84416개 ([junelee님의 데이터 사용](https://huggingface.co/datasets/junelee/sharegpt_deepl_ko))
- 두 데이터를 Instruction 형태로 변환하여 사용

## 4. Service Architecture

<p align="center"><img width="1558" alt="image" src="https://github.com/chaeminsoo/QnA_GPT/assets/79351899/b2d479c8-e930-48a5-ae84-3d148e614cf9"></p>

## 5. Usage
``` bash
pip install -r requirements.txt
```
``` bash
python app.py
```
<br>
<p align="center"><img width="1336" alt="스크린샷 2024-02-05 오후 9 25 28" src="https://github.com/chaeminsoo/QnA_GPT/assets/79351899/0302ce17-00ff-42f2-9a6f-274a01291aae">< 정보 탐색 ></p>

<br>

<p align="center"><img width="1291" alt="스크린샷 2024-02-05 오후 9 27 08" src="https://github.com/chaeminsoo/QnA_GPT/assets/79351899/3b0714c0-d2b0-4599-a4e7-147cb1ca92d6">< 일반 대화 ></p>
<br>


## 6. Details
- Author : [채민수](https://github.com/chaeminsoo)
- 언어 : Python 
- LLM : [KoRani-5.8b](https://huggingface.co/ChaeMs/KoRani-5.8b) (created by [채민수](https://github.com/chaeminsoo))
- GPU : A100 (1개)
- 작업 기간 : 2024.02.01 ~ 2024.02.06
# Chat-KoRani

<p align="center"><img width="150" alt="image" src="https://github.com/chaeminsoo/QnA_GPT/assets/79351899/7d65925a-e6e7-42ac-9c50-785d5646f1f0"></p>


## Contents
[1. Abstract](#1.-Abstract)  
[2. Model](#2.-Model)  
[3. Data](#3.-Data)  
[4. Architecture](#4.-Architecture)  
[5. Usage](#5.-Usage)  
[6. Details](#5.-Details)

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

- EleutherAI의 polyglot-ko-5.8b를 기반으로 Instruction Tuning한 모델
- 

## 3. Data



## 4. Architecture

<p align="center"></p>

## 5. Usage
``` bash
pip install -r requirements.txt
```
``` bash
python app.py
```
<br>
<p align="center"></p>
<br>


## 6. Details
- Author : [채민수](https://github.com/chaeminsoo)
- 언어 : Python 
- LLM : [KoRani-5.8b](https://huggingface.co/ChaeMs/KoRani-5.8b) (created by [채민수](https://github.com/chaeminsoo))
- GPU : A100 (1개)
- 작업 기간 : 2024.02.01 ~ 2024.02.06
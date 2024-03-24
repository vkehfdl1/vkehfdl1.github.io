---
layout: post
title: Facebook research의 contributor가 되었다.
date: 2024-03-24 14:50
description: 생각보다 쉬운 오픈소스 기여 경험 공유.
comments: true
tags:
- open-source
---
Facebook Research의 [Nougat](https://github.com/facebookresearch/nougat)라는 프로젝트의 contributor가 되었다! 그 동안 오탈자 수정 같은 조그마한 contribution을 한 적은 있지만, 어떤 하나의 기능?을 추가해 본적은 처음이다. 혹시나 오픈소스 기여에 대해서 고민하고 있는 모든 개발자들을 위하여, 나의 약소한 경험을 남겨본다.

![](https://velog.velcdn.com/images/vkehfdl1/post/373ef562-8e2e-4392-84bf-0e6b0bd66d0f/image.png)

# Nougat란?
nougat는 간단히 말해서 **논문용 OCR 모델**이다. nougat가 다른 많은 OCR 모델들에 비해서 awesome한 점은, mathpix markdown (이하 .mmd)이라는 파일로 변환을 해 주는 것이다. 이 .mmd는 논문 작성을 위해 특화된 마크다운이다. 논문의 표나 수식을 마크다운으로 표현할 수 있어, 마크다운의 논문용 확장자 느낌이 크다. 
일반적으로 pdf loader에 논문을 넣으면 페이지 수나 이것저것 복잡하게 나오고, 표는 아예 읽지 못하는데, nougat를 쓰면 .mmd를 사용해서 표와 수식까지도 깔끔하게 볼 수 있다! 

# Nougat 사용 계기
나는 현재 [RAGchain](https://github.com/NomaDamas/RAGchain)이라는 프레임워크를 열심히 개발하고 있다. 
이 RAGchain을 개발하는데, 더 높은 품질의 File Loader를 넣고 싶었다. 아무래도 파일을 로드하는 부분이 가장 앞부분에 있고, 표를 잘 못 읽는 문제를 몇달간 해결하고 싶었기 때문이다. 그래서 nougat를 우연히 접하게 되고, 곧장 RAGchain에 도입하겠다고 생각한 것이다.

# 그래서 본론
일단 nougat 모델 전체를 구동하기 위한 코드를 RAGchain에 포함시킬 생각은 전혀 없었다. RAGchain은 pypi에서 다운로드 받아서 사용하는데, nougat를 몽땅 포함시켜 버리면 그 크기라던가 부담이 엄청 커질 것 같았기 때문이다. OCR이 장비빨 (엔비디아 gpu)를 무조건 타기도 할 것이고, 많은 유저들이 OCR까지는 필요하지 않을 것이기 때문이기도 하다. 

그래서 api 서버를 만들려고 했는데, 고맙게도 이미 nougat에는 Fast API 서버가 있었다! 덕분에 서버를 직접 만드는 수고는 덜었다. 

이제 누구나 Nougat의 API 서버를 열어서, RAGchain에서 사용하기 쉽게 해야 하는데, 엔비디아 gpu를 사용하는 torch 환경을 구축하는 것은 여간 간단한 일이 아니다. 또한 배포도 어렵고. 그래서 docker를 써야겠다고 생각했는데 nougat 기본 레포에는 Dockerfile이 전혀 없었다!

**그래서 내가 만들었다**

[나의 PR](https://github.com/facebookresearch/nougat/pull/124)

![](https://velog.velcdn.com/images/vkehfdl1/post/97d256a4-f35c-4326-91bd-ac9894f5e9b6/image.png)

사실 Dockerfile을 만드는 일이 docker를 이리저리 해본 나에게는 그리 어려운 일이 아니었고, 원래부터 nvidia cuda를 쓰는 torch 환경의 Dockerfile을 직접 만들어서 빌드해놓고는 쓰고 있어서 시간이 오래 걸리지는 않았다. 
FastAPI가 잘 작동하는지 등등을 확인해야 했지만, 어짜피 RAGchain에서 써야 하니깐 재밌게 만들었다.
대신 README.md를 조금 열심히 적었는데, 이 부분은 우리의 인공지능을 활용했댜. Pycharm에 탑재된 AI Assistant가 굉장히 유능하더라.

아무튼 위에 사진처럼 다음 날 nougat를 만든 @lukas-blecher에게 LGTM을 선사받고 머지가 되었다! 

# 글을 마치며
멋진 고수 오픈소스 컨트리뷰터 분들이 하는 말을 유튜브 등지에서 들어보니, 회사나 하고 있는 프로젝트에서 사용하는 라이브러리부터 기여를 하라고 하셨다. 정말 그 말이 맞는 것 같다. 사실 Langchain을 쓰면서 괴상한 부분이 한 두 가지가 아닌데, 이런 부분들도 조금씩 기여해 봐야겠다. 

추가로, nougat를 이용해서 바로 RAG 시스템을 구축해보고 싶다면 [RAGchain](https://github.com/NomaDamas/RAGchain)을 한 번 써보라고 권하고 싶다. 한국인들이 개발한 것이다 보니, 한국어로 이슈를 남겨도 친절하게 답변을 남겨드릴 수 있다는 장점이 있다 ㅎㅎ 
더 많은 오픈소스 활동을 기원하며 글을 마치겠다. (특히 RAGchain에서!)

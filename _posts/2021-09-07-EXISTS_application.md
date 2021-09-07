---
layout: single
title: "EXISTS문 활용"
search: true
permalink: /SQL/
excerpt: "How to use EXISTS on Oracle DB."
last_modified_at: 2021-09-06T13:33:00+09:00
toc: true
categories:
  - SQL
  - Oracle
tags:
  - query
  - EXISTS
  - Tuning
---
# 실제로 `EXISTS()`가 필요한 상황

--------------------------------------------------------------------------------------------

앞선 포스팅에서는 `Oracle WHERE` 절에서 `EXISTS`가 `IN`과 유사하게 작동하나 속도가 빠르다 정도로 설명을 마쳤다.
실제로 같은 결과 값을 출력하도록 할 수도 있지만, 특정값 뭉치와 같은 값을 갖고 있는지를 비교를 하는 `IN`과 달리 `EXISTS()`는 특정값을 기준으로 범위 비교를 할 수 있다는 장점이 있다.

말로 설명하기 보다 예시 쿼리로 함께 설명하는 것이 더욱 와닿으므로 예시로 설명하겠다.

## [`EXISTS` 범위 비교]

※다음은 금액과 일자를 가진 내역 중, 연속한 달에 유사한 금액과 유사한 일자에 발생하였음을 알기 위해 작성한 QUERY이다.

```sql
SELECT a.date
     , a.amount
  FROM account a
 WHERE a.amount > 100000    /* 금액이 10만원 이상 */
   AND EXISTS (SELECT 'X'
                 FROM account b
                WHERE a.acountno = b.acountno  /* 계좌번호가 동일한 것 */
                  AND b.amount BETWEEN a.amount * 0.80
                                   AND b.amount * 1.20  /* 금액이 서로 ±20% 차이 */
                  AND b.date BETWEEN ADD_MONTHS(a.date, -1) - 5
                                 AND ADD_MONTHS(a.date, -1) + 5  /* 일자가 서로 ±5일 차이 */
               )
```

위와 같은 쿼리를 실행하면, `SELECT`되는 결과는 `WHERE`의 `EXISTS`문에 따라 두 달에 걸쳐 유사한 금액과 유사한 시기에 발생한 내역만이 남게 된다.

다만 위와 같이 범위 비교 QUERY는 `EXISTS`를 활용해 쿼리 성능을 올렸다고 하더라도, 아주 많은 연산이 따르게 된다. 이에 따라 빅데이터 수준으로 데이터를 관리하는 경우가 많은 요즘과 같은 상황에서 위와 같은 QUERY를 DB에 대고 무턱대고 질의하면 좋지 못한 결과를 얻게 될 확률이 높다.

당연한 말이지만, QUERY 성능이 매우 떨어질 수 있으니 이 경우, QUERY Tuning을 진행해야 한다.

## [`EXISTS`문 Tuning]

필자가 실제 상황에서 위의 상황에 맞닥뜨렸을 때 활용했던 방법은 두 가지였다.

1. `INDEX` 활용
2. `HASH_SJ` Hint 활용

### [`INDEX` 활용]

`INDEX`라는 개념을 제대로 활용해보지 못한 사람을 위해 간략하게 설명하자면, `INDEX` 는 RDBMS에서 검색 속도를 높이기 위한 기술이다. TABLE의 컬럼을 색인화(따로 파일로 저장하는 것을 일컬음)하여 검색 시 해당 TABLE의 record를 Full Scan하지 않고 색인화 되어있는 `INDEX` 파일을 검색하여 검색속도를 빠르게 한다.

자세한 설명은 추후에 `INDEX`와 관련된 포스팅을 따로 하여 다루도록 하겠다.

당장 위의 상황에서 `INDEX`를 잡으려고 하면 'account' table에 두 개의 `INDEX`가 필요하다.

`INDEX`가 없는 상태에서 위의 QUERY를 질의하면 해당 QUERY는 'account' table을 총 두 번 Full Scan하게 된다. 만일 EXISTS문을 반복해서 사용해야 하는 경우라면(필자는 12번의 EXISTS가 필요했던 상황이었다), 이 QUERY는 결과를 보기까지 영겁의 시간이 걸릴 것이다. Full Scan이 DB에게 얼마나 해로우며 우리 개발자에게 얼마나 고통을 주는 상황인지는 몇 번만 DB를 다뤄본 사람이면 동감할 수 있을 것이다. 이와 관련된 내용도 추후 `INDEX`를 다룰 때 함께 하도록 하겠다.

잠시 고통의 시간이 생각나 말이 길어졌지만 각설하고, 아래는 `EXISTS`문의 `INDEX` 잡는 방법을 직접 위에서 본 동일한 QUERY와 함께 설명하겠다. 편의를 위해 아래에 다시 첨부한다.

```sql
SELECT a.date
     , a.amount
  FROM account a
 WHERE a.amount > 100000    /* 금액이 10만원 이상 */
```

QUERY를 천천히 읽어보면, `WHERE`문에서 account TABLE의 amount 컬럼에 10만원 이상의 조건을 달아놓은 것을 볼 수 있다. 이는 `EXISTS`문이 실행되기 전에 `EXISTS`문에 `SELECT`될 레코드를 최소한으로 줄이기 위한 과정이다. 다른 말로 하자면, QUERY할 모수를 줄였다고 할 수 있다.

그러나, 이 또한 범위 비교를 실행하게 되기 때문에 해당 컬럼이 `INDEX`를 잡아놓지 않은 컬럼이라면 필연적으로 성능 저하가 일어나게 된다. 그러므로 필자는 이 상황에서 account TABLE의 amount 컬럼을 첫번째 `INDEX`로 잡았다.

```sql
   AND EXISTS (SELECT 'X'
                 FROM account b
                WHERE a.acountno = b.acountno  /* 계좌번호가 동일한 것 */
                  AND b.amount BETWEEN a.amount * 0.80
                                   AND b.amount * 1.20  /* 금액이 서로 ±20% 차이 */
                  AND b.date BETWEEN ADD_MONTHS(a.date, -1) - 5
                                 AND ADD_MONTHS(a.date, -1) + 5  /* 일자가 서로 ±5일 차이 */
                )
```

위의 EXISTS 문에서는 무자비한(?) 범위 비교가 실행된다. 다소 DB한테 미안하지만, 조금이나마 덜 미안하기 위해 `INDEX`를 잡아주자. 필자는 위의 상황에서 acountno, amount, date 컬럼에 대해 두번째 `INDEX`를 잡아주었다.

### [`HASH_SJ` Hint 활용]

사실 필자는 `INDEX`만 잡아주면 이 혼란한 상황이 종결될 줄 알았지만, Tuning의 세상은 그리 호락호락하지 않았다.

그래서 결국 `EXISTS`문에 Hint를 활용하였는데, 활용한 힌트는 `HASH_SJ`이다. QUERY에 익숙하지 않은 사람이라면 Hint 개념 또한 익숙지 않을 것이다. Hint는 QUERY 문에서 `/*+ hint */` 이와 같은 형태로 사용하게 된다.

힌트와 관련된 설명 또한 해당 포스팅에서 언급하기에 너무 포스팅이 길어질 것 같아, 다음에 한번 더 정리하도록 하겠다.

이번 포스팅 EXISTS문의 최종 형태 다음과 같다. Hint 위치가 중요하니 체크하면 좋을 것 같다.

```sql
SELECT a.date
     , a.amount
  FROM account a
 /* INDEX 1 */
 WHERE a.amount > 100000    /* 금액이 10만원 이상 */
   AND EXISTS (SELECT 'X' /*+ HASH_SJ */
                 FROM account b
                /* INDEX 2 */
                WHERE a.acountno = b.acountno  /* 계좌번호가 동일한 것 */
                  AND b.amount BETWEEN a.amount * 0.80
                                   AND b.amount * 1.20  /* 금액이 서로 ±20% 차이 */
                  AND b.date BETWEEN ADD_MONTHS(a.date, -1) - 5
                                 AND ADD_MONTHS(a.date, -1) + 5  /* 일자가 서로 ±5일 차이 */
                )
```

위의 두 Tuning을 진행하였을 때, 필자는 반나절 혹은 멈추지 않던 QUERY가 10분 내로 결과를 표출하도록 성능 개선을 얻을 수 있었다. 필자가 QUERY한 TABLE의 크기는 레코드가 총 2.5억 건이었으며, 첫 번째 `WHERE` 절에서 전체 모수에서 400만건 정도 SELECT되도록 상황을 설계하여 QUERY를 진행하였다.

지금까지 `EXISTS` 문과 관련된 포스팅을 읽어 주셔서 감사드리며, 혹 포스팅에서 궁금한 사항이나 잘못된 부분이 있다면 언제든지 필자에게 연락해 주시길 바랍니다 :)

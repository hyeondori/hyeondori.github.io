---
layout: single
title: "Oracle로 달력을 출력하고 날짜 계산을 해보자"
search: true
last_modified_at: 2021-09-09T17:25:00+09:00
toc: true
categories:
  - SQL
tags:
  - query
  - function
---
QUERY문을 작성하다보면, 가끔 달력을 만들어 활용할 때가 있다.

자주 쓰지 않기 때문에 필요로 할 때마다 찾아서 다시 쓰곤 하여 정리해본다.

달력을 쓸 때는 보통 날짜 계산을 해야하는 경우가 많기 때문에, 날짜 함수와 함께 정리하겠다.

## Oracle 달력 출력

Oracle 쿼리를 통해 달력을 필요로 할 때는 대개 다음과 같은 요소가 필요할 때가 많다.

* 해당 월의 시작일

* 해당 월의 말일

* 각 일자들의 요일(숫자 or 문자)

* 각 일자의 주차 수

이에 따라 이 요소들이 담긴 QUERY를 다음과 같이 작성할 수 있다.

사실 달력을 만드는 방법은 굉장히 다양하지만, 그 중에 하나만 설명하도록 하겠다.

~~어찌됐건 우리는 결과만 잘 나오면 된다.~~

사실 달력을 만드는 자체보다 달력에서 어떤 일자를 통해 어떤 데이터와 `JOIN`해서 활용하는 등의 작업이 더욱 중요하다.

일자를 계산하다보면 생각보다 꽤 골머리를 싸매는 경우가 많다(웃음).

```sql
SELECT TO_CHAR(FIRST_DAY + LEVEL - 1, 'YYYY-MM-DD') DAY     /* 해당일자 */
     , TO_CHAR(FIRST_DAY + LEVEL - 1, 'YYYYMMDD') DAY_STR   /* 해당일자_STRING */
     , TO_CHAR(FIRST_DAY + LEVEL - 1,'IW') WEEK_NUM         /* 주차수 */
     , TO_CHAR(FIRST_DAY + LEVEL - 1,'DY') DAYS_K           /* 요일(한글) */
     , TO_CHAR(FIRST_DAY + LEVEL - 1, 'D') DAYS             /* 요일(숫자) */
  FROM (SELECT TRUNC(SYSDATE, 'MM') FIRST_DAY
          FROM DUAL)
CONNECT BY FIRST_DAY + LEVEL - 1 <= LAST_DAY(SYSDATE)       /* 출력하고 싶은 날짜를 지정 */
```

결과는 다음과 같다.

<img src="\assets\images\typora-user-images\image-20210909165518832.png" alt="image-20210909165518832" style="zoom:120%;" />

날짜 형태의 일자, 문자 형태의 일자, 주차수, 요일(한글, 숫자)이 예쁘게 잘 뽑힌 것을 볼 수 있다.

날짜의 경우 계산을 필요로 할 때는 문자 형태일 때보다 날짜 형태로 계산하는 것이 더 적절한 상황이 많으나,

본인이 필요한 상황에 맞추어 써보도록 하자.

여기서 중요한 점은, 본인이 출력하고 싶은 날짜 구간을 바꿀 수 있다는 것이다.

예시에서는 해당 월의 처음부터 말일까지를 출력했지만, 현재 상황이 현재 날짜로 부터 세달 후를 필요로 한다면 다음과 같이 수정하면 된다.

```sql
SELECT TO_CHAR(SYSDATE + LEVEL - 1, 'YYYY-MM-DD') DAY     /* 해당일자 */
     , TO_CHAR(SYSDATE + LEVEL - 1, 'YYYYMMDD') DAY_STR   /* 해당일자_STRING */
     , TO_CHAR(SYSDATE + LEVEL - 1,'IW') WEEK_NUM         /* 주차수 */
     , TO_CHAR(SYSDATE + LEVEL - 1,'DY') DAYS_K           /* 요일(한글) */
     , TO_CHAR(SYSDATE + LEVEL - 1, 'D') DAYS             /* 요일(숫자) */
  FROM (SELECT TRUNC(SYSDATE, 'MM')
          FROM DUAL)
CONNECT BY SYSDATE + LEVEL - 1 <= ADD_MONTHS(SYSDATE, +3) /* 출력하고 싶은 날짜를 지정 */
```

작성일 기준으로 2021년 9월 9일이므로 지금으로 부터 3달 후까지의 달력이라 하면, 31일인 달이 2번 끼어있어 92개의 데이터 컬럼이 출력되는 것을 볼 수 있다.

![image-20210909171446601](\assets\images\typora-user-images\image-20210909171446601.png)

데이터를 다루면서 날짜 계산을 하다보면 알면서도 놓치기 쉬운 실수가 바로 한달이 30일일 것이라는 생각이므로 한번쯤 유념하고 넘어가도록 하자.

자, 그럼 위의 QUERY에서 주목할 점은 `ADD_MONTHS()`함수와 같은 함수들이다.

지금부터 날짜와 관련된 함수를 한번 짚어보고 포스팅을 마무리하겠다.

## Oracle 날짜 관련 함수 모음

날짜 함수를 쓰다보면 날짜 Data Format을 접하게 되는데 이게 처음보면 꽤 헷갈린다.

한번쯤 보고 가도록 하자. 자주 쓰게 되는 Format은 역시 YYYYMMDD 혹은 YYYYMMDDHH24MISS 이다.

| FORMAT | Description                   |
| ------ | ----------------------------- |
| YYYY   | 네자리 연도 (Ex. 2021)        |
| YY     | 두자리 연도 (Ex. 21)          |
| D      | Day of week (1-7)             |
| DAY    | 요일(월-일)                   |
| DD     | Day of month (1-31)           |
| DDD    | Day of year (1-366)           |
| MM     | 해당월 두자리로 표현          |
| MONTH  | Name of month                 |
| WW     | Week of month (1-53)          |
| IW     | Week of year (1-53) 국제 표준 |
| W      | Week of month (1-5)           |
| HH24   | Hour of day (0-23)            |
| HH     | Hour of day (1-12)            |
| MI     | Minute (0-59)                 |
| SS     | Second (0-59)                 |

### `TO_DATE(char or number, format)`

`TO_DATE` 함수는 CHAR, VARCHAR2형을 DATE 타입으로 변환한다.

위의 표에서 'W', 'WW' Format을 제외한 나머지는 TO_DATE 함수의 format으로 사용 할 수 있다.

### `TO_CHAR(date or number, format)`

`TO_CHAR` 함수는 date형, number형 데이터를 문자 타입으로 변환하는 함수이다.

Format에 유의해서 변환을 실수하지 않도록 하자.

### `ADD_MONTHS(date, integer)`

`ADD_MONTHS`함수는 변수 date에 integer만큼의 월을 더한 날짜를 리턴한다.

integer 부분에 음수를 넣어 과거의 날짜도 계산할 수 있으니 잘 활용하면 좋은 함수이다.

### `LAST_DATE(date)`

`LAST_DATE`함수는 현재 월의 마지막 일자를 리턴한다.

달력 계산을 하다보면 말일을 활용할 때가 많은데 해당 함수를 활용해보자.

### `MONTHS_BETWEEN(date1, date2)`

`MONTHS_BETWEEN`함수는 date1과 date2 사이의 개월 수를 리턴한다.

날짜 사이의 개월 수 차이를 구할 때 유용한 함수이다.



지금까지 Oracle에서 달력을 출력하는 방법과 날짜 계산을 위한 유용한 함수에 관한 포스팅이었습니다.

작게나마 도움이 되었길 바라며, 궁금한 사항은 언제든지 연락 바랍니다 :)
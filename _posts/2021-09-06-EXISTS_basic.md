---
layout: single
title: "EXISTS문 기본"
search: true
permalink: /SQL/Oracle
excerpt: "How to use EXISTS on Oracle DB."
last_modified_at: 2021-09-06T13:33:00+09:00
toc: true
categories:
  - SQL
  - Oracle
tags:
  - query
  - EXISTS
  - IN
  - Tuning
---
Oracle에서의 EXISTS()
====================

EXISTS()함수는 WHERE 절 활용 중 IN을 사용하고자 할 때, 더욱 효과적인 성능을 기대하기 위해 활용되는 경우가 많다.
EXISTS(SUBQUERY)는 SUBQUERY의 결과가 QUERY하는 TABLE 중 한 건이라도 존재한다면 TRUE를, 없으면 FALSE를 리턴하는 함수이다.
EXISTS문이 종종 IN으로 조건을 거는 것에 비해 효과적인 이유는 EXISTS문 수행 중에 SUBQUERY의 WHERE 조건에 TRUE를 리턴하는 ROW가 한 건이라도 존재하면, 더 이상 QUERY를 수행하지 않기 때문이다.

TABLE 크기가 작은 경우에는 서로간의 성능 차이가 극명히 나타나지 않겠지만, 최소 몇백만건 이상의 DATA를 QUERY하게 될 때 해당 함수의 진가를 발휘하게 된다.

**※ NOT EXISTS**

EXISTS문의 부정은 NOT EXISTS로 활용할 수 있다.
그러나, 이 때 주의할 점은 EXISTS문 내의 SUBQUERY에 따라 EXISTS문이 빠른 성능을 내었더라도 NOT EXISTS문에서 동일하게 우수한 성능을 기대하기는 어렵다.
이는 SUBQUERY의 결과의 TRUE/FALSE 비율에 따라 달라지는 것으로 보인다.

본론으로 다시 돌아오면, EXISTS문은 WHERE문과 CASE문에서 활용할 수 있다.

예시 QUERY는 다음과 같다.

[WHERE문 EXISTS/NOT EXISTS 활용]
-------------------
```oracle
SELECT a.empno
     , a.ename
     , a.deptno
  FROM emp a
 WHERE a.job = 'MANAGER'
   AND EXISTS (SELECT 'X'
                 FROM dept_history aa
                WHERE aa.empno = a.empno)
```
```oracle
SELECT a.empno
     , a.ename
     , a.deptno
  FROM emp a
 WHERE a.job = 'MANAGER'
   AND NOT EXISTS (SELECT 'X'
                     FROM dept_history aa
                    WHERE aa.empno = a.empno)
```

여기서 EXISTS문을 처음 접하는 사람의 경우 SELECT 문 뒤의 'X'가 다소 낯설게 느껴질텐데,
EXISTS문의 경우 WHERE절의 조건에 맞는 컬럼이 존재하는지를 비교하기 때문에 딱히 SELECT문 뒤에 컬럼을 필요로 하지 않는다.
그렇기 때문에 의미없는 값을 넣어주곤 하는데, 필자는 'X'를 기입하는 것을 선호한다.

[CASE문 EXISTS 활용]
-------------------
```oracle
SELECT a.empno
     , a.ename
     , a.job
     , a.deptno
     , CASE WHEN EXISTS (SELECT 'X'
                           FROM dept_history aa
                          WHERE aa.empno = a.empno) THEN 'Y'
             END dept_history_yn
  FROM emp a
 WHERE a.job = 'MANAGER'
```

CASE문에서는 위와 같이 활용할 수 있다.
그러나, CASE문에서 EXISTS를 사용하면 쿼리가 복잡해지므로 가능하면 유사한 다른 문법을 활용하기를 바란다.

[IN과의 비교]
-------------------
위 WHERE문에서의 EXISTS 활용을 IN을 활용하여 작성하면 다음과 같다.
```oracle
SELECT a.empno
     , a.ename
     , a.deptno
  FROM emp a
 WHERE a.job = 'MANAGER'
   AND a.empno IN (SELECT aa.empno
                     FROM dept_history aa
                    WHERE aa.empno = a.empno)
```
조회해보면 EXISTS와 IN에 의한 결과는 동일하게 나타난다.
위에서 말했듯 IN을 활용하게 되면 SUBQUERY 결과를 모두 수행하게 되므로,
조건에 일치하는 결과가 1건이라도 있으면 수행을 멈추고 결과를 리턴하는 EXISTS문에 비해 성능이 다소 떨어진다.
그렇기 때문에 대용량의 데이터를 조회할 때, IN을 활용해야 하는 상황이라면 EXISTS를 활용해 보는 것을 추천한다.

[JOIN과의 비교]
-------------------
EXISTS문을 사용하다 보면, JOIN문과 크게 다르지 않다는 생각이 들 수 있다.
실제로 작동하는 원리도 비슷하기 때문에 혼동할 수 있으나
EXISTS문에 사용한 SUBQUERY를 JOIN문으로 사용해보면 중복된 데이터가 있을 시, 여러건의 데이터가 표출될 수 있으니 주의하는 것이 좋다.
JOIN을 사용하는 것이 성능적으로 좋을 수는 있으나 MAINQUERY와 SUBQUERY의 데이터가 1:1일 때만 활용하는 것이 좋다.
# Controlled Agent vs Day5.5 Diff

This report compares saved Day5.5 strict-repair predictions with saved controlled-agent traces on the same sample set.

## Summary

- Total: `50`
- same_ex_count: `48`
- day5_only_correct_count: `2`
- controlled_only_correct_count: `0`
- controlled_execution_failed_count: `1`
- initial_sql_mismatch_count: `13`
- final_sql_mismatch_count: `13`
- repair_success_mismatch_count: `1`

Difference cause counts:
- initial SQL mismatch: `13`
- repair SQL mismatch: `3`
- execution result mismatch: `6`
- parsing / extract_sql mismatch: `0`
- final SQL field selection mismatch: `0`

Controlled metrics snapshot:
- EX: `0.32`
- VSR: `0.98`
- repair_attempt_count: `6`
- repair_success_count: `5`

## Day5-Only Correct Cases

### Sample 27

- Difficulty: `moderate`
- Status: `day5_only_correct`
- Causes: `initial SQL mismatch; execution result mismatch`
- Day5 pred_success / EX: `True` / `True`
- Controlled pred_success / EX: `True` / `False`

Question: What is the average score in writing for the schools that were opened after 1991 or closed before 2000? List the school names along with the score. Also, list the communication number of the schools if there is any.

Day5 final SQL:
```sql
SELECT s.`School`, sats.`AvgScrWrite`, s.`Phone` AS `communication_number`
FROM schools s
LEFT JOIN satscores sats ON s.`CDSCode` = sats.`cds`
WHERE (s.`OpenDate` > '1991-12-31' OR s.`ClosedDate` < '2000-01-01');
```

Controlled final SQL:
```sql
SELECT s.`School`, s.`Phone`, sa.`AvgScrWrite`
FROM schools s
LEFT JOIN satscores sa ON s.`CDSCode` = sa.`cds`
WHERE (s.`OpenDate` > '1991-12-31' OR s.`ClosedDate` < '2000-01-01')
  AND sa.`AvgScrWrite` IS NOT NULL;
```

### Sample 41

- Difficulty: `simple`
- Status: `day5_only_correct`
- Causes: `initial SQL mismatch; repair SQL mismatch; execution result mismatch`
- Day5 pred_success / EX: `True` / `True`
- Controlled pred_success / EX: `True` / `False`

Question: List the names of virtual schools that are among the top 5 in their respective counties based on average reading scores.

Day5 final SQL:
```sql
SELECT s.School
FROM satscores AS sc
JOIN schools AS s ON sc.cds = s.CDSCode
WHERE s.Virtual = 'F'
  AND (
    SELECT COUNT(*)
    FROM satscores AS sc2
    JOIN schools AS s2 ON sc2.cds = s2.CDSCode
    WHERE s2.Virtual = 'F'
      AND s2.County = s.County
      AND (sc2.AvgScrRead > sc.AvgScrRead OR (sc2.AvgScrRead = sc.AvgScrRead AND sc2.cds < sc.cds))
  ) < 5;
```

Controlled final SQL:
```sql
SELECT sc.sname
FROM satscores sc
JOIN schools s ON sc.cds = s.CDSCode
WHERE s.Virtual = 'F'
AND (
    SELECT COUNT(*)
    FROM satscores sc2
    JOIN schools s2 ON sc2.cds = s2.CDSCode
    WHERE s2.Virtual = 'F'
    AND s2.County = s.County
    AND (sc2.AvgScrRead > sc.AvgScrRead OR (sc2.AvgScrRead = sc.AvgScrRead AND sc2.cds < sc.cds))
) < 5;
```

## Controlled Execution Failures

### Sample 32

- Difficulty: `challenging`
- Status: `controlled_execution_failed`
- Causes: `initial SQL mismatch; repair SQL mismatch; execution result mismatch`
- Day5 pred_success / EX: `True` / `False`
- Controlled pred_success / EX: `False` / `False`

Question: For the top 5 schools with the highest number of students eligible for free or reduced price meals in grades K-12 among schools with ownership code 66, what are their eligibility rates, SAT performance metrics, and how do they compare in terms of SAT participation and high scorer rates?

Day5 final SQL:
```sql
SELECT 
    sc.School AS School_Name,
    f.`FRPM Count (K-12)` AS FRPM_Count,
    f.`Enrollment (K-12)` AS Enrollment,
    (f.`FRPM Count (K-12)` / f.`Enrollment (K-12)`) AS Eligibility_Rate,
    sat.NumTstTakr AS SAT_Takers,
    sat.enroll12 AS Enrollment_12,
    (sat.NumTstTakr * 1.0 / sat.enroll12) AS SAT_Participation_Rate,
    sat.NumGE1500 AS High_Scorers,
    (sat.NumGE1500 * 1.0 / sat.NumTstTakr) AS High_Scorer_Rate
FROM schools sc
JOIN frpm f ON sc.CDSCode = f.CDSCode
JOIN satscores sat ON sc.CDSCode = sat.cds
WHERE sc.SOC = '66'
ORDER BY f.`FRPM Count (K-12)` DESC
LIMIT 5;
```

Controlled final SQL:
```sql
-- empty --
```

## Repair-Related Cases

### Sample 5

- Difficulty: `challenging`
- Status: `both_wrong`
- Causes: `none identified`
- Day5 pred_success / EX: `True` / `False`
- Controlled pred_success / EX: `True` / `False`

Question: What are the details of fully virtual schools that have an average SAT Math score above 400, including their SAT performance rankings, enrollment, and poverty levels, ordered by their total SAT scores?

Day5 final SQL:
```sql
SELECT s.`School` AS school_name, 
       s.`City`, 
       s.`Zip`, 
       sa.`AvgScrMath`, 
       sa.`AvgScrRead`, 
       sa.`AvgScrWrite`, 
       (sa.`AvgScrRead` + sa.`AvgScrMath` + sa.`AvgScrWrite`) AS total_sat_score,
       sa.`NumTstTakr`, 
       sa.`NumGE1500`, 
       CAST(sa.`NumGE1500` AS REAL) / sa.`NumTstTakr` AS excellence_rate,
       f.`Enrollment (K-12)`, 
       f.`Percent (%) Eligible FRPM (K-12)` AS poverty_level
FROM schools s
JOIN satscores sa ON s.`CDSCode` = sa.`cds`
JOIN frpm f ON s.`CDSCode` = f.`CDSCode`
WHERE s.`Virtual` = 'F' 
  AND sa.`AvgScrMath` > 400
ORDER BY total_sat_score DESC;
```

Controlled final SQL:
```sql
SELECT s.`School` AS school_name, 
       s.`City`, 
       s.`Zip`, 
       sa.`AvgScrMath`, 
       sa.`AvgScrRead`, 
       sa.`AvgScrWrite`, 
       (sa.`AvgScrRead` + sa.`AvgScrMath` + sa.`AvgScrWrite`) AS total_sat_score,
       sa.`NumTstTakr`, 
       sa.`NumGE1500`, 
       CAST(sa.`NumGE1500` AS REAL) / sa.`NumTstTakr` AS excellence_rate,
       f.`Enrollment (K-12)`, 
       f.`Percent (%) Eligible FRPM (K-12)` AS poverty_level
FROM schools s
JOIN satscores sa ON s.`CDSCode` = sa.`cds`
JOIN frpm f ON s.`CDSCode` = f.`CDSCode`
WHERE s.`Virtual` = 'F' 
  AND sa.`AvgScrMath` > 400
ORDER BY total_sat_score DESC;
```

### Sample 6

- Difficulty: `challenging`
- Status: `both_wrong`
- Causes: `none identified`
- Day5 pred_success / EX: `True` / `False`
- Controlled pred_success / EX: `True` / `False`

Question: For magnet schools with over 500 SAT test takers, provide a comprehensive performance analysis including their SAT scores, rankings, poverty levels, and performance categories, sorted by total average SAT scores.

Day5 final SQL:
```sql
SELECT 
    sc.sname AS School_Name,
    sc.AvgScrRead,
    sc.AvgScrMath,
    sc.AvgScrWrite,
    (sc.AvgScrRead + sc.AvgScrMath + sc.AvgScrWrite) AS Total_Avg_SAT,
    sc.NumTstTakr,
    sc.NumGE1500,
    (sc.NumGE1500 * 1.0 / sc.NumTstTakr) AS Excellence_Rate,
    f.`Percent (%) Eligible FRPM (K-12)` AS Poverty_Level,
    CASE 
        WHEN f.`Percent (%) Eligible FRPM (K-12)` >= 0.75 THEN 'High Poverty'
        WHEN f.`Percent (%) Eligible FRPM (K-12)` >= 0.5 THEN 'Medium Poverty'
        ELSE 'Low Poverty'
    END AS Poverty_Category,
    CASE 
        WHEN (sc.AvgScrRead + sc.AvgScrMath + sc.AvgScrWrite) >= 1500 THEN 'High Performance'
        WHEN (sc.AvgScrRead + sc.AvgScrMath + sc.AvgScrWrite) >= 1200 THEN 'Medium Performance'
        ELSE 'Low Performance'
    END AS Performance_Category
FROM satscores sc
JOIN schools s ON sc.cds = s.CDSCode
LEFT JOIN frpm f ON s.CDSCode = f.CDSCode
WHERE s.Magnet = 1 AND sc.NumTstTakr > 500
ORDER BY Total_Avg_SAT DESC;
```

Controlled final SQL:
```sql
SELECT 
    sc.sname AS School_Name,
    sc.AvgScrRead,
    sc.AvgScrMath,
    sc.AvgScrWrite,
    (sc.AvgScrRead + sc.AvgScrMath + sc.AvgScrWrite) AS Total_Avg_SAT,
    sc.NumTstTakr,
    sc.NumGE1500,
    (sc.NumGE1500 * 1.0 / sc.NumTstTakr) AS Excellence_Rate,
    f.`Percent (%) Eligible FRPM (K-12)` AS Poverty_Level,
    CASE 
        WHEN f.`Percent (%) Eligible FRPM (K-12)` >= 0.75 THEN 'High Poverty'
        WHEN f.`Percent (%) Eligible FRPM (K-12)` >= 0.5 THEN 'Medium Poverty'
        ELSE 'Low Poverty'
    END AS Poverty_Category,
    CASE 
        WHEN (sc.AvgScrRead + sc.AvgScrMath + sc.AvgScrWrite) >= 1500 THEN 'High Performance'
        WHEN (sc.AvgScrRead + sc.AvgScrMath + sc.AvgScrWrite) >= 1200 THEN 'Medium Performance'
        ELSE 'Low Performance'
    END AS Performance_Category
FROM satscores sc
JOIN schools s ON sc.cds = s.CDSCode
LEFT JOIN frpm f ON s.CDSCode = f.CDSCode
WHERE s.Magnet = 1 AND sc.NumTstTakr > 500
ORDER BY Total_Avg_SAT DESC;
```

### Sample 19

- Difficulty: `challenging`
- Status: `sql_mismatch`
- Causes: `initial SQL mismatch; repair SQL mismatch`
- Day5 pred_success / EX: `True` / `False`
- Controlled pred_success / EX: `True` / `False`

Question: For the school with the highest average math SAT score among active schools with at least 10 test takers, provide its contact information, all SAT scores, charter status, enrollment, and percentage of students eligible for free or reduced price meals.

Day5 final SQL:
```sql
SELECT s.School, s.Phone, s.Website, 
       sa.AvgScrRead, sa.AvgScrMath, sa.AvgScrWrite, sa.NumTstTakr, sa.NumGE1500,
       s.Charter AS charter_status,
       f.`Enrollment (K-12)`, f.`Percent (%) Eligible FRPM (K-12)`
FROM satscores sa
JOIN schools s ON sa.cds = s.CDSCode
JOIN frpm f ON s.CDSCode = f.CDSCode
JOIN (
    SELECT sa2.cds
    FROM satscores sa2
    JOIN schools s2 ON sa2.cds = s2.CDSCode
    WHERE s2.StatusType = 'Active' AND sa2.NumTstTakr >= 10
    ORDER BY sa2.AvgScrMath DESC
    LIMIT 1
) top ON sa.cds = top.cds;
```

Controlled final SQL:
```sql
SELECT s.School, s.Phone, s.Website, 
       sa.AvgScrRead, sa.AvgScrMath, sa.AvgScrWrite, sa.NumTstTakr, sa.NumGE1500,
       sc.Charter AS charter_status,
       f.`Enrollment (K-12)`, f.`Percent (%) Eligible FRPM (K-12)`
FROM satscores sa
JOIN schools s ON sa.cds = s.CDSCode
JOIN frpm f ON s.CDSCode = f.CDSCode
JOIN schools sc ON s.CDSCode = sc.CDSCode
WHERE s.StatusType = 'Active'
  AND sa.NumTstTakr >= 10
ORDER BY sa.AvgScrMath DESC
LIMIT 1;
```


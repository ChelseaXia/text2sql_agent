# Controlled Agent Trace Examples

Stepwise traces for the controlled execution-repair agent.

## Sample 4

- Difficulty: `moderate`
- EX: `True`
- Repair attempted: `False`
- Repair success: `False`

Final SQL:
```sql
SELECT s.Phone
FROM schools s
JOIN frpm f ON s.CDSCode = f.CDSCode
WHERE f.`Charter Funding Type` = 'Directly funded'
  AND s.OpenDate > '2000-01-01';
```

Trace summary:
- Step 1 `retrieve_schema`: Selected tables: schools, frpm.
- Step 2 `generate_sql`: Generated initial SQL candidate.
- Step 3 `execute_sql`: Initial SQL executed successfully.
- Step 6 `finish`: Final SQL selected for evaluation.

## Sample 5

- Difficulty: `challenging`
- EX: `False`
- Repair attempted: `True`
- Repair success: `True`

Final SQL:
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

Trace summary:
- Step 1 `retrieve_schema`: Selected tables: satscores, schools, frpm.
- Step 2 `generate_sql`: Generated initial SQL candidate.
- Step 3 `execute_sql`: Initial SQL failed: no such column: s.sname
- Step 4 `repair_sql`: Generated repaired SQL candidate.
- Step 5 `execute_sql`: Repaired SQL executed successfully.
- Step 6 `finish`: Final SQL selected for evaluation.


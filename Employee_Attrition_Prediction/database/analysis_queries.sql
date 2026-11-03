-- =============================================================
-- analysis_queries.sql
-- 10+ Meaningful Analytical SQL Queries
-- Database: employee_attrition_db
-- =============================================================

USE employee_attrition_db;

-- -----------------------------------------------------------------
-- Q1: Overall attrition count and rate
-- -----------------------------------------------------------------
SELECT
    attrition,
    COUNT(*)                                         AS employee_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) AS percentage
FROM Employees
GROUP BY attrition
ORDER BY employee_count DESC;

-- -----------------------------------------------------------------
-- Q2: Department-wise attrition rate (JOIN + GROUP BY + HAVING)
-- -----------------------------------------------------------------
SELECT
    d.department_name,
    COUNT(*)                                               AS total,
    SUM(e.attrition = 'Yes')                               AS attrited,
    ROUND(SUM(e.attrition = 'Yes') * 100.0 / COUNT(*), 2) AS attrition_rate_pct
FROM Employees e
INNER JOIN Departments d ON e.department_id = d.department_id
GROUP BY d.department_name
HAVING attrition_rate_pct > 0
ORDER BY attrition_rate_pct DESC;

-- -----------------------------------------------------------------
-- Q3: Average monthly income by job role (JOIN + AVG)
-- -----------------------------------------------------------------
SELECT
    r.job_role_name,
    ROUND(AVG(c.monthly_income), 2)   AS avg_income,
    MIN(c.monthly_income)             AS min_income,
    MAX(c.monthly_income)             AS max_income,
    COUNT(*)                          AS headcount
FROM Employees e
INNER JOIN Job_Roles    r ON e.job_role_id = r.job_role_id
INNER JOIN Compensation c ON e.employee_id = c.employee_id
GROUP BY r.job_role_name
ORDER BY avg_income DESC;

-- -----------------------------------------------------------------
-- Q4: Overtime vs attrition (GROUP BY + aggregate)
-- -----------------------------------------------------------------
SELECT
    over_time,
    COUNT(*)                                               AS total,
    SUM(attrition = 'Yes')                                 AS attrited,
    ROUND(SUM(attrition = 'Yes') * 100.0 / COUNT(*), 2)   AS attrition_rate_pct
FROM Employees
GROUP BY over_time
ORDER BY attrition_rate_pct DESC;

-- -----------------------------------------------------------------
-- Q5: Job satisfaction distribution among attrited employees
-- -----------------------------------------------------------------
SELECT
    job_satisfaction,
    COUNT(*)                          AS total_attrited
FROM Employees
WHERE attrition = 'Yes'
GROUP BY job_satisfaction
ORDER BY job_satisfaction;

-- -----------------------------------------------------------------
-- Q6: Top 10 highest-paid employees who left (ORDER BY + LIMIT)
-- -----------------------------------------------------------------
SELECT
    e.emp_number,
    e.age,
    d.department_name,
    r.job_role_name,
    c.monthly_income
FROM Employees e
INNER JOIN Departments  d ON e.department_id = d.department_id
INNER JOIN Job_Roles    r ON e.job_role_id   = r.job_role_id
INNER JOIN Compensation c ON e.employee_id   = c.employee_id
WHERE e.attrition = 'Yes'
ORDER BY c.monthly_income DESC
LIMIT 10;

-- -----------------------------------------------------------------
-- Q7: Subquery – employees earning below department average who left
-- -----------------------------------------------------------------
SELECT
    e.emp_number,
    d.department_name,
    c.monthly_income,
    dept_avg.avg_income
FROM Employees e
INNER JOIN Departments  d ON e.department_id = d.department_id
INNER JOIN Compensation c ON e.employee_id   = c.employee_id
INNER JOIN (
    SELECT
        e2.department_id,
        AVG(c2.monthly_income) AS avg_income
    FROM Employees e2
    INNER JOIN Compensation c2 ON e2.employee_id = c2.employee_id
    GROUP BY e2.department_id
) dept_avg ON e.department_id = dept_avg.department_id
WHERE e.attrition = 'Yes'
  AND c.monthly_income < dept_avg.avg_income
ORDER BY d.department_name, c.monthly_income;

-- -----------------------------------------------------------------
-- Q8: Experience vs attrition (years at company buckets)
-- -----------------------------------------------------------------
SELECT
    CASE
        WHEN years_at_company <= 2  THEN '0-2 years'
        WHEN years_at_company <= 5  THEN '3-5 years'
        WHEN years_at_company <= 10 THEN '6-10 years'
        ELSE '10+ years'
    END                                                    AS tenure_bucket,
    COUNT(*)                                               AS total,
    SUM(attrition = 'Yes')                                 AS attrited,
    ROUND(SUM(attrition = 'Yes') * 100.0 / COUNT(*), 2)   AS attrition_rate_pct
FROM Employees
GROUP BY tenure_bucket
ORDER BY FIELD(tenure_bucket, '0-2 years', '3-5 years', '6-10 years', '10+ years');

-- -----------------------------------------------------------------
-- Q9: LEFT JOIN – employees with no risk score yet (unscored)
-- -----------------------------------------------------------------
SELECT
    e.emp_number,
    d.department_name,
    e.attrition,
    i.attrition_risk_score
FROM Employees e
LEFT JOIN Employee_Insights i ON e.employee_id = i.employee_id
INNER JOIN Departments d ON e.department_id = d.department_id
WHERE i.attrition_risk_score IS NULL
ORDER BY e.emp_number
LIMIT 20;

-- -----------------------------------------------------------------
-- Q10: Education level vs attrition rate
-- -----------------------------------------------------------------
SELECT
    CASE education
        WHEN 1 THEN 'Below College'
        WHEN 2 THEN 'College'
        WHEN 3 THEN 'Bachelor'
        WHEN 4 THEN 'Master'
        WHEN 5 THEN 'Doctor'
    END                                                    AS education_level,
    COUNT(*)                                               AS total,
    SUM(attrition = 'Yes')                                 AS attrited,
    ROUND(SUM(attrition = 'Yes') * 100.0 / COUNT(*), 2)   AS attrition_rate_pct
FROM Employees
GROUP BY education
ORDER BY education;

-- -----------------------------------------------------------------
-- Q11: Gender-wise average income and attrition using view
-- -----------------------------------------------------------------
SELECT
    gender,
    attrition,
    COUNT(*)                          AS count,
    ROUND(AVG(monthly_income), 2)     AS avg_monthly_income
FROM vw_employee_full
GROUP BY gender, attrition
ORDER BY gender, attrition;

-- -----------------------------------------------------------------
-- Q12: Stored procedure – department attrition report
-- -----------------------------------------------------------------
DROP PROCEDURE IF EXISTS sp_department_attrition_report;

DELIMITER $$
CREATE PROCEDURE sp_department_attrition_report(IN p_department VARCHAR(100))
BEGIN
    SELECT
        d.department_name,
        r.job_role_name,
        COUNT(*)                                               AS total,
        SUM(e.attrition = 'Yes')                               AS attrited,
        ROUND(SUM(e.attrition = 'Yes') * 100.0 / COUNT(*), 2) AS attrition_rate_pct,
        ROUND(AVG(c.monthly_income), 2)                        AS avg_income
    FROM Employees e
    INNER JOIN Departments  d ON e.department_id = d.department_id
    INNER JOIN Job_Roles    r ON e.job_role_id   = r.job_role_id
    INNER JOIN Compensation c ON e.employee_id   = c.employee_id
    WHERE d.department_name = p_department
    GROUP BY d.department_name, r.job_role_name
    ORDER BY attrition_rate_pct DESC;
END$$
DELIMITER ;

-- Usage example:
-- CALL sp_department_attrition_report('Sales');

-- =============================================================
-- schema.sql
-- Employee Attrition Prediction DB - Full Schema
-- Database: employee_attrition_db
-- =============================================================

-- Create and select the database
CREATE DATABASE IF NOT EXISTS employee_attrition_db
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE employee_attrition_db;

-- -----------------------------------------------------------------
-- 1. Departments (lookup table)
-- -----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS Departments (
    department_id   INT AUTO_INCREMENT PRIMARY KEY,
    department_name VARCHAR(100) NOT NULL UNIQUE,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_dept_name (department_name)
) ENGINE=InnoDB;

-- -----------------------------------------------------------------
-- 2. Job_Roles (lookup table)
-- -----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS Job_Roles (
    job_role_id   INT AUTO_INCREMENT PRIMARY KEY,
    job_role_name VARCHAR(100) NOT NULL UNIQUE,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_role_name (job_role_name)
) ENGINE=InnoDB;

-- -----------------------------------------------------------------
-- 3. Employees (core table)
-- -----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS Employees (
    employee_id          INT AUTO_INCREMENT PRIMARY KEY,
    emp_number           INT NOT NULL UNIQUE COMMENT 'EmployeeNumber from IBM dataset',

    -- Demographics
    age                  TINYINT UNSIGNED NOT NULL,
    gender               VARCHAR(10) NOT NULL,
    marital_status       VARCHAR(20) NOT NULL,
    education            TINYINT UNSIGNED NOT NULL COMMENT '1=Below College,2=College,3=Bachelor,4=Master,5=Doctor',
    education_field      VARCHAR(50)  NOT NULL,

    -- Work profile
    department_id        INT NOT NULL,
    job_role_id          INT NOT NULL,
    job_level            TINYINT UNSIGNED NOT NULL COMMENT '1=Entry ... 5=Executive',
    business_travel      VARCHAR(30)  NOT NULL,
    over_time            VARCHAR(5)   NOT NULL COMMENT 'Yes/No',

    -- Performance & satisfaction
    environment_satisfaction TINYINT UNSIGNED NOT NULL COMMENT '1-4 scale',
    job_involvement          TINYINT UNSIGNED NOT NULL COMMENT '1-4 scale',
    job_satisfaction         TINYINT UNSIGNED NOT NULL COMMENT '1-4 scale',
    performance_rating       TINYINT UNSIGNED NOT NULL COMMENT '1-4 scale',
    relationship_satisfaction TINYINT UNSIGNED NOT NULL COMMENT '1-4 scale',
    work_life_balance        TINYINT UNSIGNED NOT NULL COMMENT '1-4 scale',

    -- Experience
    total_working_years       TINYINT UNSIGNED NOT NULL,
    years_at_company          TINYINT UNSIGNED NOT NULL,
    years_in_current_role     TINYINT UNSIGNED NOT NULL,
    years_since_last_promotion TINYINT UNSIGNED NOT NULL,
    years_with_curr_manager   TINYINT UNSIGNED NOT NULL,
    num_companies_worked      TINYINT UNSIGNED NOT NULL,
    training_times_last_year  TINYINT UNSIGNED NOT NULL,

    -- Target
    attrition VARCHAR(5) NOT NULL COMMENT 'Yes/No',

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (department_id) REFERENCES Departments(department_id) ON DELETE RESTRICT,
    FOREIGN KEY (job_role_id)   REFERENCES Job_Roles(job_role_id)    ON DELETE RESTRICT,

    INDEX idx_attrition      (attrition),
    INDEX idx_department     (department_id),
    INDEX idx_job_role       (job_role_id),
    INDEX idx_age            (age),
    INDEX idx_emp_number     (emp_number)
) ENGINE=InnoDB;

-- -----------------------------------------------------------------
-- 4. Compensation
-- -----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS Compensation (
    compensation_id         INT AUTO_INCREMENT PRIMARY KEY,
    employee_id             INT NOT NULL UNIQUE,
    monthly_income          DECIMAL(10,2) NOT NULL,
    daily_rate              DECIMAL(10,2) NOT NULL,
    hourly_rate             DECIMAL(10,2) NOT NULL,
    monthly_rate            DECIMAL(10,2) NOT NULL,
    percent_salary_hike     TINYINT UNSIGNED NOT NULL,
    stock_option_level      TINYINT UNSIGNED NOT NULL COMMENT '0-3 scale',

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (employee_id) REFERENCES Employees(employee_id) ON DELETE CASCADE,
    INDEX idx_monthly_income (monthly_income)
) ENGINE=InnoDB;

-- -----------------------------------------------------------------
-- 5. Employee_Insights (derived / aggregated KPIs)
-- -----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS Employee_Insights (
    insight_id            INT AUTO_INCREMENT PRIMARY KEY,
    employee_id           INT NOT NULL UNIQUE,
    distance_from_home    SMALLINT UNSIGNED NOT NULL,
    over18                VARCHAR(5) NOT NULL DEFAULT 'Y',
    standard_hours        TINYINT UNSIGNED NOT NULL DEFAULT 80,
    attrition_risk_score  DECIMAL(5,4) NULL COMMENT 'Filled after ML scoring',

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (employee_id) REFERENCES Employees(employee_id) ON DELETE CASCADE
) ENGINE=InnoDB;

-- -----------------------------------------------------------------
-- Analytical Views
-- -----------------------------------------------------------------

-- View 1: Full employee flat view (convenience for queries)
CREATE OR REPLACE VIEW vw_employee_full AS
SELECT
    e.employee_id,
    e.emp_number,
    e.age,
    e.gender,
    e.marital_status,
    e.education,
    e.education_field,
    d.department_name,
    r.job_role_name,
    e.job_level,
    e.business_travel,
    e.over_time,
    e.environment_satisfaction,
    e.job_involvement,
    e.job_satisfaction,
    e.performance_rating,
    e.relationship_satisfaction,
    e.work_life_balance,
    e.total_working_years,
    e.years_at_company,
    e.years_in_current_role,
    e.years_since_last_promotion,
    e.years_with_curr_manager,
    e.num_companies_worked,
    e.training_times_last_year,
    e.attrition,
    c.monthly_income,
    c.daily_rate,
    c.hourly_rate,
    c.monthly_rate,
    c.percent_salary_hike,
    c.stock_option_level,
    i.distance_from_home,
    i.attrition_risk_score
FROM Employees e
JOIN Departments  d ON e.department_id = d.department_id
JOIN Job_Roles    r ON e.job_role_id   = r.job_role_id
JOIN Compensation c ON e.employee_id   = c.employee_id
JOIN Employee_Insights i ON e.employee_id = i.employee_id;


-- View 2: Department-wise attrition summary
CREATE OR REPLACE VIEW vw_dept_attrition AS
SELECT
    d.department_name,
    COUNT(*)                                           AS total_employees,
    SUM(e.attrition = 'Yes')                           AS attrited,
    SUM(e.attrition = 'No')                            AS retained,
    ROUND(SUM(e.attrition = 'Yes') * 100.0 / COUNT(*), 2) AS attrition_rate_pct
FROM Employees e
JOIN Departments d ON e.department_id = d.department_id
GROUP BY d.department_name;


-- View 3: Job-role attrition summary
CREATE OR REPLACE VIEW vw_role_attrition AS
SELECT
    r.job_role_name,
    COUNT(*)                                               AS total_employees,
    SUM(e.attrition = 'Yes')                               AS attrited,
    ROUND(SUM(e.attrition = 'Yes') * 100.0 / COUNT(*), 2) AS attrition_rate_pct,
    ROUND(AVG(c.monthly_income), 2)                        AS avg_monthly_income
FROM Employees e
JOIN Job_Roles    r ON e.job_role_id = r.job_role_id
JOIN Compensation c ON e.employee_id = c.employee_id
GROUP BY r.job_role_name;

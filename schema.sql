-- Portable PostgreSQL / SQLite schema. Account-month grain prevents inflated balances.
CREATE TABLE IF NOT EXISTS accounts (
  account_id INTEGER PRIMARY KEY,
  limit_bal NUMERIC NOT NULL CHECK (limit_bal > 0)
);
CREATE TABLE IF NOT EXISTS repayment_history (
  account_id INTEGER NOT NULL REFERENCES accounts(account_id),
  month_index INTEGER NOT NULL CHECK (month_index BETWEEN 1 AND 6),
  pay_status INTEGER NOT NULL,
  bill_amt NUMERIC NOT NULL,
  pay_amt NUMERIC NOT NULL,
  PRIMARY KEY (account_id, month_index)
);
DROP VIEW IF EXISTS credit_features;
CREATE VIEW credit_features AS
WITH history AS (
  SELECT a.account_id, a.limit_bal, h.month_index, h.pay_status, h.bill_amt, h.pay_amt,
         LAG(bill_amt) OVER (
           PARTITION BY a.account_id ORDER BY month_index
         ) AS previous_bill,
         ROW_NUMBER() OVER (
           PARTITION BY a.account_id ORDER BY month_index DESC
         ) AS recency_rank
  FROM accounts a JOIN repayment_history h ON a.account_id=h.account_id
)
SELECT account_id, MAX(limit_bal) AS limit_bal,
       MAX(pay_status) FILTER (WHERE recency_rank=1) AS latest_status,
       MAX(pay_status) AS max_status,
       COUNT(*) FILTER (WHERE pay_status>0) AS delinquent_months,
       AVG(1.0*bill_amt/NULLIF(limit_bal,0)) AS avg_utilization,
       1.0*SUM(pay_amt)/NULLIF(SUM(bill_amt),0) AS payment_to_bill,
       MAX(bill_amt-previous_bill) FILTER (WHERE recency_rank=1) AS balance_change
FROM history GROUP BY account_id;

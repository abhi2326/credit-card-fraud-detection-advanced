-- PostgreSQL: overall fraud KPIs
SELECT
    COUNT(*) AS total_transactions,
    COUNT(*) FILTER (WHERE is_fraud = 1) AS fraud_transactions,
    ROUND(100.0 * COUNT(*) FILTER (WHERE is_fraud = 1) / NULLIF(COUNT(*), 0), 3) AS fraud_rate_pct,
    ROUND(SUM(amount), 2) AS total_value,
    ROUND(SUM(amount) FILTER (WHERE is_fraud = 1), 2) AS fraud_value
FROM transactions;

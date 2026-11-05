-- PostgreSQL hourly monitoring; assumes transaction_time is a timestamp.
SELECT
    DATE_TRUNC('hour', transaction_time) AS transaction_hour,
    COUNT(*) AS transaction_count,
    SUM(is_fraud) AS fraud_count,
    ROUND(100.0 * SUM(is_fraud) / NULLIF(COUNT(*), 0), 3) AS fraud_rate_pct,
    ROUND(SUM(amount), 2) AS gross_value
FROM transactions
GROUP BY 1
ORDER BY 1;

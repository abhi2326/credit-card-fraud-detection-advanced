-- Prioritise transactions for manual review. Tune the cutoff to analyst capacity.
SELECT transaction_id, transaction_time, amount, merchant_category, risk_score
FROM transactions
WHERE risk_score >= 0.80
ORDER BY risk_score DESC, amount DESC
LIMIT 500;

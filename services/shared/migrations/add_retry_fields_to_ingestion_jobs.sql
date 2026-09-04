-- Migration: Add retry fields to ingestion_jobs table
-- Requirements: 1.5, 13.3
-- Description: Adds retry_count, max_retries, and next_retry_at fields to support
--              exponential backoff retry logic for failed ingestion jobs

-- Add retry_count column (default 0)
ALTER TABLE ingestion_jobs 
ADD COLUMN IF NOT EXISTS retry_count INTEGER NOT NULL DEFAULT 0;

-- Add max_retries column (default 3)
ALTER TABLE ingestion_jobs 
ADD COLUMN IF NOT EXISTS max_retries INTEGER NOT NULL DEFAULT 3;

-- Add next_retry_at column (nullable, for scheduled retries)
ALTER TABLE ingestion_jobs 
ADD COLUMN IF NOT EXISTS next_retry_at TIMESTAMP WITH TIME ZONE;

-- Add check constraints
ALTER TABLE ingestion_jobs 
ADD CONSTRAINT IF NOT EXISTS check_retry_count_non_negative 
CHECK (retry_count >= 0);

ALTER TABLE ingestion_jobs 
ADD CONSTRAINT IF NOT EXISTS check_max_retries_non_negative 
CHECK (max_retries >= 0);

-- Add index on next_retry_at for efficient retry job queries
CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_next_retry 
ON ingestion_jobs(next_retry_at) 
WHERE next_retry_at IS NOT NULL;

-- Add comment
COMMENT ON COLUMN ingestion_jobs.retry_count IS 'Number of retry attempts made for this job';
COMMENT ON COLUMN ingestion_jobs.max_retries IS 'Maximum number of retry attempts allowed';
COMMENT ON COLUMN ingestion_jobs.next_retry_at IS 'Timestamp when the job should be retried (NULL if not scheduled for retry)';

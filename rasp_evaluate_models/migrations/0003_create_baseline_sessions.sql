-- depends: 0002_add_energy_per_image

CREATE TABLE IF NOT EXISTS baseline_sessions (
    id                SERIAL          PRIMARY KEY,
    session_id        VARCHAR(255)    NOT NULL,
    period            VARCHAR(20),
    day               VARCHAR(20),
    mode              VARCHAR(20),
    window_start      TIMESTAMPTZ     NOT NULL,
    window_end        TIMESTAMPTZ     NOT NULL,
    analysis_start    TIMESTAMPTZ     NOT NULL,
    analysis_end      TIMESTAMPTZ     NOT NULL,
    avg_cpu_pct       DOUBLE PRECISION,
    max_cpu_pct       DOUBLE PRECISION,
    avg_mem_pct       DOUBLE PRECISION,
    avg_mem_mb        DOUBLE PRECISION,
    max_mem_mb        DOUBLE PRECISION,
    avg_current_a     DOUBLE PRECISION,
    max_current_a     DOUBLE PRECISION,
    avg_power_w       DOUBLE PRECISION,
    max_power_w       DOUBLE PRECISION,
    data_quality_ok   BOOLEAN         NOT NULL,
    quality_warnings  TEXT,
    created_at        TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_baseline_sessions_session_id UNIQUE (session_id)
);

CREATE INDEX IF NOT EXISTS idx_baseline_sessions_period_day
    ON baseline_sessions (period, day);

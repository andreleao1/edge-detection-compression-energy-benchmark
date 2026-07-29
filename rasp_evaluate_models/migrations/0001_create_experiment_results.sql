-- depends:

CREATE TABLE IF NOT EXISTS experiment_results (
    id              SERIAL          PRIMARY KEY,
    nome_modelo     VARCHAR(255)    NOT NULL,
    nome_dataset    VARCHAR(255)    NOT NULL,
    avg_watt        DOUBLE PRECISION,
    max_watt        DOUBLE PRECISION,
    avg_cpu         DOUBLE PRECISION,
    avg_mem         DOUBLE PRECISION,
    avg_temp        DOUBLE PRECISION,
    data_execucao   TIMESTAMPTZ     NOT NULL,
    duracao_total   DOUBLE PRECISION NOT NULL,
    erro            TEXT,
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_results_model
    ON experiment_results (nome_modelo);

CREATE INDEX IF NOT EXISTS idx_results_dataset
    ON experiment_results (nome_dataset);

CREATE INDEX IF NOT EXISTS idx_results_execucao
    ON experiment_results (data_execucao DESC);

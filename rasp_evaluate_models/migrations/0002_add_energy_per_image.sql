-- depends: 0001_create_experiment_results

ALTER TABLE experiment_results
    ADD COLUMN IF NOT EXISTS total_imagens     INTEGER,
    ADD COLUMN IF NOT EXISTS joules_por_imagem DOUBLE PRECISION;

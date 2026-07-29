ALTER TABLE experiment_results
    DROP COLUMN IF EXISTS total_imagens,
    DROP COLUMN IF EXISTS joules_por_imagem;

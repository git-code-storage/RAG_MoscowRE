import subprocess
import itertools
from dotenv import load_dotenv

load_dotenv()

# 1. Задаем, какие пайплайны запускать
BASE_PIPELINE = False
GIBRID_PIPELINE = True

# 2. Задаем сетку параметров для перебора (Grid Search)
# Обратите внимание: с точки зрения MLproject, with_lema должен быть строкой "True" или "False"
param_grid = {
    "model": ["GigaChat-2"],
    "test_path": ["raw_data/questions_full.jsonl"],
    "with_lema": ["False"],  # Передаем как строку для совместимости с MLproject
    "chunks": [700],
    "weight_vect": [0.5],
    "const_rrf": [50, 70],
    "emb": ["qwen"],
}

# Словарь валидных параметров для каждого entry-point (как указано в MLproject)
VALID_PARAMS = {
    "main": ["model", "test_path", "chunks", "emb"],
    "gibrid": ["model", "test_path", "weight_vect", "const_rrf", "with_lema", "chunks", "emb"]
}


def build_mlflow_command(entry_point, params, run_index):
    """Формирует команду mlflow run для конкретного entry-point и параметров."""
    cmd = [
        "mlflow", "run", ".",
        "--entry-point", entry_point,
        "--experiment-name", "RAG_MoscowRE_v_1",
        "--env-manager", "local",
        # Делаем имя запуска уникальным для каждой конфигурации, чтобы MLflow не конфликтовал
        "--run-name", f"{entry_point}_config_{run_index}"
    ]

    # Добавляем только те параметры, которые разрешены для данного entry-point
    for key, value in params.items():
        if key in VALID_PARAMS.get(entry_point, []):
            # Гарантируем строковое представление (особенно важно для "True"/"False")
            val_str = str(value)
            cmd.extend(["-P", f"{key}={val_str}"])

    return cmd


def run_mlflow_pipeline(entry_point, params, run_index):
    """Последовательно запускает mlflow run."""
    cmd = build_mlflow_command(entry_point, params, run_index)

    print(f"\n{'=' * 70}")
    print(f"🚀 MLflow Запуск: {entry_point} (Run #{run_index})")
    print(f"Команда: {' '.join(cmd)}")
    print(f"{'=' * 70}")

    # Запуск процесса
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"❌ Ошибка при выполнении mlflow run {entry_point} (код {result.returncode})")
        # Можно раскомментировать, если нужно прерывать весь скрипт при первой ошибке:
        # sys.exit(1)
    else:
        print(f"✅ {entry_point} успешно завершен и залогирован в MLflow.")


if __name__ == "__main__":
    # Генерируем все комбинации параметров (декартово произведение)
    keys = param_grid.keys()
    values = param_grid.values()
    configs = [dict(zip(keys, combination)) for combination in itertools.product(*values)]

    print(f"📊 Всего уникальных конфигураций для тестов: {len(configs)}")
    print(f"[DEBUG] Configs: {configs}")

    global_run_counter = 1

    for i, config in enumerate(configs):
        print(f"\n{'#' * 70}")
        print(f" ⚙️  КОНФИГУРАЦИЯ {i + 1} / {len(configs)} ")
        print(f" Параметры: {config}")
        print(f"{'#' * 70}")

        # 1. Последовательно запускаем base_pipeline (entry-point: main)
        if BASE_PIPELINE:
            run_mlflow_pipeline("main", config, global_run_counter)
            global_run_counter += 1

        # 2. Последовательно запускаем gibrid_pipeline (entry-point: gibrid)
        if GIBRID_PIPELINE:
            run_mlflow_pipeline("gibrid", config, global_run_counter)
            global_run_counter += 1

    print("\n🎉 Все эксперименты завершены!")
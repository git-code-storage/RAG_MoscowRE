# Базовые импорты
import os
import tempfile
import json
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

# Импорты Python
from pathlib import Path

# Импорты для работы с данными и тестирования
import argparse
import mlflow
import pandas as pd

# Импорты для работы с langchain
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.retrievers import BM25Retriever
from langchain_gigachat.chat_models import GigaChat
from langchain_huggingface import HuggingFaceEmbeddings

# Импорты проекта
import prompts
import utils

# Константы
ALLOWED_MODELS = {"GigaChat-2", "GigaChat-2-Pro", "GigaChat-3-Lightning"}
ALLOWED_EMBEDDINGS = {"se", "qwen"}
DB_NAME = "chroma"

# Переменные окружения
gigachat_ak = os.getenv("GIGACHAT_AK")
collection_name = "langchain"
base_dir = Path(__file__).resolve().parent

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
csv_filename = f"base_result_{timestamp}.csv"
csv_dir = base_dir / "outputs" / csv_filename
metrics_filename = f"metrics_result_{timestamp}.csv"
metrics_dir = base_dir / "outputs" / metrics_filename
persist_dir = str(base_dir / "db" / DB_NAME)


def main(model, test_path, weight_vect, const_rrf, with_lema, chunks, emb):
    with mlflow.start_run(tags={"model": model,
                                "pipeline": "gibrid",
                                "prompt_version": "v4",
                                }):

        # Валидация
        model = utils.validate_choice(model, ALLOWED_MODELS, "model")
        emb = utils.validate_choice(emb, ALLOWED_EMBEDDINGS, "embedding")

        weight_b25 = 1 - weight_vect
        weights = [weight_b25, weight_vect]

        json_parser = JsonOutputParser()

        debug_log =[]

        llm = GigaChat(
            credentials=gigachat_ak,
            model=model,
            verify_ssl_certs=False,
            timeout = 120.0,
        )

        if emb == "qwen":
            embeddings = HuggingFaceEmbeddings(model_name="Qwen/Qwen3-Embedding-0.6B")
        if emb == "se":
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        col_name = "chunk_" + str(chunks) + "_" + emb

        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_name=col_name,
        )

        metadata_prompt = ChatPromptTemplate.from_messages([
            ("system", prompts.metadata_system),
            ("human", prompts.metadata_human),
        ], template_format="jinja2")

        base_prompt = ChatPromptTemplate.from_messages([
            ('system', prompts.system_prompt_2),
            ('human', prompts.human_prompt)
        ], template_format="jinja2")

        metrics_prompt = ChatPromptTemplate.from_messages([
            ('system', prompts.sys_metrics_prompt_2),
            ('human', prompts.hum_metrics_prompt)
        ], template_format="jinja2")

        extractor_chain = (
                RunnablePassthrough()
                | metadata_prompt
                | llm
                | json_parser
        )

        all_docs_raw = vectorstore.get()
        docs_docs = all_docs_raw["documents"]
        docs_meta = all_docs_raw["metadatas"]
        all_documents = [
            Document(page_content=d, metadata=m)
            for d, m in zip(docs_docs, docs_meta)
        ]

        if with_lema:
            bm25_retriever = BM25Retriever.from_documents(all_documents,
                                                          k=10,
                                                          preprocess_func=utils.russian_preprocess)
            print(f"[INIT] BM25 ретривер создан по {len(all_documents)} документам. Лемматизация используется")
            debug_log.append(
                f"[INIT] BM25 ретривер создан по {len(all_documents)} документам. Лемматизация используется")
        else:
            bm25_retriever = BM25Retriever.from_documents(all_documents, k=10)
            print(f"[INIT] BM25 ретривер создан по {len(all_documents)} документам. Лемматизация не используется")
            debug_log.append(
                f"[INIT] BM25 ретривер создан по {len(all_documents)} документам. Лемматизация не используется")

        def rag_with_filter(user_question: str) -> tuple[str, str]:
            # 1. Получаем фильтр из вопроса
            raw_filter = utils.safe_llm_invoke(extractor_chain, {"question": user_query})
            #raw_filter = extractor_chain.invoke({"question": user_question})
            print("[DEBUG] raw filter from LLM:", raw_filter, "type:", type(raw_filter))
            debug_log.append(f"[RAW FILTER] {raw_filter}")

            # 2. Парсим фильтр в словарь
            parsed_filter = utils.parse_filter(raw_filter)
            print("[DEBUG] parsed filter from LLM:", parsed_filter, "type:", type(parsed_filter))
            debug_log.append(f"[PARSED FILTER] {parsed_filter}")

            # 3. Нормализуем фильтр под требования Chroma
            normalized_filter = utils.normalize_chroma_filter(parsed_filter)
            print("[DEBUG] normalized filter:", normalized_filter, "type:", type(normalized_filter))
            debug_log.append(f"[NORMALIZED FILTER] {normalized_filter}")

            # 4. Создаём ретривер для векторного поиска
            search_kwargs = {"k": 10}
            if normalized_filter is not None:
                search_kwargs["filter"] = normalized_filter
                print(f"[RAG] Поиск с фильтром {normalized_filter}")
                debug_log.append(f"[RAG] Поиск с фильтром {normalized_filter}")
            else:
                print("[RAG] Фильтр не извлечён или пуст → поиск без фильтра")
                debug_log.append("[RAG] Фильтр не извлечён или пуст → поиск без фильтра")

            vector_retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
            docs = vector_retriever.invoke(user_question)
            print(f"[RAG] Найдено документов векторным поиском: {len(docs)}")
            debug_log.append(f"[RAG] Найдено документов векторным поиском: {len(docs)}")

            # 5. Если ретривер векторного поиска с фильтром ничего не вернул, то проводим повторный поиск без фильтра
            if normalized_filter and not docs:
                print("[RAG] Поиск с фильтром по запросу вернул пустоту → повторный поиск без фильтра")
                debug_log.append("[RAG] Поиск с фильтром по запросу вернул пустоту → повторный поиск без фильтра")
                search_kwargs = {"k": 10}
                vector_retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
                docs = vector_retriever.invoke(user_question)
                print(f"[RAG] Повторный поиск без фильтра → найдено: {len(docs)}")
                debug_log.append(f"[RAG] Повторный поиск без фильтра → найдено: {len(docs)}")

            # 6. Получение ответов от ретриверов
            retrievers = [bm25_retriever, vector_retriever]
            docs = utils.ensemble_retrieve(user_question, retrievers, weights=weights, const_rrf=const_rrf, k=5)

            # 7. Если и после этого пусто — не отправляем в LLM
            if not docs:
                return "Не знаю", "Не знаю"

            # 8. Формируем контекст
            context_str = utils.get_context(docs)

            # 9. Готовим промпт и вызываем LLM
            messages = base_prompt.format_messages(context=context_str, question=user_question)
            response = utils.safe_llm_invoke(llm, messages)

            return response.content, context_str

        questions = []

        with open(test_path, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line)
                data = dict(record)
                questions.append(data)

        for num, question in enumerate(questions):
            user_query = question['question']
            print(f"[DEBUG] user query: {user_query}")
            debug_log.append(f"[DEBUG] user query: {user_query}")
            answer_rag, docs = rag_with_filter(user_query)
            if answer_rag is None:
                mlflow.log_param("llm_status", "failed_network")
                mlflow.set_tag("error_type", "RemoteProtocolError")
                mlflow.log_metric("llm_retries_count", 3)
                answer_rag = "Не удалось получить ответ от языковой модели из-за временной ошибки соединения."
            print(f"[DEBUG] answer_rag: {answer_rag}")
            debug_log.append(f"[DEBUG] answer_rag: {answer_rag}")
            debug_log.append("-" * 60)
            question['id'] = num
            question['answer_rag'] = utils.truncate_before_answer(answer_rag)
            question['chunks'] = docs

        question_df = pd.DataFrame(questions)

        answers = [{
            'id': el['id'],
            'answer': el['answer'],
            'answer_rag': el['answer_rag']
        } for el in questions]

        batch_size = 50
        metrics_dfs = []

        # Разбиваем answers на батчи и запрашиваем оценки
        for i in range(0, len(answers), batch_size):
            batch = answers[i:i + batch_size]
            if not batch:
                continue
            batch_metrics = utils.get_metrics_from_llm(metrics_prompt, llm, batch)
            if batch_metrics:
                frame = pd.DataFrame(batch_metrics)
                metrics_dfs.append(frame)
                print(len(metrics_dfs))
                print(len(frame))

        # Собираем все батчи в один metrics_df
        if metrics_dfs:
            metrics_df = pd.concat(metrics_dfs, axis=0, ignore_index=True)
        else:
            # Если ни один батч не дал валидных данных — пустой DF с колонкой id (чтобы merge не упал)
            metrics_df = pd.DataFrame(columns=["id", "correct"])


        result_df = utils.safe_merge_metrics(question_df, metrics_df)


        metrics_df.to_csv(metrics_dir)
        result_df.to_csv(csv_dir)

        accuracy = result_df['correct'].sum() / len(result_df['correct'])

        print(accuracy)

        # Логируем параметры в mlflow
        mlflow.log_param("model", model)
        mlflow.log_param("test_path", test_path)

        # Логируем метрику в mlflow
        mlflow.log_metric("accuracy", accuracy)

        # Логируем артефакты в mlflow
        log_content = "\n".join(debug_log)
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix="_debug.txt", encoding="utf-8") as tmp:
            tmp.write(log_content)
            tmp_path = tmp.name
        mlflow.log_artifact(tmp_path)
        os.unlink(tmp_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="GigaChat-2")
    parser.add_argument("--test_path", type=str, default="raw_data/questions_lite.jsonl")
    parser.add_argument("--weight_vect", type=float, default=1.0)
    parser.add_argument("--const_rrf", type=int, default=60)
    parser.add_argument("--with_lema", type=utils.str2bool, default=True)
    parser.add_argument("--chunks", type=int, default=500)
    parser.add_argument("--emb", type=str, default="se")
    args = parser.parse_args()

    main(args.model, args.test_path, args.weight_vect, args.const_rrf, args.with_lema, args.chunks, args.emb)
# Базовые импорты
import json
import ast
import re
import time
import pymorphy3

# Импорты Python
from typing import Any, Dict, Optional
from httpx import RemoteProtocolError
from collections import defaultdict


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise ValueError('Boolean value expected.')


def parse_filter(text: Any) -> Dict:
    """
    Пытается превратить ответ LLM в dict.
    Поддерживает:
      - Чистый JSON со двойными кавычками
      - Markdown-блоки ```json ... ```
      - Python-style dict с одинарными кавычками (через ast)
      - Лишний текст вокруг JSON (вырезается регуляркой)
    """

    if text is None:
        return {}

    # Если это уже dict (благодаря JsonOutputParser), просто возвращаем его
    if isinstance(text, dict):
        return text

    raw = str(text).strip()
    clean_text = raw

    # 1. Markdown
    if "```" in clean_text:
        parts = clean_text.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("{") or part.startswith("["):
                clean_text = part
                break

    # 2. Regex для вырезки JSON
    match = re.search(r'\{[\s\S]*\}', clean_text)
    if match:
        clean_text = match.group(0)

    # 3. Стандартный JSON
    try:
        data = json.loads(clean_text)
        return data if isinstance(data, (dict, list)) else {}
    except json.JSONDecodeError:
        pass

    # 4. Fallback на ast (для одинарных кавычек)
    try:
        data = ast.literal_eval(clean_text)
        return data if isinstance(data, (dict, list)) else {}
    except Exception:
        pass

    print(f"[WARNING] Ошибка парсинга JSON. Вернём пустой фильтр. Текст: {raw[:100]}...")
    return {}


def get_context(docs):
    if not docs:
        return "Контекст пустой"
    full_context = []
    for doc in docs:
        context_lines = [
            f"[START_BLOCK]"
            f"[START_META_BLOCK]"
            f"Застройщик: {doc.metadata.get('dev', 'не указан')}",
            f"Название ЖК: {doc.metadata.get('proj', 'не указано')}",
            f"Класс жилья: {doc.metadata.get('h_class', 'не указан')}",
            f"Район: {doc.metadata.get('distr', 'не указан')}",
            f"Метро: {doc.metadata.get('metro', 'не указано')}",
            f"Расстояние до метро в км: {doc.metadata.get('dist_to_metro', 'не указано')}",
            f"Количество жильцов: {doc.metadata.get('num_resid', 'не указано')}",
            f"Планируемое количество мест в школе: {doc.metadata.get('s_plan', 'не указано')}",
            f"Доступных мест в школе сейчас: {doc.metadata.get('s_fact', 'не указано')}",
            f"Планируемое количество мест в детсадах: {doc.metadata.get('n_plan', 'не указано')}",
            f"Доступных мест в детсадах сейчас: {doc.metadata.get('n_fact', 'не указано')}",
            f"Тип парковки: {doc.metadata.get('parking', 'не указано')}",
            f"[END_META_BLOCK]"
            f"{doc.metadata.get('proj', 'без названия')}: {doc.page_content.strip()}",
            f"[END_BLOCK]"
        ]
        context = '\n'.join(context_lines)
        full_context.append(context)

    return '\n\n'.join(full_context)


def normalize_chroma_filter(raw_filter: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not raw_filter or not isinstance(raw_filter, dict) or len(raw_filter) == 0:
        return None

    operators = {"$eq", "$ne", "$gte", "$gt", "$lte", "$lt", "$in", "$nin"}
    top_level_logic = []

    # Сначала смотрим, есть ли $or / $and на верхнем уровне
    has_or = "$or" in raw_filter
    has_and = "$and" in raw_filter

    if has_or or has_and:
        # Если есть $or или $and, считаем, что это уже готовая логическая конструкция.
        # Но нужно убедиться, что внутри — валидные условия.
        result_conditions = []
        source_key = "$or" if has_or else "$and"
        candidates = raw_filter[source_key]

        if not isinstance(candidates, list):
            # Если $or/$and не список — отбрасываем эту часть
            candidates = []

        for cond in candidates:
            if not isinstance(cond, dict):
                continue
            # Рекурсивно нормализуем каждое условие (на случай вложенности)
            normalized_cond = normalize_chroma_filter(cond)
            if normalized_cond:
                result_conditions.append(normalized_cond)

        if len(result_conditions) == 0:
            return None
        elif len(result_conditions) == 1:
            # Chroma не любит $or/and с одним элементом — возвращаем само условие
            return result_conditions[0]

        # Проверяем 'proj' — при нескольких критериях $and меняем на $or
        if source_key == "$and":
            proj_conditions = []      # условия, касающиеся только поля 'proj'
            other_conditions = []     # все остальные условия

            for cond in result_conditions:
                if isinstance(cond, dict) and len(cond) == 1 and "proj" in cond:
                    proj_conditions.append(cond)
                else:
                    other_conditions.append(cond)

            if len(proj_conditions) > 1:
                if not other_conditions:
                    # все критерии — по 'proj': $and целиком превращаем в $or
                    return {"$or": proj_conditions}
                # есть и другие поля: proj-критерии объединяем в $or,
                # остальные условия оставляем под $and
                other_conditions.insert(0, {"$or": proj_conditions})
                return {"$and": other_conditions}

        return {source_key: result_conditions}

    # Если нет $or/$and на верхнем уровне — обрабатываем как набор полей
    conditions = []
    for key, value in raw_filter.items():
        # Пропускаем служебные ключи, если вдруг попадут
        if key in ("$or", "$and"):
            continue

        # Пропускаем None, пустые строки
        if value is None or (isinstance(value, str) and value.strip() == ""):
            continue

        # Список → для 'proj' сразу $or по элементам, для остальных полей $in
        if isinstance(value, list):
            if len(value) == 0:
                continue  # пустой список — невалидно, пропускаем
            if key == "proj":
                conditions.append({
                    "$or": [{key: {"$eq": v}} for v in value]
                })
            else:
                conditions.append({key: {"$in": value}})
            continue

        # Если value — словарь, проверяем, есть ли в нём валидный оператор
        if isinstance(value, dict):
            has_operator = any(op in value for op in operators)
            if has_operator:
                # NEW: защита от $eq/$ne со списком — Chroma это не принимает
                if ("$eq" in value or "$ne" in value) and isinstance(value["$eq" if "$eq" in value else "$ne"], list):
                    if key == "proj":
                        conditions.append({
                            "$or": [{key: {"$eq": v}} for v in value["$eq" if "$eq" in value else "$ne"]]
                        })
                    else:
                        conditions.append({key: {"$in": value["$eq" if "$eq" in value else "$ne"]}})
                else:
                    conditions.append({key: value})
            else:
                # Пустой dict или dict без операторов → пропускаем
                continue
        else:
            # Обычное значение → оборачиваем в $eq
            conditions.append({key: {"$eq": value}})

    if len(conditions) == 0:
        return None
    elif len(conditions) == 1:
        return conditions[0]
    else:
        return {"$and": conditions}


def safe_llm_invoke(llm, messages, max_retries = 3, base_delay = 2.0) -> Optional[Any]:
    """
    Вызывает LLM с повторными попытками при сетевых сбоях.
    Возвращает результат или None, если все попытки исчерпаны.
    """
    last_exception = None

    for attempt in range(1, max_retries + 1):
        try:
            return llm.invoke(messages)
        except RemoteProtocolError as e:
            last_exception = e
            print(f"[Retry {attempt}/{max_retries}] Network error (GigaChat disconnected): {e}")
            if attempt < max_retries:
                delay = base_delay * (2 ** (attempt - 1))  # экспоненциальная задержка: 2, 4, 8 сек
                print(f"Waiting {delay:.1f}s before retry...")
                time.sleep(delay)
            continue
        except Exception as e:
            # Другие ошибки (валидация, формат и т.п.) не повторяем
            print(f"[Fatal error] LLM invocation failed: {e}")
            return None

    print("[ERROR] All retries failed. Returning None as fallback.")
    return None


def ensemble_retrieve(query: str, retrievers: list, weights: list = None, const_rrf:int = 60, k: int = 5):
    """
    Объединяет результаты нескольких ретриверов по алгоритму RRF.
    """
    if weights is None:
        weights = [1.0] * len(retrievers)

    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    # Собираем результаты от каждого ретривера
    fused_scores = defaultdict(float)
    doc_map = {}

    for retriever, weight in zip(retrievers, weights):
        docs = retriever.invoke(query)
        for rank, doc in enumerate(docs, start=1):
            # Уникальный ключ — хэш от содержимого + метаданных
            doc_key = (doc.page_content, str(sorted(doc.metadata.items())))
            fused_scores[doc_key] += weight / (const_rrf + rank)  # 60 — стандартная константа RRF
            doc_map[doc_key] = doc

    # Сортируем по итоговому скору и берем топ-k
    sorted_keys = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:k]
    return [doc_map[key] for key, _ in sorted_keys]


def safe_merge_metrics(question_df, metrics_df):
    """
    Безопасно объединяет question_df и metrics_df по колонке 'id'.
    Возвращает объединённый DataFrame. Если 'id' нет — поднимает ошибку.
    """
    if "id" not in question_df.columns:
        raise ValueError("В question_df отсутствует колонка 'id'")
    if "id" not in metrics_df.columns:
        raise ValueError("В metrics_df отсутствует колонка 'id'")

    # Важно: how='left', чтобы не потерять вопросы, если для них нет оценки
    merged = question_df.merge(metrics_df[["id", "correct"]], on="id", how="left")

    # Заполняем NaN (если для каких-то вопросов не пришла оценка) значением 0
    merged["correct"] = merged["correct"].fillna(0).astype(int)
    return merged


def truncate_before_answer(text: str) -> str:
    marker = "Ответ:"
    idx = text.find(marker)
    if idx == -1:
        return text
    return text[idx + len(marker):]


def russian_preprocess(text: str) -> list[str]:
    """
    Токенизация + лемматизация через pymorphy3.
    Возвращает список лемм — именно его ждёт BM25Retriever.
    """
    morph = pymorphy3.MorphAnalyzer()
    _word_re = re.compile(r"[а-яёa-z0-9\-]+", re.IGNORECASE)
    STOPWORDS = {"и", "в", "на", "с", "по", "для", "от", "у", "какой", "какая", "какое"}
    tokens = _word_re.findall(text.lower())
    lemmas = []
    for tok in tokens:
        if tok in STOPWORDS:
            continue
        # parse возвращает список разборов; берём первый (наиболее вероятный)
        lemma = morph.parse(tok)[0].normal_form
        lemmas.append(lemma)
    return lemmas


def _normalize_metrics(raw: str) -> list[dict]:
    """Приводит сырой ответ LLM к списку словарей.
    Поддерживает форматы:
      1) newline-separated JSON объектов
      2) JSON-массив объектов: [{...}, {...}]
      3) JSON-массив строк, где каждая строка — JSON объекта: ["{...}", "{...}"]
      4) Обёртка в ```json ... ```
    """
    s = raw.strip()
    # Снимаем markdown-обёртку
    if s.startswith("```"):
        s = re.sub(r"^```(?:jsonl)?\s*", "", s)
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s).strip()

    # Попытка 1: сразу как JSON (массив объектов или массив строк)
    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            result = []
            for item in parsed:
                if isinstance(item, dict):
                    result.append(item)
                elif isinstance(item, str):
                    # строка может быть JSON-объектом
                    try:
                        result.append(json.loads(item))
                    except json.JSONDecodeError:
                        repaired = repair_json(item)
                        try:
                            result.append(json.loads(repaired))
                        except json.JSONDecodeError:
                            print(f"[WARN] не удалось распарсить элемент: {item[:120]}")
            if result:
                return result
    except json.JSONDecodeError:
        pass

    # Попытка 2: newline-separated JSON
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    result = []
    for ln in lines:
        try:
            result.append(json.loads(ln))
            continue
        except json.JSONDecodeError:
            pass
        try:
            result.append(json.loads(repair_json(ln)))
        except json.JSONDecodeError:
            print(f"[WARN] пропущена невалидная строка: {ln[:120]}")
    return result


def get_metrics_from_llm(prompt, llm, answers_batch):
    """Отправляет батч в LLM и возвращает список словарей (или пустой список при ошибке)."""
    messages = prompt.format_messages(answers=str(answers_batch))

    response = safe_llm_invoke(llm, messages)

    raw = response.content if hasattr(response, "content") else str(response)

    metrics = _normalize_metrics(raw)

    if not metrics:
        print("[DEBUG] get_metrics_from_llm: не удалось получить список JSON. Вернул пустой список")
        return []

    # Валидация: оставляем только корректные записи с id и correct
    valid = []
    for doc in metrics:
        if isinstance(doc, dict) and "id" in doc and "correct" in doc:
            valid.append(doc)
        else:
            print(f"[WARN] пропущена запись без id/correct: {doc}")
    return valid


def repair_json(raw: str) -> str:
    """Дозакрывает оборванный JSON: строки (в т.ч. в одинарных кавычках), escape-и,
    неэкранированные кавычки внутри значений, висячие ключи/запятые/двоеточия,
    недописанные литералы true/false/null и незакрытые/лишние скобки."""
    s = raw.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s).strip()

    out = []
    stack = []  # 'obj' / 'arr'
    in_string = False
    escape = False
    expecting = None  # 'key' | 'colon' | 'value' | 'comma'
    n = len(s)
    i = 0

    while i < n:
        ch = s[i]

        if escape:
            # Обработка избыточного экранирования одинарных кавычек
            if ch == "'":
                out.append("'")  # \' -> '
                escape = False
                i += 1
                continue
            out.append(ch)
            escape = False
            i += 1
            continue

        if in_string:
            if ch == "\\":
                out.append(ch)
                escape = True
                i += 1
                continue

            if ch == '"':
                # настоящий конец строки или кавычка внутри значения?
                j = i + 1
                while j < n and s[j] in " \t\r\n":
                    j += 1

                # Проверяем, что после кавычки идёт конец значения или структуры
                if j >= n or s[j] in ",:}]":
                    in_string = False
                    out.append(ch)
                else:
                    # Кавычка внутри значения — экранируем
                    out.append('\\"')
                i += 1
                continue

            if ch == "'":
                # Проверяем, не является ли это началом строки в одинарных кавычках
                # (когда значение обернуто как '..."...')
                if not out or out[-1] not in ':,':
                    # Это одинарная кавычка внутри строки — оставляем как есть
                    out.append(ch)
                    i += 1
                    continue

            out.append(ch)
            i += 1
            continue

        # Пробельные символы вне строки
        if ch in " \t\r\n":
            out.append(ch)
            i += 1
            continue

        if ch == "{":
            stack.append("obj")
            expecting = "key"
            out.append(ch)
            i += 1
            continue

        if ch == "[":
            stack.append("arr")
            expecting = "value"
            out.append(ch)
            i += 1
            continue

        if ch in "}]":
            want = "obj" if ch == "}" else "arr"
            if stack and stack[-1] == want:
                stack.pop()
                out.append(ch)
                expecting = "comma" if stack else None
            i += 1
            continue

        if ch == ",":
            if expecting == "comma":
                expecting = "value" if (stack and stack[-1] == "arr") else "key"
                out.append(ch)
                i += 1
                continue

        if ch == ":":
            expecting = "value"
            out.append(ch)
            i += 1
            continue

        if ch == '"':
            in_string = True
            expecting = "colon" if expecting == "key" else "comma"
            out.append(ch)
            i += 1
            continue

        if ch == "'":
            # модель иногда оборачивает значение в одинарные кавычки: '...текст...'
            k = s.find("'", i + 1)
            if k == -1:
                k = n  # обрыв внутри такой строки — тянем до конца
            else:
                j = k + 1
                while j < n and s[j] in " \t\r\n":
                    j += 1
                if not (j >= n or s[j] in ",}]"):
                    out.append(ch)
                    i += 1
                    expecting = "comma"
                    continue

                inner = s[i + 1:k]
                # Экранируем двойные кавычки внутри одинарных
                ph = "\x01"
                inner = inner.replace('\\"', ph).replace('"', '\\"').replace(ph, '\\"')
                inner = inner.replace("\\'", "'")  # \' -> '
                out.append('"' + inner + '"')
                expecting = "comma"
                i = k + 1
                continue

        m = re.match(r"[-+0-9A-Za-z_.]+", s[i:])
        if m:
            out.append(m.group(0))
            i += m.end()
        else:
            out.append(ch)
            i += 1
        expecting = "comma"
        continue

    # --- обрыв: достраиваем хвост ---
    if escape:
        if out and out[-1] == "\\":
            out.pop()  # висячий backslash убираем

    if in_string:
        out.append('"')  # закрываем строку

    text = "".join(out)

    # недописанный литерал после двоеточия: tru -> true, nul -> null и т.д.
    m = re.search(r":\s*([A-Za-z]+)$", text)
    if m and m.group(1) not in ("true", "false", "null"):
        comp = {"t": "true", "tr": "true", "tru": "true",
                "f": "false", "fa": "false", "fal": "false",
                "n": "null", "nu": "null", "nul": "null"}.get(m.group(1))
        if comp:
            text = text[:m.start()] + ": " + comp

    if expecting == "key":
        text = text.rstrip()
        if text.endswith(","):
            text = text[:-1]
    elif expecting == "colon":
        text = text.rstrip() + ': ""'
    elif expecting == "value":
        text = text.rstrip()
        if re.search(r":\s*$", text):
            text += ' ""'

    if text.rstrip().endswith(","):
        text = text.rstrip()[:-1]

    for b in reversed(stack):
        text += "}" if b == "obj" else "]"

    return text


def validate_choice(value: str, allowed: set, param_name: str) -> str:
    if value not in allowed:
        raise ValueError(
            f"Недопустимое значение для {param_name}: '{value}'. "
            f"Разрешённые: {sorted(allowed)}"
        )
    return value
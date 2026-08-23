#!/usr/bin/env python3
"""
Motor de recomendación de libros.

CAMBIOS respecto a la versión anterior:
- Ya no existe "me gusta" como juicio a ciegas. Ahora un libro se marca
  "ya lo he leído" con una nota de 1 a 5. La nota determina si ese libro
  cuenta como señal POSITIVA o NEGATIVA para el cálculo, y con cuánto
  peso (3 = neutro, 5 = fuerte señal positiva, 1 = fuerte señal negativa,
  aunque lo hayas leído entero).
- El año de publicación ya no se inventa nunca (antes los libros de listas
  del NYT se marcaban con el año actual por defecto, lo cual era falso
  para libros de fondo de catálogo). Si no se sabe el año real, se deja
  sin año y no penaliza ni beneficia en el cálculo de época.
- Las recomendaciones se agrupan por categoría (misterio, histórico,
  fantasía...) en vez de una lista única mezclada. Las categorías que más
  coinciden con tu historial de lecturas positivas aparecen primero.

Sigue sin ser recomendación colaborativa ("la gente que leyó esto también
leyó esto otro") — ver README para el porqué. Es recomendación por
contenido: materias, autor, época y similitud de sinopsis, con la nota
media/nº de valoraciones públicas como señal extra.
"""
import json
import os
import re
import time
import datetime
import traceback
import urllib.request
import urllib.parse
import urllib.error

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROFILES_DIR = "data/profiles"
RECS_DIR = "data/recommendations"
HEADERS = {"User-Agent": "personal-book-recs/1.0 (uso personal, no comercial)"}

NYT_API_KEY = os.environ.get("NYT_API_KEY", "").strip()
GOOGLE_BOOKS_API_KEY = os.environ.get("GOOGLE_BOOKS_API_KEY", "").strip()

MAX_RECS_PER_CATEGORY = 5
MAX_CATEGORIES = 4
CANDIDATE_POOL_TARGET = 150
NEW_RELEASE_WINDOW_DAYS = 35
YEAR_RANGE_PADDING = 12

# Categorías controladas. Un libro se asigna a la categoría con más
# palabras clave coincidentes en sus materias. Es un mapeo simple por
# palabras clave, no un modelo de IA -- transparente y fácil de ajustar.
CATEGORY_KEYWORDS = {
    "Misterio y thriller": ["mystery", "detective", "thriller", "crime", "suspense", "noir",
                             "misterio", "intriga", "policiaca", "policial", "asesinato"],
    "Fantasía": ["fantasy", "magic", "dragon", "sword and sorcery", "fantasia", "magia", "epica"],
    "Ciencia ficción": ["science fiction", "sci-fi", "dystopia", "space opera",
                         "ciencia ficcion", "distopia", "espacio"],
    "Histórico": ["historical fiction", "history", "war", "historia", "historico",
                  "guerra", "epoca"],
    "Romance": ["romance", "love stories", "romantica"],
    "Terror": ["horror", "terror", "gothic", "ghost stories", "gotico"],
    "Clásicos": ["classics", "classic literature", "clasico", "literatura clasica"],
    "No ficción": ["biography", "memoir", "essays", "nonfiction", "true crime",
                    "biografia", "ensayo", "no ficcion"],
    "Infantil y juvenil": ["juvenile fiction", "children's books", "young adult",
                            "infantil", "juvenil"],
    "Drama y literatura general": ["fiction", "drama", "literary fiction", "novela", "literatura"],
}
DEFAULT_CATEGORY = "Otros"


# ---------------------------------------------------------------- utils --
def http_get_json(url, retries=3):
    req = urllib.request.Request(url, headers=HEADERS)
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                if attempt == retries:
                    print(f"  aviso: 429 persistente en {url[:90]}..., se abandona")
                    return None
                wait = 4 * (attempt + 1)
                print(f"  aviso: 429, esperando {wait}s...")
                time.sleep(wait)
                continue
            if attempt == retries:
                print(f"  aviso: fallo al pedir {url[:90]}... -> {e}")
                return None
            time.sleep(1.5)
        except Exception as e:
            if attempt == retries:
                print(f"  aviso: fallo al pedir {url[:90]}... -> {e}")
                return None
            time.sleep(1.5)
    return None


def norm_subject(s):
    return re.sub(r"\s+", " ", s.strip().lower())


def parse_year(value):
    if value is None:
        return None
    m = re.search(r"(1[5-9]\d{2}|20\d{2})", str(value))
    return int(m.group(1)) if m else None


def parse_date_best_effort(value):
    if not value:
        return None
    value = str(value)
    try:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            return datetime.date.fromisoformat(value)
        if re.fullmatch(r"\d{4}-\d{2}", value):
            y, m = value.split("-")
            return datetime.date(int(y), int(m), 28)
        if re.fullmatch(r"\d{4}", value):
            return datetime.date(int(value), 12, 31)
    except Exception:
        return None
    return None


def book_text(b):
    return " ".join([
        b.get("title", ""),
        b.get("author", ""),
        " ".join(b.get("subjects", [])[:15]),
        (b.get("description") or "")[:600],
    ])


def categorize(subjects):
    subj_text = " ".join(subjects or []).lower()
    best_cat, best_hits = DEFAULT_CATEGORY, 0
    for cat, kws in CATEGORY_KEYWORDS.items():
        hits = sum(1 for kw in kws if kw in subj_text)
        if hits > best_hits:
            best_hits, best_cat = hits, cat
    return best_cat


# ------------------------------------------------------ fuentes de datos --
def ol_search(query, limit=10):
    url = ("https://openlibrary.org/search.json?q=" + urllib.parse.quote(query) +
           f"&limit={limit}&fields=key,title,author_name,first_publish_year,subject,cover_i,ratings_average,ratings_count")
    data = http_get_json(url)
    out = []
    if not data:
        return out
    for d in data.get("docs", []):
        out.append({
            "id": "ol:" + d.get("key", ""),
            "title": d.get("title", "Sin título"),
            "author": ", ".join(d.get("author_name", []) or ["Desconocido"]),
            "subjects": [norm_subject(s) for s in (d.get("subject") or [])[:20]],
            "description": "",
            "cover_url": (f"https://covers.openlibrary.org/b/id/{d['cover_i']}-M.jpg"
                          if d.get("cover_i") else ""),
            "rating_avg": d.get("ratings_average") or 0,
            "rating_count": d.get("ratings_count") or 0,
            "year": d.get("first_publish_year"),
            "source": "openlibrary",
        })
    return out


def ol_subject_works(subject, limit=25):
    slug = re.sub(r"[^a-z0-9]+", "_", subject.lower()).strip("_")
    if not slug:
        return []
    url = f"https://openlibrary.org/subjects/{urllib.parse.quote(slug)}.json?limit={limit}"
    data = http_get_json(url)
    out = []
    if not data:
        return out
    for w in data.get("works", []):
        out.append({
            "id": "ol:" + w.get("key", ""),
            "title": w.get("title", "Sin título"),
            "author": ", ".join(a.get("name", "") for a in (w.get("authors") or [])) or "Desconocido",
            "subjects": [norm_subject(s) for s in (w.get("subject") or [])[:20]],
            "description": "",
            "cover_url": (f"https://covers.openlibrary.org/b/id/{w['cover_id']}-M.jpg"
                          if w.get("cover_id") else ""),
            "rating_avg": 0,
            "rating_count": 0,
            "year": w.get("first_publish_year"),
            "source": "openlibrary",
        })
    return out


def gb_search(query, order="relevance", limit=20, lang=None):
    params = {"q": query, "maxResults": min(limit, 40), "orderBy": order, "printType": "books"}
    if lang:
        params["langRestrict"] = lang
    if GOOGLE_BOOKS_API_KEY:
        params["key"] = GOOGLE_BOOKS_API_KEY
    url = "https://www.googleapis.com/books/v1/volumes?" + urllib.parse.urlencode(params)
    data = http_get_json(url)
    time.sleep(0.35 if GOOGLE_BOOKS_API_KEY else 1.2)
    out = []
    if not data:
        return out
    for item in data.get("items", []):
        vi = item.get("volumeInfo", {})
        out.append({
            "id": "gb:" + item.get("id", ""),
            "title": vi.get("title", "Sin título"),
            "author": ", ".join(vi.get("authors", []) or ["Desconocido"]),
            "subjects": [norm_subject(s) for s in (vi.get("categories") or [])],
            "description": vi.get("description", ""),
            "cover_url": (vi.get("imageLinks") or {}).get("thumbnail", ""),
            "rating_avg": vi.get("averageRating") or 0,
            "rating_count": vi.get("ratingsCount") or 0,
            "year": parse_year(vi.get("publishedDate")),
            "published_date": vi.get("publishedDate", ""),
            "language": vi.get("language", ""),
            "source": "googlebooks",
        })
    return out


def nyt_new_releases():
    """Listas actuales del NYT. OJO: el NYT no da un año de publicación
    fiable por libro en este endpoint -- antes se rellenaba con el año
    actual como aproximación, pero eso era FALSO para libros de fondo de
    catálogo que llevan tiempo en la lista. Ahora se deja sin año (None)
    y se intenta recuperar el año real más adelante, en el enriquecimiento
    contra Google Books/Open Library."""
    if not NYT_API_KEY:
        print("  aviso: no hay NYT_API_KEY configurada, se omiten novedades NYT")
        return []
    url = f"https://api.nytimes.com/svc/books/v3/lists/overview.json?api-key={NYT_API_KEY}"
    data = http_get_json(url)
    out = []
    if not data:
        return out
    for lst in data.get("results", {}).get("lists", []):
        for b in lst.get("books", []):
            out.append({
                "id": "nyt:" + (b.get("primary_isbn13") or b.get("title", "")),
                "title": b.get("title", "Sin título").title(),
                "author": b.get("author", "Desconocido"),
                "subjects": [norm_subject(lst.get("list_name", ""))],
                "description": b.get("description", ""),
                "cover_url": b.get("book_image", ""),
                "rating_avg": 0,
                "rating_count": 0,
                "year": None,  # se recupera de verdad en el enriquecimiento
                "published_date": "",
                "source": "nyt",
            })
    return out


def ol_find_spanish_edition(title, author_first):
    try:
        query = f"{title} {author_first}".strip()
        url = ("https://openlibrary.org/search.json?q=" + urllib.parse.quote(query) +
               "&limit=3&fields=key,title")
        data = http_get_json(url)
        time.sleep(0.3)
        if not data:
            return None
        for doc in data.get("docs", [])[:3]:
            work_key = doc.get("key")
            if not work_key:
                continue
            ed_data = http_get_json(f"https://openlibrary.org{work_key}/editions.json?limit=50")
            time.sleep(0.3)
            if not ed_data:
                continue
            for ed in ed_data.get("entries", []):
                langs = [l.get("key", "") for l in (ed.get("languages") or [])]
                if any("spa" in l for l in langs) and ed.get("title"):
                    return ed["title"]
    except Exception as e:
        print(f"    aviso: fallo buscando edición en español -> {e}")
    return None


def enrich_book(candidate):
    """Título en español + sinopsis + año real. Se llama solo sobre el
    top final de cada categoría, no sobre todo el pool. Nunca lanza
    excepción: si todo falla, devuelve el candidato tal cual."""
    title = candidate.get("title", "")
    author_first = (candidate.get("author") or "").split(",")[0].strip()
    if not title:
        return candidate

    try:
        titulo_es = ol_find_spanish_edition(title, author_first)
        if titulo_es:
            candidate["title"] = titulo_es
    except Exception as e:
        print(f"    aviso: no se pudo buscar título en español de '{title}' -> {e}")

    try:
        best = None
        field_query = f'intitle:"{title}"' + (f' inauthor:"{author_first}"' if author_first else "")

        if not candidate.get("description"):
            es_results = gb_search(field_query, limit=3, lang="es")
            best = es_results[0] if es_results else None
            if not best:
                any_results = gb_search(field_query, limit=3)
                best = any_results[0] if any_results else None
            if not best:
                plain_results = gb_search(f"{title} {author_first}".strip(), limit=3)
                best = plain_results[0] if plain_results else None

        if best:
            if best.get("description") and not candidate.get("description"):
                candidate["description"] = best["description"]
            if best.get("cover_url") and not candidate.get("cover_url"):
                candidate["cover_url"] = best["cover_url"]
            if not candidate.get("rating_count"):
                candidate["rating_count"] = best.get("rating_count", 0)
                candidate["rating_avg"] = best.get("rating_avg", 0)

        # el año se intenta recuperar SIEMPRE que falte, tanto si hubo
        # "best" de la búsqueda de sinopsis como si no
        if not parse_year(candidate.get("year")):
            year_source = best
            if not year_source:
                yr_results = gb_search(f"{title} {author_first}".strip(), limit=3)
                year_source = yr_results[0] if yr_results else None
            if year_source and year_source.get("year"):
                candidate["year"] = year_source["year"]
    except Exception as e:
        print(f"    aviso: no se pudo enriquecer '{title}' -> {e}")

    return candidate


# ------------------------------------------------------------- perfiles --
def load_profile(path):
    with open(path, "r") as f:
        data = json.load(f)
    # migración suave desde el esquema antiguo (liked/disliked binarios)
    if "read" not in data:
        data["read"] = [{**b, "score": b.get("score", 4)} for b in data.get("liked", [])]
    data.setdefault("seed_books", [])
    data.setdefault("disliked", [])
    data.setdefault("shown_ids", [])
    return data


def weighted_read(profile):
    """Cada libro 'ya leído' tiene una nota 1-5 (3=neutro). Se convierte
    en (libro, peso) con signo: notas altas -> peso positivo, notas
    bajas -> peso negativo, aunque el libro esté 'leído'. Los libros
    semilla se tratan como nota 5 fija (los elegiste tú a propósito como
    favoritos, no hace falta puntuarlos)."""
    positive, negative = [], []
    for b in profile.get("seed_books", []):
        positive.append((b, float(b.get("score", 5) - 3)))  # 5★ por defecto -> peso 2.0
    for b in profile.get("read", []):
        score = b.get("score", 3)
        weight = score - 3
        if weight > 0:
            positive.append((b, float(weight)))
        elif weight < 0:
            negative.append((b, float(-weight)))
    return positive, negative


def top_subjects_weighted(weighted_books, n=8):
    counts = {}
    for b, w in weighted_books:
        for s in b.get("subjects", []):
            counts[s] = counts.get(s, 0) + w
    return [s for s, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:n]]


def compute_year_range(weighted_positive):
    years = [parse_year(b.get("year")) for b, _ in weighted_positive]
    years = [y for y in years if y]
    if len(years) < 2:
        return None
    lo, hi = min(years), max(years)
    return (lo - YEAR_RANGE_PADDING, hi + YEAR_RANGE_PADDING)


def category_weights(weighted_positive):
    weights = {}
    for b, w in weighted_positive:
        cat = categorize(b.get("subjects", []))
        weights[cat] = weights.get(cat, 0) + w
    total = sum(weights.values()) or 1
    return {k: v / total for k, v in weights.items()}


def build_candidate_pool(positive_subjects, positive_authors=None):
    pool = {}

    def add_all(items):
        for it in items:
            if it["id"] and it["id"] not in pool:
                pool[it["id"]] = it

    add_all(nyt_new_releases())
    for subj in positive_subjects[:6]:
        add_all(ol_subject_works(subj, limit=20))
        add_all(gb_search(f"subject:{subj}", order="newest", limit=15))
        if len(pool) >= CANDIDATE_POOL_TARGET:
            break

    if not positive_subjects and positive_authors:
        # último recurso: si no tenemos ninguna materia (por eso no hay
        # nada en el pool todavía), buscamos por autor para no quedarnos
        # sin candidatos.
        print("    sin materias disponibles, buscando por autor como alternativa")
        for author in positive_authors[:5]:
            add_all(gb_search(f'inauthor:"{author}"', order="newest", limit=15))
            add_all(ol_search(author, limit=15))
    return list(pool.values())


def build_new_release_pool(positive_subjects):
    pool = {}
    cutoff = datetime.date.today() - datetime.timedelta(days=NEW_RELEASE_WINDOW_DAYS)
    for subj in positive_subjects[:6]:
        for it in gb_search(f"subject:{subj}", order="newest", limit=20):
            d = parse_date_best_effort(it.get("published_date"))
            if d and d >= cutoff and it["id"] not in pool:
                pool[it["id"]] = it
    for it in nyt_new_releases():
        if it["id"] not in pool:
            pool[it["id"]] = it
    return list(pool.values())


def score_candidates(candidates, weighted_positive, weighted_negative, year_range=None, popularity_weight=0.6):
    if not candidates:
        return []

    pos_subjects = set(s for b, _ in weighted_positive for s in b.get("subjects", []))
    neg_subjects = set(s for b, _ in weighted_negative for s in b.get("subjects", []))
    pos_authors = set(b.get("author", "").lower() for b, _ in weighted_positive if b.get("author"))

    pos_books = [b for b, _ in weighted_positive]
    neg_books = [b for b, _ in weighted_negative]
    pos_w = np.array([w for _, w in weighted_positive]) if weighted_positive else np.array([])
    neg_w = np.array([w for _, w in weighted_negative]) if weighted_negative else np.array([])

    corpus = [book_text(b) for b in pos_books] + [book_text(b) for b in neg_books] + [book_text(c) for c in candidates]
    pos_n, neg_n = len(pos_books), len(neg_books)

    tfidf_pos_sim = np.zeros(len(candidates))
    tfidf_neg_sim = np.zeros(len(candidates))
    if any(c.strip() for c in corpus) and (pos_n + neg_n) > 0:
        try:
            vec = TfidfVectorizer(max_features=4000, stop_words=None)
            mat = vec.fit_transform(corpus)
            cand_mat = mat[pos_n + neg_n:]
            if pos_n > 0:
                sims = cosine_similarity(cand_mat, mat[:pos_n])
                tfidf_pos_sim = (sims * pos_w).sum(axis=1) / pos_w.sum()
            if neg_n > 0:
                sims = cosine_similarity(cand_mat, mat[pos_n:pos_n + neg_n])
                tfidf_neg_sim = (sims * neg_w).sum(axis=1) / neg_w.sum()
        except Exception as e:
            print(f"  aviso: fallo calculando similitud de texto -> {e}")

    scored = []
    for i, c in enumerate(candidates):
        subj = set(c.get("subjects", []))
        subj_pos_overlap = len(subj & pos_subjects)
        subj_neg_overlap = len(subj & neg_subjects)
        author_match = 1 if c.get("author", "").lower() in pos_authors else 0
        rating_count = c.get("rating_count") or 0
        popularity_boost = min(1.0, (rating_count ** 0.3) / 20) if rating_count else 0

        year_penalty = 0.0
        cy = parse_year(c.get("year"))
        if year_range and cy:
            lo, hi = year_range
            if cy < lo:
                year_penalty = min(1.5, (lo - cy) / 25)
            elif cy > hi:
                year_penalty = min(1.5, (cy - hi) / 25)

        score = (
            1.6 * subj_pos_overlap
            - 2.2 * subj_neg_overlap
            + 1.8 * tfidf_pos_sim[i]
            - 1.3 * tfidf_neg_sim[i]
            + 1.0 * author_match
            + popularity_weight * popularity_boost
            - 1.4 * year_penalty
        )
        c2 = dict(c)
        c2["score"] = round(float(score), 4)
        c2["category"] = categorize(c.get("subjects", []))
        scored.append(c2)

    scored.sort(key=lambda c: -c["score"])
    return scored


def synopsis_short(text, n_words=6):
    words = (text or "").split()
    if not words:
        return "Sin sinopsis disponible."
    short = " ".join(words[:n_words])
    return short + ("…" if len(words) > n_words else "")


def group_by_category(scored, cat_weights, max_categories=MAX_CATEGORIES, per_category=MAX_RECS_PER_CATEGORY):
    """Agrupa candidatos ya puntuados por categoría, y ordena las
    categorías según cuánto coinciden con tu historial de lecturas
    positivas (las que más te gustan salen primero)."""
    by_cat = {}
    for c in scored:
        by_cat.setdefault(c["category"], []).append(c)

    # orden de categorías: primero las que tienen peso en tu historial
    # (de mayor a menor), luego el resto que tengan candidatos, por si
    # acaso hay hueco / quieres descubrir algo nuevo
    ranked_known = sorted(
        [cat for cat in by_cat if cat in cat_weights],
        key=lambda cat: -cat_weights[cat]
    )
    other_cats = [cat for cat in by_cat if cat not in cat_weights]
    ordered_cats = (ranked_known + other_cats)[:max_categories]

    result = []
    used_ids = set()
    for cat in ordered_cats:
        items = [c for c in by_cat[cat] if c["id"] not in used_ids][:per_category]
        if not items:
            continue
        used_ids.update(c["id"] for c in items)
        result.append({"name": cat, "weight": round(cat_weights.get(cat, 0), 3), "items": items})
    return result


def enrich_and_format(scored_group_items):
    out = []
    for c in scored_group_items:
        c = enrich_book(dict(c))
        out.append({
            "id": c["id"],
            "title": c["title"],
            "author": c["author"],
            "cover_url": c.get("cover_url", ""),
            "subjects": c.get("subjects", [])[:8],
            "year": parse_year(c.get("year")),
            "category": c.get("category"),
            "synopsis_short": synopsis_short(c.get("description", "") or ""),
            "synopsis_full": c.get("description") or "No hay sinopsis disponible para este libro.",
            "source": c.get("source"),
            "score": c.get("score"),
        })
    con_sinopsis = sum(1 for it in out if it["synopsis_full"] != "No hay sinopsis disponible para este libro.")
    print(f"    sinopsis conseguidas: {con_sinopsis}/{len(out)}")
    return out


def backfill_book_metadata(books):
    """Si un libro tuyo (semilla o leído) no tiene año y/o materias
    guardadas -típicamente porque vino de un resultado de Google Books,
    que muy a menudo no rellena el campo de categorías-, se busca aquí
    mismo en Open Library, que suele tener mejores materias. Muta los
    diccionarios in-place, así que al guardar el perfil de vuelta, ya no
    hace falta repetir la búsqueda la próxima vez."""
    for b in books:
        needs_year = not parse_year(b.get("year"))
        needs_subjects = not b.get("subjects")
        if not needs_year and not needs_subjects:
            continue
        title = (b.get("title") or "").strip()
        if not title:
            continue
        author_first = (b.get("author") or "").split(",")[0].strip()
        try:
            results = ol_search(f"{title} {author_first}".strip(), limit=3)
            if not results:
                results = gb_search(f"{title} {author_first}".strip(), limit=3)
            if results:
                r = results[0]
                if needs_year and r.get("year"):
                    b["year"] = r["year"]
                    print(f"    año recuperado para '{title}': {r['year']}")
                if needs_subjects and r.get("subjects"):
                    b["subjects"] = r["subjects"]
                    print(f"    materias recuperadas para '{title}': {r['subjects'][:5]}")
        except Exception as e:
            print(f"    aviso: no se pudo completar '{title}' -> {e}")


def process_profile(username):
    path = os.path.join(PROFILES_DIR, f"{username}.json")
    profile = load_profile(path)

    weighted_positive, weighted_negative = weighted_read(profile)
    if not weighted_positive:
        print(f"  {username}: sin libros semilla ni lecturas valoradas positivamente, se omite")
        return

    print(f"  {username}: comprobando año/materias de tus libros...")
    backfill_book_metadata([b for b, _ in weighted_positive] + [b for b, _ in weighted_negative])

    pos_subjects = top_subjects_weighted(weighted_positive, n=8)
    pos_authors = list({b.get("author", "").strip() for b, _ in weighted_positive if b.get("author")})
    year_range = compute_year_range(weighted_positive)
    cat_weights = category_weights(weighted_positive)
    print(f"  {username}: materias favoritas -> {pos_subjects}")
    print(f"  {username}: rango de época -> {year_range or 'sin datos suficientes, sin filtrar'}")
    print(f"  {username}: pesos por categoría -> { {k: round(v,2) for k,v in cat_weights.items()} }")

    shown_ids = set(profile.get("shown_ids", []))
    known_ids = set(b["id"] for b, _ in weighted_positive) | set(b["id"] for b, _ in weighted_negative) \
        | set(b["id"] for b in profile.get("disliked", []))

    all_negative = weighted_negative + [(b, 2.0) for b in profile.get("disliked", [])]

    # ---- lista principal "para ti" ----
    candidates = build_candidate_pool(pos_subjects, pos_authors)
    candidates = [c for c in candidates if c["id"] not in shown_ids and c["id"] not in known_ids]
    print(f"  {username}: {len(candidates)} candidatos nuevos (principal)")
    scored = score_candidates(candidates, weighted_positive, all_negative, year_range=year_range, popularity_weight=0.6)
    main_groups_raw = group_by_category(scored, cat_weights)
    main_groups = [{"name": g["name"], "weight": g["weight"], "items": enrich_and_format(g["items"])}
                   for g in main_groups_raw]

    os.makedirs(RECS_DIR, exist_ok=True)
    with open(os.path.join(RECS_DIR, f"{username}.json"), "w") as f:
        json.dump({
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
            "categories": main_groups,
        }, f, ensure_ascii=False, indent=2)
    total_main = sum(len(g["items"]) for g in main_groups)
    print(f"  {username}: {total_main} recomendaciones principales en {len(main_groups)} categorías")

    # ---- novedades ----
    shown_after_main = shown_ids | {it["id"] for g in main_groups for it in g["items"]}
    new_candidates = build_new_release_pool(pos_subjects)
    new_candidates = [c for c in new_candidates if c["id"] not in shown_after_main and c["id"] not in known_ids]
    print(f"  {username}: {len(new_candidates)} candidatos nuevos (novedades)")
    scored_new = score_candidates(new_candidates, weighted_positive, all_negative, year_range=None, popularity_weight=0.0)
    new_groups_raw = group_by_category(scored_new, cat_weights)
    new_groups = [{"name": g["name"], "weight": g["weight"], "items": enrich_and_format(g["items"])}
                  for g in new_groups_raw]

    with open(os.path.join(RECS_DIR, f"{username}-nuevos.json"), "w") as f:
        json.dump({
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
            "categories": new_groups,
        }, f, ensure_ascii=False, indent=2)
    total_new = sum(len(g["items"]) for g in new_groups)
    print(f"  {username}: {total_new} novedades en {len(new_groups)} categorías")

    # ---- actualizar shown_ids ----
    all_shown = shown_after_main | {it["id"] for g in new_groups for it in g["items"]}
    profile["shown_ids"] = list(all_shown)
    with open(path, "w") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)


def main():
    if not os.path.isdir(PROFILES_DIR):
        print("No existe data/profiles/, nada que hacer todavía.")
        return
    profiles = [f[:-5] for f in os.listdir(PROFILES_DIR) if f.endswith(".json")]
    if not profiles:
        print("No hay ningún perfil en data/profiles/ todavía.")
        return
    for username in profiles:
        print(f"Procesando perfil: {username}")
        try:
            process_profile(username)
        except Exception as e:
            print(f"  ERROR procesando {username}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()

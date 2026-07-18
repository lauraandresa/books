#!/usr/bin/env python3
"""
Motor de recomendación de libros.

Para cada archivo data/profiles/<usuario>.json genera DOS listas:

  1. data/recommendations/<usuario>.json        -> top 10 "para ti"
     (puntuado por similitud de contenido + época + popularidad pública)
  2. data/recommendations/<usuario>-nuevos.json  -> top 10 "novedades"
     (libros publicados en las últimas ~5 semanas que encajan con tu
     gusto, sin exigir valoraciones porque son demasiado recientes para
     tenerlas)

No es recomendación colaborativa ("la gente que leyó esto también leyó
esto otro") — esa información es propiedad de Goodreads/Amazon y no existe
de forma gratuita y legal en ningún sitio (ver README). Es recomendación
por contenido: materias, autor, época de publicación y similitud del
texto de la sinopsis, con la nota media / nº de valoraciones públicas como
señal extra de popularidad.

Fuentes, todas gratuitas y dentro de sus términos de uso:
  - Open Library API   (sin clave)
  - Google Books API   (sin clave; opcional GOOGLE_BOOKS_API_KEY para más cuota)
  - NYT Books API       (clave gratuita obligatoria: NYT_API_KEY)
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

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROFILES_DIR = "data/profiles"
RECS_DIR = "data/recommendations"
HEADERS = {"User-Agent": "personal-book-recs/1.0 (uso personal, no comercial)"}

NYT_API_KEY = os.environ.get("NYT_API_KEY", "").strip()
GOOGLE_BOOKS_API_KEY = os.environ.get("GOOGLE_BOOKS_API_KEY", "").strip()

MAX_RECS = 10
CANDIDATE_POOL_TARGET = 120
NEW_RELEASE_WINDOW_DAYS = 35
YEAR_RANGE_PADDING = 12  # años de margen a cada lado del rango de tus libros


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
                    print(f"  aviso: 429 persistente en {url[:90]}..., se abandona esta petición")
                    return None
                wait = 4 * (attempt + 1)
                print(f"  aviso: 429 (demasiadas peticiones), esperando {wait}s...")
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
    """Intenta sacar un año (int) de cualquier formato: 1985, '1985',
    '2024-05', '2024-05-13'... Devuelve None si no hay forma."""
    if value is None:
        return None
    m = re.search(r"(1[5-9]\d{2}|20\d{2})", str(value))
    return int(m.group(1)) if m else None


def parse_date_best_effort(value):
    """Devuelve un datetime.date aproximado a partir de 'YYYY', 'YYYY-MM'
    o 'YYYY-MM-DD'. Si falta mes/día, asume el más reciente posible del
    trozo que sí hay (útil para no descartar novedades por precisión baja)."""
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
    # Pequeño respiro entre peticiones a Google Books: sin esto, en una
    # sola ejecución se pueden lanzar 50-80 peticiones seguidas y saltar
    # el límite de peticiones por segundo aunque la cuota diaria total
    # (con clave) sea de sobra suficiente.
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
    if not NYT_API_KEY:
        print("  aviso: no hay NYT_API_KEY configurada, se omiten novedades NYT")
        return []
    url = f"https://api.nytimes.com/svc/books/v3/lists/overview.json?api-key={NYT_API_KEY}"
    data = http_get_json(url)
    out = []
    if not data:
        return out
    today = datetime.date.today()
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
                "year": today.year,
                "published_date": today.isoformat(),
                "source": "nyt",
            })
    return out


def ol_find_spanish_edition(title, author_first):
    """Busca la 'obra' en Open Library y revisa sus distintas ediciones
    para encontrar una en español. A diferencia de buscar por texto en
    Google Books, esto SÍ funciona para títulos traducidos, porque Open
    Library agrupa todas las ediciones (en cualquier idioma) bajo la misma
    obra — no hace falta adivinar cómo se llama la traducción."""
    try:
        query = f"{title} {author_first}".strip()
        url = ("https://openlibrary.org/search.json?q=" + urllib.parse.quote(query) +
               "&limit=3&fields=key,title")
        data = http_get_json(url)
        time.sleep(0.3)
        if not data:
            return None
        for doc in data.get("docs", [])[:3]:
            work_key = doc.get("key")  # p.ej. "/works/OL12345W"
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
        print(f"    aviso: fallo buscando edición en español (Open Library) -> {e}")
    return None


def enrich_spanish(candidate):
    """Título en español (vía Open Library, buscando ediciones de la
    misma obra) + sinopsis (vía Google Books, en el idioma que sea con
    tal de tener alguna). Se llama solo sobre el top final (10-15 libros),
    no sobre todo el pool de candidatos. Nunca lanza excepción: si todo
    falla, devuelve el candidato tal cual, con su título original."""
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

        es_results = gb_search(field_query, limit=3, lang="es")
        best = es_results[0] if es_results else None

        if not best:
            any_results = gb_search(field_query, limit=3)
            best = any_results[0] if any_results else None

        if not best:
            # tercer intento, más permisivo: sin restringir a campos exactos
            # (algunos títulos con caracteres raros no casan bien con intitle:)
            plain_query = f"{title} {author_first}".strip()
            plain_results = gb_search(plain_query, limit=3)
            best = plain_results[0] if plain_results else None

        if best:
            if best.get("description") and not candidate.get("description"):
                candidate["description"] = best["description"]
            if best.get("cover_url") and not candidate.get("cover_url"):
                candidate["cover_url"] = best["cover_url"]
            if not candidate.get("rating_count"):
                candidate["rating_count"] = best.get("rating_count", 0)
                candidate["rating_avg"] = best.get("rating_avg", 0)
            if not candidate.get("year"):
                candidate["year"] = best.get("year")
    except Exception as e:
        print(f"    aviso: no se pudo enriquecer '{title}' -> {e}")

    return candidate


# ------------------------------------------------------------- perfiles --
def load_profile(path):
    with open(path, "r") as f:
        return json.load(f)


def top_subjects(books, n=8):
    counts = {}
    for b in books:
        for s in b.get("subjects", []):
            counts[s] = counts.get(s, 0) + 1
    return [s for s, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:n]]


def compute_year_range(books):
    """Rango de años que sueles leer, con un margen. None si no hay datos
    suficientes (en ese caso no se penaliza por época a nadie)."""
    years = [parse_year(b.get("year")) for b in books]
    years = [y for y in years if y]
    if len(years) < 2:
        return None
    lo, hi = min(years), max(years)
    return (lo - YEAR_RANGE_PADDING, hi + YEAR_RANGE_PADDING)


def build_candidate_pool(positive_subjects):
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

    return list(pool.values())


def build_new_release_pool(positive_subjects):
    """Solo libros publicados en la ventana de novedades reciente."""
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


def score_candidates(candidates, liked, disliked, year_range=None, popularity_weight=0.6):
    if not candidates:
        return []

    pos_subjects = set(s for b in liked for s in b.get("subjects", []))
    neg_subjects = set(s for b in disliked for s in b.get("subjects", []))
    pos_authors = set(b.get("author", "").lower() for b in liked if b.get("author"))

    corpus = [book_text(b) for b in liked] + [book_text(b) for b in disliked] + [book_text(c) for c in candidates]
    pos_n, neg_n = len(liked), len(disliked)

    tfidf_pos_sim = [0.0] * len(candidates)
    tfidf_neg_sim = [0.0] * len(candidates)
    if any(c.strip() for c in corpus) and (pos_n + neg_n) > 0:
        try:
            vec = TfidfVectorizer(max_features=4000, stop_words=None)
            mat = vec.fit_transform(corpus)
            pos_mat = mat[:pos_n] if pos_n else None
            neg_mat = mat[pos_n:pos_n + neg_n] if neg_n else None
            cand_mat = mat[pos_n + neg_n:]
            if pos_mat is not None and pos_mat.shape[0] > 0:
                sims = cosine_similarity(cand_mat, pos_mat)
                tfidf_pos_sim = sims.mean(axis=1).tolist()
            if neg_mat is not None and neg_mat.shape[0] > 0:
                sims = cosine_similarity(cand_mat, neg_mat)
                tfidf_neg_sim = sims.mean(axis=1).tolist()
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
        c2["score"] = round(score, 4)
        scored.append(c2)

    scored.sort(key=lambda c: -c["score"])
    return scored


def synopsis_short(text, n_words=6):
    words = (text or "").split()
    if not words:
        return "Sin sinopsis disponible."
    short = " ".join(words[:n_words])
    return short + ("…" if len(words) > n_words else "")


def to_output_items(scored_top):
    out_items = []
    for c in scored_top:
        c = enrich_spanish(dict(c))
        out_items.append({
            "id": c["id"],
            "title": c["title"],
            "author": c["author"],
            "cover_url": c.get("cover_url", ""),
            "subjects": c.get("subjects", [])[:8],
            "year": parse_year(c.get("year")),
            "synopsis_short": synopsis_short(c.get("description", "") or ""),
            "synopsis_full": c.get("description") or "No hay sinopsis disponible para este libro.",
            "source": c.get("source"),
            "score": c.get("score"),
        })
    con_sinopsis = sum(1 for it in out_items if it["synopsis_full"] != "No hay sinopsis disponible para este libro.")
    print(f"    sinopsis conseguidas: {con_sinopsis}/{len(out_items)}")
    return out_items


def backfill_years(books):
    """Si un libro (semilla o 'me gusta') no tiene año guardado —típicamente
    porque se añadió antes de que la app empezara a guardarlo—, lo busca
    aquí mismo. Muta los diccionarios in-place, así que al escribir el
    perfil de vuelta al final, el año queda guardado para no tener que
    volver a buscarlo la próxima semana."""
    for b in books:
        if parse_year(b.get("year")):
            continue
        title = (b.get("title") or "").strip()
        if not title:
            continue
        author_first = (b.get("author") or "").split(",")[0].strip()
        query = f"{title} {author_first}".strip()
        year = None
        try:
            for r in gb_search(query, limit=3):
                year = parse_year(r.get("year"))
                if year:
                    break
            if not year:
                for r in ol_search(query, limit=3):
                    year = parse_year(r.get("year"))
                    if year:
                        break
        except Exception as e:
            print(f"    aviso: no se pudo recuperar año de '{title}' -> {e}")
        if year:
            b["year"] = year
            print(f"    año recuperado para '{title}': {year}")
        else:
            print(f"    no se encontró año para '{title}' (quedará fuera del cálculo de época)")


def process_profile(username):
    path = os.path.join(PROFILES_DIR, f"{username}.json")
    profile = load_profile(path)

    seed = profile.get("seed_books", [])
    liked = profile.get("liked", [])
    disliked = profile.get("disliked", [])
    shown_ids = set(profile.get("shown_ids", []))

    positive_all = seed + liked
    if not positive_all:
        print(f"  {username}: sin libros semilla ni 'me gusta' todavía, se omite")
        return

    print(f"  {username}: comprobando años de publicación...")
    backfill_years(positive_all)

    pos_subjects = top_subjects(positive_all, n=8)
    year_range = compute_year_range(positive_all)
    print(f"  {username}: materias favoritas -> {pos_subjects}")
    print(f"  {username}: rango de época preferido -> {year_range or 'sin datos suficientes, sin filtrar'}")

    known_ids = set(b["id"] for b in positive_all + disliked)

    # ---- lista principal "para ti" ----
    candidates = build_candidate_pool(pos_subjects)
    candidates = [c for c in candidates if c["id"] not in shown_ids and c["id"] not in known_ids]
    print(f"  {username}: {len(candidates)} candidatos nuevos (principal)")
    scored = score_candidates(candidates, positive_all, disliked, year_range=year_range, popularity_weight=0.6)
    top_main = scored[:MAX_RECS]
    main_items = to_output_items(top_main)

    os.makedirs(RECS_DIR, exist_ok=True)
    with open(os.path.join(RECS_DIR, f"{username}.json"), "w") as f:
        json.dump({
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
            "items": main_items,
        }, f, ensure_ascii=False, indent=2)
    print(f"  {username}: {len(main_items)} recomendaciones principales guardadas")

    # ---- lista de novedades (sin exigir popularidad, sin filtrar por época) ----
    new_candidates = build_new_release_pool(pos_subjects)
    new_candidates = [c for c in new_candidates if c["id"] not in shown_ids and c["id"] not in known_ids
                       and c["id"] not in {m["id"] for m in main_items}]
    print(f"  {username}: {len(new_candidates)} candidatos nuevos (novedades)")
    scored_new = score_candidates(new_candidates, positive_all, disliked, year_range=None, popularity_weight=0.0)
    top_new = scored_new[:MAX_RECS]
    new_items = to_output_items(top_new)

    with open(os.path.join(RECS_DIR, f"{username}-nuevos.json"), "w") as f:
        json.dump({
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
            "items": new_items,
        }, f, ensure_ascii=False, indent=2)
    print(f"  {username}: {len(new_items)} novedades guardadas")

    # ---- actualizar shown_ids para no repetir en semanas futuras ----
    profile["shown_ids"] = list(shown_ids | {c["id"] for c in top_main} | {c["id"] for c in top_new})
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

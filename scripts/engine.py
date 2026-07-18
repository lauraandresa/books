#!/usr/bin/env python3
"""
Motor de recomendación de libros.

Para cada archivo data/profiles/<usuario>.json:
  1. Construye un "perfil positivo" (materias/autores de los libros que le
     gustan) y un "perfil negativo" (de los que no).
  2. Busca candidatos nuevos: novedades/más vendidos (NYT) + libros
     recientes en las materias que más le gustan (Open Library / Google
     Books).
  3. Puntúa cada candidato por similitud de contenido (materias, autor,
     texto de la sinopsis) y por señales públicas de popularidad (nº de
     valoraciones, nota media) — NO es recomendación colaborativa
     ("la gente que leyó esto también leyó"), porque esa información no
     está disponible de forma gratuita y legal en ningún sitio (ver
     README).
  4. Escribe el top 10 en data/recommendations/<usuario>.json

Fuentes usadas, todas gratuitas y dentro de sus términos de uso:
  - Open Library API   (sin clave)
  - Google Books API   (sin clave, límite bajo sin clave; se puede añadir
                         GOOGLE_BOOKS_API_KEY como secreto del repo)
  - NYT Books API       (clave gratuita obligatoria: NYT_API_KEY)
"""
import json
import os
import re
import time
import datetime
import urllib.request
import urllib.parse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROFILES_DIR = "data/profiles"
RECS_DIR = "data/recommendations"
HEADERS = {"User-Agent": "personal-book-recs/1.0 (uso personal, no comercial)"}

NYT_API_KEY = os.environ.get("NYT_API_KEY", "").strip()
GOOGLE_BOOKS_API_KEY = os.environ.get("GOOGLE_BOOKS_API_KEY", "").strip()

MAX_RECS = 10
CANDIDATE_POOL_TARGET = 120


# ---------------------------------------------------------------- utils --
def http_get_json(url, retries=2):
    req = urllib.request.Request(url, headers=HEADERS)
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.load(r)
        except Exception as e:
            if attempt == retries:
                print(f"  aviso: fallo al pedir {url[:90]}... -> {e}")
                return None
            time.sleep(1.5)


def norm_subject(s):
    return re.sub(r"\s+", " ", s.strip().lower())


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


def gb_search(query, order="relevance", limit=20):
    params = {"q": query, "maxResults": min(limit, 40), "orderBy": order, "printType": "books"}
    if GOOGLE_BOOKS_API_KEY:
        params["key"] = GOOGLE_BOOKS_API_KEY
    url = "https://www.googleapis.com/books/v1/volumes?" + urllib.parse.urlencode(params)
    data = http_get_json(url)
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
            "year": (vi.get("publishedDate") or "")[:4],
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
                "year": datetime.date.today().year,
                "source": "nyt",
            })
    return out


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


def score_candidates(candidates, liked, disliked):
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

        score = (
            1.6 * subj_pos_overlap
            - 2.2 * subj_neg_overlap
            + 1.8 * tfidf_pos_sim[i]
            - 1.3 * tfidf_neg_sim[i]
            + 1.0 * author_match
            + 0.6 * popularity_boost
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

    pos_subjects = top_subjects(positive_all, n=8)
    print(f"  {username}: materias favoritas -> {pos_subjects}")

    candidates = build_candidate_pool(pos_subjects)
    candidates = [c for c in candidates if c["id"] not in shown_ids]
    # quita libros ya marcados
    known_ids = set(b["id"] for b in positive_all + disliked)
    candidates = [c for c in candidates if c["id"] not in known_ids]

    print(f"  {username}: {len(candidates)} candidatos nuevos encontrados")

    scored = score_candidates(candidates, positive_all, disliked)
    top = scored[:MAX_RECS]

    out_items = []
    for c in top:
        out_items.append({
            "id": c["id"],
            "title": c["title"],
            "author": c["author"],
            "cover_url": c.get("cover_url", ""),
            "subjects": c.get("subjects", [])[:8],
            "synopsis_short": synopsis_short(c.get("description", "") or c["title"]),
            "synopsis_full": c.get("description") or "No hay sinopsis disponible para este libro.",
            "source": c.get("source"),
            "score": c.get("score"),
        })

    os.makedirs(RECS_DIR, exist_ok=True)
    out_path = os.path.join(RECS_DIR, f"{username}.json")
    with open(out_path, "w") as f:
        json.dump({
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
            "items": out_items,
        }, f, ensure_ascii=False, indent=2)

    # actualiza shown_ids en el propio perfil para no repetir en semanas futuras
    profile["shown_ids"] = list(shown_ids | set(c["id"] for c in top))
    with open(path, "w") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)

    print(f"  {username}: {len(out_items)} recomendaciones guardadas")


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


if __name__ == "__main__":
    main()

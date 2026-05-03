# Quality Recommendations

Etat apres stabilisation du pipeline PDF/RAG et des erreurs LLM.

## Priorites Court Terme

1. Desindexer les fichiers runtime deja suivis par Git.
   `.gitignore` protege maintenant les nouveaux artefacts, mais les fichiers deja trackes comme `.chroma_db/chroma.sqlite3` resteront dans `git status` tant qu'ils ne sont pas retires de l'index Git avec `git rm --cached`.

2. Pinner les versions de dependances.
   `requirements.txt` liste les packages sans versions. Pour un projet de soutenance/demo, generer un lock ou ajouter des versions minimales reduira les regressions.

3. Ajouter une CI minimale.
   Lancer `python -m py_compile` et `pytest -q` a chaque push suffit pour attraper les regressions actuelles.

4. Isoler les stores RAG par environnement.
   Garder des chemins distincts pour dev/test/prod: Chroma, BM25, uploads, registry. Les tests unitaires doivent rester sur `tmp_path` ou mocks.

5. Traiter les warnings techniques.
   Les warnings restants viennent surtout de SQLAlchemy 2.0, Chroma et PyMuPDF/SWIG. Ils ne bloquent pas, mais ils doivent etre nettoyes avant une version stable.

## Robustesse Produit

1. Ajouter une validation stricte des uploads: taille maximale, extensions autorisees, MIME type, message utilisateur clair.
2. Afficher l'etat exact d'indexation: upload OK, extraction OK, OCR utilise ou non, chunks generes, vector index OK, keyword index OK, intelligence LLM OK/degradee.
3. Ajouter des tests end-to-end avec un vrai PDF scanne de reference et un PDF texte natif.
4. Prevoir un bouton de re-indexation forcee pour reconstruire proprement les index d'un fichier.
5. Ajouter une page admin simple pour nettoyer les fichiers non indexes, reconstruire BM25/Chroma, et exporter/importer la base documentaire.

## Configuration Recommandee

- `PDF_OCR_LANG=eng+fra`
- `PDF_OCR_DPI=200`
- `PDF_NATIVE_MIN_CHARS=80`
- `PDF_OCR_MAX_PAGES=0` pour OCR complet, ou une limite explicite en demo pour eviter les longs traitements.


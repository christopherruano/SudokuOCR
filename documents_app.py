#!/usr/bin/env python3
"""
Documents portal — standalone on :5059.

Same mechanics as census_review_app (reuses ALL of its routes/helpers: /documents, /render_saoa,
/render, /scan, matrix rendering, confirm→queue-OCR) but runs as its OWN process on a SEPARATE port
so it never disturbs the shared review app on :5058. Land on http://localhost:5059/documents

  python3 documents_app.py
"""
import census_review_app as A
from flask import redirect

# focused landing: root -> the documents list
A.app.add_url_rule("/portal", "portal_home", lambda: redirect("/documents"))

if __name__ == "__main__":
    print("Documents portal -> http://localhost:5059/documents  (separate from :5058)")
    A.app.run(port=5059, debug=False)

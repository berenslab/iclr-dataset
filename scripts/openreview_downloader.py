# =============================================================================
#   This script downloads all revision PDFs for a list of OpenReview papers - the final and all intermediate revisions uploaded
#   during the review process.
#
#   To run:
#   1. Install dependencies: pip install openreview-py requests
#   2. Fill in your OpenReview credentials below (username/password)
#   3. Add your forum IDs to the `forum_ids` list
#   4. Run: python openreview_downloader.py
#
# Output:
#   PDFs are saved to ./<forum_id>/<ref_id>.pdf
#   Each paper gets its own folder named by its forum ID.
# =============================================================================

import openreview
import time
import re
import os

USERNAME = 'email@example.com'
PASSWORD = 'password'

# Add forum IDs (the ID in the OpenReview URL)
# e.g. https://openreview.net/forum?id=BJgnXpVYwS -> 'BJgnXpVYwS'
FORUM_IDS = [
    'BJgnXpVYwS',
    'HJxdTxHYvB'
]

def download_with_retry(client, references, forum_id, access_token):
    """Download all unique revision PDFs for a given forum's references."""
    seen_pdfs = set()

    for ref in references:
        pdf_path = ref.content.get('pdf')
        if pdf_path and pdf_path not in seen_pdfs:
            seen_pdfs.add(pdf_path)

            # Revision PDFs are served at /references/pdf?id=<ref_note_id>
            # NOT at the raw /pdf/<hash>.pdf path stored in ref.content
            url = f"https://openreview.net/references/pdf?id={ref.id}"
            print(f"  Downloading: {url}")

            while True:
                try:
                    # Pass access token explicitly — the requests session scopes
                    # cookies to .openreview.net but the download host is openreview.net
                    response = client.session.get(
                        url,
                        cookies={'openreview.accessToken': access_token}
                    )
                    if response.status_code == 200:
                        filename = f"{forum_id}/{ref.id}_{ref.tcdate}.pdf"
                        with open(filename, "wb") as f:
                            f.write(response.content)
                        print(f"    Saved: {filename}")
                        time.sleep(1)  # stay under rate limit (3 req/window)
                        break
                    else:
                        print(f"    Failed: {response.status_code}")
                        break

                except Exception as e:
                    error = str(e)
                    if '429' in error or 'RateLimitError' in error:
                        match = re.search(r'try again in (\d+) seconds', error)
                        wait = int(match.group(1)) + 2 if match else 30
                        print(f"    Rate limited, waiting {wait}s...")
                        time.sleep(wait)
                    else:
                        print(f"    Error: {e}")
                        break

    return seen_pdfs


def main():
    # Log in
    client = openreview.Client(
        baseurl='https://api.openreview.net',
        username=USERNAME,
        password=PASSWORD
    )
    print(f"Logged in as: {client.user}\n")

    access_token = client.session.cookies.get('openreview.accessToken')

    for forum_id in FORUM_IDS:
        print(f"Processing: {forum_id}")
        os.makedirs(forum_id, exist_ok=True)

        try:
            # original=True is required to get full revision history with PDF fields
            references = client.get_references(referent=forum_id, original=True)
            seen = download_with_retry(client, references, forum_id, access_token)
            print(f"  Done — {len(seen)} unique PDFs downloaded\n")
        except Exception as e:
            print(f"  Skipping {forum_id} — Error: {e}\n")


if __name__ == '__main__':
    main()

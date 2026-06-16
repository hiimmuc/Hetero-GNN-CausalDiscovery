import os

import dotenv
import gradio as gr
import requests
import serpapi
from bs4 import BeautifulSoup

dotenv.load_dotenv()

client = serpapi.Client(api_key=os.getenv("SERP_API"))

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://scholar.google.com/",
}


def label_for(result):
    summary = result.get("publication_info", {}).get("summary", "")
    return result["title"] if not summary else f"{result['title']} — {summary}"


def search_papers(title):
    title = (title or "").strip()
    if not title:
        return gr.update(choices=[], value=None), [], "Enter a paper title to search."

    results = client.search({"engine": "google_scholar", "q": title, "hl": "en"})
    organic_results = results.get("organic_results", [])[:10]
    if not organic_results:
        return gr.update(choices=[], value=None), [], f"No results found for '{title}'."

    choices = [label_for(r) for r in organic_results]
    status = f"Found {len(organic_results)} result(s) for '{title}'. Select one below, then click \"Get BibTeX\"."
    return gr.update(choices=choices, value=choices[0]), organic_results, status


def get_bibtex(selected_label, organic_results):
    if not organic_results:
        return "Search for a paper first."
    if not selected_label:
        return "Select a paper from the list first."

    match = next((r for r in organic_results if label_for(r) == selected_label), None)
    if match is None:
        return "Could not match the selection to a search result. Try searching again."

    cite_results = client.search({"engine": "google_scholar_cite", "q": match["result_id"]})
    bib_url = next(
        (link["link"] for link in cite_results.get("links", []) if link.get("name") == "BibTeX"),
        None,
    )
    if not bib_url:
        return "No BibTeX citation is available for this paper."

    response = requests.get(bib_url, headers=HEADERS, timeout=15)
    if response.status_code != 200:
        return f"Failed to fetch BibTeX (HTTP {response.status_code})."

    soup = BeautifulSoup(response.text, "html.parser")
    return soup.get_text().strip()


with gr.Blocks(title="Google Scholar BibTeX Lookup") as demo:
    gr.Markdown(
        "# Google Scholar BibTeX Lookup\n"
        "Search for a paper by title, pick the matching result, then fetch its BibTeX citation."
    )

    results_state = gr.State([])

    with gr.Row():
        title_input = gr.Textbox(
            label="Paper title",
            placeholder="e.g. DirectLiNGAM: A direct method for learning a linear non-Gaussian structural equation model",
            scale=4,
        )
        search_btn = gr.Button("Search", variant="primary", scale=1)

    status = gr.Markdown()
    result_picker = gr.Radio(label="Matching papers", choices=[], interactive=True)
    bibtex_btn = gr.Button("Get BibTeX", variant="primary")
    bibtex_output = gr.Code(label="BibTeX", language=None)

    search_btn.click(
        search_papers,
        inputs=title_input,
        outputs=[result_picker, results_state, status],
    )
    title_input.submit(
        search_papers,
        inputs=title_input,
        outputs=[result_picker, results_state, status],
    )
    bibtex_btn.click(
        get_bibtex,
        inputs=[result_picker, results_state],
        outputs=bibtex_output,
    )

if __name__ == "__main__":
    demo.launch()

"""Create HTML page to view results."""

thumbnails_dir = "../resources/thumbnails/"
titles_path = "../resources/titles.txt"
generated_titles_path = "../resources/generated.txt"
results_path = "results.html"

# Bookend words to remove from generated titles
stopwords = ["startseq", "endseq"]


def load_titles(titles_path):
    """
    Load titles generated from get_titles() in training_data.py.

    Args
    ----
    titles_path: str
        Titles file.
    """
    titles = dict()
    with open(titles_path, "r") as f:
        for line in f:
            tokens = line.strip("\n").split("\t")
            video_id, title = tokens[0], tokens[1]
            title = " ".join([t for t in title.split() if t not in stopwords])
            titles[video_id] = title

    return titles


def opening():
    """HTML boilerplate."""
    entry = """<html>
<table style="width:100%">
<tr>
<th>thumbnail</th>
<th>title</th>
</tr>"""
    return entry


def new_entry(path, title, gen_title):
    """Make new entry in table."""
    entry = """<tr>
<td><img src="{}"></td>
<td>
<p><b>Original:</b> {}</p>
<p><b>Generated:</b> {}</p>
</td>
</tr>""".format(
        path, title, gen_title
    )

    return entry


if __name__ == "__main__":
    # Load titles. For now, if no generated titles path exists
    # just return None
    try:
        generated_titles = load_titles(generated_titles_path)
    except Exception:
        generated_titles = None

    with open(results_path, "w") as f:
        f.write(opening())
        with open(titles_path, "r") as f2:
            for line in f2:
                video_id = line.split("\t")[0]
                title = line.split("\t")[1]

                path = thumbnails_dir + video_id + ".jpg"

                try:
                    entry = new_entry(path, title, generated_titles[video_id])
                except Exception:
                    entry = new_entry(path, title, None)

                f.write(entry)

        f.write("</table>")
        f.write("</html>")

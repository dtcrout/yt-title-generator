"""Create HTML page to view results."""

import re

THUMBNAILS_DIR = '../resources/thumbnails/'
TITLES = '../resources/titles.txt'
GENERATED = './generated.txt'
PATH = './results.html'

stopwords = ['startseq', 'endseq']

def load_titles(path):
    titles = dict()
    with open(path, 'r') as f:
        for line in f:
            tokens = line.strip('\n').split('\t')
            video_id, title = tokens[0], tokens[1]
            title = ' '.join([t for t in title.split() if t not in stopwords])
            titles[video_id] = title
    return titles


def opening():
    entry = """<html>
<table style="width:100%">
<tr>
<th>thumbnail</th>
<th>title</th>
</tr>"""
    return entry


def new_entry(path, title, gen_title):
    entry = """<tr>
<td><img src="{}"></td>
<td>
<p><b>Original:</b> {}</p>
<p><b>Generated:</b> {}</p>
</td>
</tr>""".format(path, title, gen_title)

    return entry


if __name__ == "__main__":
    generated = load_titles(GENERATED) 

    with open(PATH, 'w') as f:
        f.write(opening())
        with open(TITLES, 'r') as f2:
            for line in f2:
                video_id = line.split('\t')[0]
                title = line.split('\t')[1]

                path = THUMBNAILS_DIR + video_id + '.jpg'

                try:
                    entry = new_entry(path, title, generated[video_id])
                except:
                    entry = new_entry(path, title, None)

                f.write(entry)

        f.write('</table>')
        f.write('</html>')

"""Create HTML page to view results."""

THUMBNAILS_DIR = '../resources/thumbnails/'
TITLES = '../resources/titles.txt'
PATH = './results.html'


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
    with open(PATH, 'w') as f:
        f.write(opening())
        with open(TITLES, 'r') as f2:
            for line in f2:
                video_id = line.split('\t')[0]
                title = line.split('\t')[1]

                path = THUMBNAILS_DIR + video_id + '.jpg'

                entry = new_entry(path, title, 'None')

                f.write(entry)

        f.write('</table>')
        f.write('</html>')

import requests
from lxml import html
import sqlite3
import progressbar


def tag_id_to_name(tag_id):
    base_url = 'https://m.tiktok.com/h5/share/tag/{}.html'
    response = requests.get(base_url.format(tag_id), headers={
        'User-Agent': 'Mozilla/5.0',
        'X-Requested-With': 'XMLHttpRequest',
        'Referer': 'https://permit.pcta.org/application/'
    })
    tree = html.fromstring(response.content)
    title = tree.xpath('//title')[0].text
    tag = title.split(' ')[0]
    return tag


def main():
    conn = sqlite3.connect('hashtags.db')
    conn.execute("CREATE TABLE IF NOT EXISTS hashtags (tag_id varchar NOT NULL, tag_name varchar NOT NULL);")
    conn.execute("CREATE INDEX IF NOT EXISTS hashtags_tag_id_idx ON hashtags (tag_id);")
    # the hashtag id is auto-incremental
    # format is https://m.tiktok.com/h5/share/tag/[hashtag id].html so I can crawl it
    min_tag_id = 1120  # empirically found, lower numbers are empty
    max_tag_id = 10000000  # actually there is much more tags, but I assume those tags are not relevant or empty

    widgets = [progressbar.Percentage(), ' ', progressbar.Counter(), ' ', progressbar.Bar(), ' ',
               progressbar.FileTransferSpeed()]
    pbar = progressbar.ProgressBar(widgets=widgets, max_value=max_tag_id - min_tag_id).start()

    for i in range(min_tag_id, max_tag_id):
        pbar.update(i - min_tag_id)
        if conn.execute(f'select exists(select 1 from hashtags where tag_id = \'{i}\')').fetchone()[0]:
            continue
        tag = tag_id_to_name(i)
        with conn:
            conn.execute('INSERT INTO hashtags (tag_id, tag_name) VALUES (?, ?)', (str(i), tag))

    pbar.finish()


if __name__ == '__main__':
    main()

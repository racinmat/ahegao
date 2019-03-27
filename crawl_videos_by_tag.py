import json
import sqlite3

from api import Api


def login_(client: Api, login, username, password):
    if login.get('code'):
        from capthca import capthca
        cc = capthca(login.get('code'))
        if cc.backdata:
            if len(cc.backdata) > 0:
                login = client.login(username, password, cc.backdata)
                del cc
                login_(client, login, username, password)
        else:
            print('empty form again')
            login = client.login(username, password, cc.backdata)
            del cc
            login_(client, login, username, password)


def create_client():
    with open('credentials.json', mode='r', encoding='utf-8') as fp:
        credentials = json.load(fp)
    client = Api()
    client.global_variable['device_id'] = credentials['device_id']
    client.global_variable['iid'] = credentials['iid']
    client.global_variable['openudid'] = credentials['openudid']
    client.global_variable['app_language'] = "cs"
    client.global_variable['language'] = "cs"
    client.global_variable['region'] = "CZ"
    client.global_variable['sys_region'] = "CZ"
    client.global_variable['carrier_region'] = "CZ"
    client.global_variable['carrier_region_v2'] = "230"

    username = credentials['username']
    password = credentials['password']
    login = client.login(username, password)
    login_(client, login, username, password)
    return client


def search_tags_by_name(client, query):
    results = client.search_hashtag(query)
    return [{'cid': i['challenge_info']['cid'], 'cha_name': i['challenge_info']['cha_name']} for i in
            results['challenge_list']]


def videos_by_hashtag(client, tag_id, count=100):
    results = client.list_hashtag(tag_id, count)
    return results_to_videos_info(results)


def videos_by_hashtag_paginated(client, tag_id, count=2000, per_page=100):
    videos = []
    prev_cursor = 0
    for page in range(round(count / per_page)):
        results = client.list_hashtag(tag_id, count=per_page, offset=prev_cursor)
        videos_page = results_to_videos_info(results)
        print(f'loaded {len(videos_page)} videos for hashtag {tag_id} in cursor {prev_cursor}')
        videos += videos_page
        if results['cursor'] == prev_cursor:
            print('end of cursor, aborting')
            break
        prev_cursor = results['cursor']
    print(f'loaded {len(videos)} videos for hashtag {tag_id} in total')
    return videos


def results_to_videos_info(results):
    return [{
        'download_url': i['video']['download_addr']['url_list'][0],
        'share_id': i['aweme_id'],
        'download_id': i['video']['download_addr']['uri'],
        'share_url': i['share_url'],
        'create_time': i['create_time'],
    } for i in results['aweme_list']]


def get_ahegao_hashtags():
    return {
        '2128154': 'ahegao',
        '30055085': 'ahegaoface',
        '1614591578030086': 'ahegaho',
        '1623459254549510': 'ahegahoface',
        '29504659': 'ahego',
        '1608178707908613': 'ahegoface',
        '73193535': 'aheago',
        '1607073744304130': 'aheagoface',
        '1618165477879878': 'aheagao',
        '1615419720169494': 'aheagaoface',
    }


def get_hitormiss_hashtags():
    return {
        '1887884': 'hitormiss',
        '21369576': 'hitormis',
    }


def main():
    # todo: try some max_cursor which filters out at least some videos (e.g. median of already colected data)
    client = create_client()
    # ahegao_searches = search_tags_by_name(client, 'aheg')
    # manually picked tag ids. Tag id can be obtained by sharing tag in url (resulting tag url contains its id).
    tag_ids_to_use = get_ahegao_hashtags()

    conn = sqlite3.connect('hashtags.db')
    conn.execute(
        "CREATE TABLE IF NOT EXISTS videos (tag_id varchar NOT NULL, share_url varchar NOT NULL, "
        "downloaded integer NOT NULL DEFAULT 0, share_id varchar NOT NULL, download_url varchar NOT NULL, "
        "download_id varchar NOT NULL, create_time integer NOT NULL);")

    for tag_id in tag_ids_to_use.keys():
        videos_list = videos_by_hashtag_paginated(client, tag_id, count=10000, per_page=100)
        for video in videos_list:
            share_id = video['share_id']
            if conn.execute(f'select exists(select 1 from videos where share_id = \'{share_id}\')').fetchone()[0]:
                continue
            with conn:
                conn.execute('INSERT INTO videos (tag_id, share_url, share_id, download_url, download_id, create_time) '
                             'VALUES (?, ?, ?, ?, ?, ?)',
                             (tag_id, video['share_url'], video['share_id'], video['download_url'],
                              video['download_id'], video['create_time']))

    print('done')


if __name__ == '__main__':
    main()

"""
F1 Race Winner Prediction - Data Collection Module
Downloads race results, qualifying and sprint results from the Jolpica
(Ergast-compatible) API. Every API page is cached under data/cache, so a
rerun reads from disk instead of calling the API again.
"""

import argparse
import glob
import json
import os
import time

import pandas as pd
import requests

import config

# Name of the list that holds the per-driver rows in each endpoint's reply
RESULT_KEYS = {
    'results': 'Results',
    'qualifying': 'QualifyingResults',
    'sprint': 'SprintResults',
}


def _cache_path(season, kind, offset):
    return os.path.join(config.CACHE_DIR, f'{season}_{kind}_{offset}.json')


def _fetch_page(season, kind, offset):
    """Return one page of an endpoint, from the cache when we have it."""
    path = _cache_path(season, kind, offset)
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            return json.load(f)

    url = f'{config.API_BASE}/{season}/{kind}.json'
    params = {'limit': config.PAGE_SIZE, 'offset': offset}
    for attempt in range(config.MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=30)
        except requests.ConnectionError as err:
            wait = 10 * (attempt + 1)
            print(f'    connection error ({err.__class__.__name__}), retrying in {wait}s')
            time.sleep(wait)
            continue
        if resp.status_code == 429:
            wait = int(resp.headers.get('Retry-After', 10 * (attempt + 1)))
            print(f'    rate limited, waiting {wait}s')
            time.sleep(wait)
            continue
        resp.raise_for_status()
        break
    else:
        raise RuntimeError(f'Gave up on {url} after {config.MAX_RETRIES} tries')
    time.sleep(config.REQUEST_PAUSE)

    data = resp.json()['MRData']
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f)
    return data


def fetch_season(season, kind):
    """Return the races of one season for one endpoint, all pages joined.

    The API pages over driver rows, so one race can be split across two pages.
    """
    key = RESULT_KEYS[kind]
    races = {}
    offset = 0
    while True:
        page = _fetch_page(season, kind, offset)
        for race in page['RaceTable']['Races']:
            rnd = int(race['round'])
            if rnd not in races:
                races[rnd] = {**race, key: []}
            races[rnd][key].extend(race[key])
        offset += config.PAGE_SIZE
        if offset >= int(page['total']):
            break
    return [races[r] for r in sorted(races)]


def _driver_fields(driver):
    code = driver.get('code') or driver['familyName'][:3].upper()
    return {
        'driver_id': driver['driverId'],
        'driver_code': code,
        'driver_name': f"{driver['givenName']} {driver['familyName']}",
    }


def _race_fields(season, race):
    return {
        'season': season,
        'round': int(race['round']),
        'race_name': race['raceName'],
        'circuit_id': race['Circuit']['circuitId'],
        'date': race['date'],
    }


def collect_season(season):
    """Return (races, sprints) DataFrames for one season."""
    print(f'  - {season}')
    results, quali, sprints = [], [], []

    for race in fetch_season(season, 'results'):
        meta = _race_fields(season, race)
        for r in race['Results']:
            results.append({
                **meta,
                **_driver_fields(r['Driver']),
                'constructor_id': r['Constructor']['constructorId'],
                'constructor_name': r['Constructor']['name'],
                'grid': int(r['grid']),
                'position': int(r['position']),
                'points': float(r['points']),
                'status': r['status'],
            })

    for race in fetch_season(season, 'qualifying'):
        meta = _race_fields(season, race)
        for q in race['QualifyingResults']:
            quali.append({
                **meta,
                **_driver_fields(q['Driver']),
                'constructor_id': q['Constructor']['constructorId'],
                'constructor_name': q['Constructor']['name'],
                'quali_position': int(q['position']),
            })

    if season >= config.SPRINT_FIRST_SEASON:
        for race in fetch_season(season, 'sprint'):
            for s in race['SprintResults']:
                sprints.append({
                    'season': season,
                    'round': int(race['round']),
                    'driver_id': s['Driver']['driverId'],
                    'constructor_id': s['Constructor']['constructorId'],
                    'points': float(s['points']),
                })

    results = pd.DataFrame(results)
    quali = pd.DataFrame(quali)
    keys = ['season', 'round', 'driver_id']

    if results.empty:
        races = quali.copy()
    else:
        races = results.merge(quali[keys + ['quali_position']], on=keys, how='left')
        # A race with qualifying but no results has not been run yet. Keep its
        # entry list so the predictor can score it.
        upcoming = quali[~quali['round'].isin(results['round'])]
        races = pd.concat([races, upcoming], ignore_index=True)

    sprint_cols = ['season', 'round', 'driver_id', 'constructor_id', 'points']
    return races, pd.DataFrame(sprints, columns=sprint_cols)


def collect_all(first=config.FIRST_SEASON, last=config.LAST_SEASON):
    print(f'Collecting F1 data {first}-{last} from {config.API_BASE}')
    races, sprints = [], []
    for season in range(first, last + 1):
        r, s = collect_season(season)
        races.append(r)
        sprints.append(s)
    races = pd.concat(races, ignore_index=True)
    sprints = pd.concat([s for s in sprints if not s.empty], ignore_index=True)

    os.makedirs(config.DATA_DIR, exist_ok=True)
    races.to_csv(config.RACES_FILE, index=False)
    sprints.to_csv(config.SPRINTS_FILE, index=False)

    n_races = races.groupby(['season', 'round']).ngroups
    n_run = races.dropna(subset=['position']).groupby(['season', 'round']).ngroups
    print(f'Saved {len(races)} driver rows from {n_races} races '
          f'({n_run} with results) to {config.RACES_FILE}')
    print(f'Saved {len(sprints)} sprint rows to {config.SPRINTS_FILE}')
    return races, sprints


def clear_cache(seasons):
    for season in seasons:
        for path in glob.glob(os.path.join(config.CACHE_DIR, f'{season}_*.json')):
            os.remove(path)
        print(f'Cleared cached pages for {season}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download F1 data from the Jolpica API')
    parser.add_argument('--refresh', type=int, nargs='*', default=[],
                        help='seasons to download again, e.g. --refresh 2026')
    args = parser.parse_args()

    clear_cache(args.refresh)
    collect_all()

#!/usr/bin/python3
#
# Apache 2.0 license

import argparse
from datetime import date
from dotted_dict import DottedDict
import itertools
import os
import random
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import sys
import yaml


def get_week():
    day_zero = date(2021, 1, 1)  # arbitrary start point
    delta = date.today() - day_zero
    return delta.days // 7


def draw_once(uids):
    uids = list(uids)
    random.shuffle(uids)
    while uids:
        if len(uids) > 3:
            curuids, uids = uids[:2], uids[2:]
        else:
            curuids, uids = uids, []
        yield curuids


# modifies history argument
def draw(uids, history):
    # Draw 100 times, compute the cost of each
    tries = []
    for n in range(100):
        groups = list(draw_once(uids))
        cost = 0
        for group in groups:
            for pair in itertools.combinations(sorted(group), 2):
                cost += history.get('|'.join(pair), 0)
        tries.append((cost, groups))

    # Pick the lowest-cost result, update history
    tries.sort(key=lambda try_: try_[0])  # stable sort
    result = tries[0][1]
    for group in result:
        for pair in itertools.combinations(sorted(group), 2):
            key = '|'.join(pair)
            history.setdefault(key, 0)
            history[key] += 1
    return result


def dry_run(message):
    print(f'---\n{message}\n---')


def try_slack_send(client, uids, message):
    try:
        cid = client.conversations_open(users=uids)['channel']['id']
        client.chat_postMessage(channel=cid, text=message)
    except SlackApiError as e:
        print(f"Error sending to {uids}: {e.response['error']}")
        return False
    return True


def do_pairing(config):
    try:
        with open(config.history_path) as fh:
            history = yaml.safe_load(fh)
    except FileNotFoundError:
        history = {}

    client = WebClient(token=config.token)
    ok = True
    week = get_week()
    uids = []
    for p in config.participants:
        if week % p.get('cadence', 1) == 0:
            uids.append(p['uid'])

    for curuids in draw(uids, history):
        print(f'Sending to {curuids}')

        message = (
            config.message_lonely,
            config.message,
            config.message_extra,
        )[len(curuids) - 1].strip().format(
            contact=config.contact,
            uids=curuids,
        )

        if config.dry_run:
            dry_run(message)
        else:
            ok = try_slack_send(client, curuids, message) and ok

    if not config.dry_run:
        with open(f'{config.history_path}.tmp', 'w') as fh:
            try:
                os.chmod(fh.fileno(), os.stat(config.history_path).st_mode)
            except FileNotFoundError:
                pass
            yaml.safe_dump(history, fh)
        os.rename(f'{config.history_path}.tmp', config.history_path)

    return ok


def main():
    parser = argparse.ArgumentParser(description='Send weekly 1:1 invitations.')
    parser.add_argument('-c', '--config', metavar='FILE', default='~/.11bot',
            help='config file')
    parser.add_argument('-H', '--history', metavar='FILE', default='~/.11bot-history',
            help='history file')
    parser.add_argument('-n', '--dry-run', action='store_true',
            help='print messages to stdout rather than sending')
    args = parser.parse_args()

    with open(os.path.expanduser(args.config)) as fh:
        config = DottedDict(yaml.safe_load(fh))
    token = os.environ.get('ELEVENBOT_TOKEN')
    if token:
        config.token = token
    config.history_path = os.path.expanduser(args.history)
    config.dry_run = args.dry_run

    ok = do_pairing(config)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())

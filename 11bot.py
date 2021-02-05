# Apache 2.0 license

import argparse
from datetime import date
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


def main():
    parser = argparse.ArgumentParser(description='Send weekly 1:1 invitations.')
    parser.add_argument('-c', '--config', metavar='FILE', default='~/.11bot',
            help='config file')
    parser.add_argument('-n', '--dry-run', action='store_true',
            help='print messages to stdout rather than sending')
    parser.add_argument('-s', '--send', metavar='MESSAGE',
            help='DM the specified message to each participant')
    args = parser.parse_args()

    with open(os.path.expanduser(args.config)) as fh:
        config = yaml.safe_load(fh)

    client = WebClient(token=config['token'])
    ok = True
    if args.send:
        # DM specified message to every user
        for p in config['participants']:
            uid = p['uid']
            print(f'Sending to {uid}')
            message = args.send
            if args.dry_run:
                dry_run(message)
            else:
                ok = try_slack_send(client, uid, message) and ok
    else:
        week = get_week()
        uids = []
        for p in config['participants']:
            if week % p.get('cadence', 1) == 0:
                uids.append(p['uid'])
        random.shuffle(uids)

        while uids:
            if len(uids) > 3:
                curuids, uids = uids[:2], uids[2:]
            else:
                curuids, uids = uids, []
            print(f'Sending to {curuids}')

            message = (
                config['message-lonely'],
                config['message'],
                config['message-extra'],
            )[len(curuids) - 1].strip().format(
                contact=config['contact'],
                uids=curuids,
            )

            if args.dry_run:
                dry_run(message)
            else:
                ok = try_slack_send(client, curuids, message) and ok

    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/python3
#
# Apache 2.0 license

import argparse
from collections import OrderedDict, namedtuple
from croniter import croniter
from datetime import date
from dotted_dict import DottedDict
from functools import reduce, wraps
from heapq import heappop, heappush
import itertools
import os
import random
import requests
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from slack_sdk.socket_mode import SocketModeClient
from slack_sdk.socket_mode.response import SocketModeResponse
import sqlite3
import time
import threading
import traceback
import yaml

ISSUE_LINK = 'https://github.com/coreos/11bot/issues'
HELP = f'''
11bot periodically arranges random 1:1s between participating channel members.
%commands%
Report bugs <{ISSUE_LINK}|here>.  Bot administrators are %admins%.
'''


def escape(message):
    '''Escape a string for inclusion in a Slack message.'''
    # https://api.slack.com/reference/surfaces/formatting#escaping
    map = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
    }
    return reduce(lambda s, p: s.replace(p[0], p[1]), map.items(), message)


def format_uids(uids):
    uidlist = [f'<@{uid}>' for uid in uids]
    if len(uids) == 2:
        return ' and '.join(uidlist)
    if len(uids) > 2:
        uidlist[-1] = 'and ' + uidlist[-1]
    return ', '.join(uidlist)


ChannelSettings = namedtuple('ChannelSettings',
        ['seed', 'groupsize', 'interval_weeks'],
        defaults=[None, 2, 1])


class Database:
    def __init__(self, config):
        # we pass Database objects between threads
        self._db = sqlite3.connect(config.database, check_same_thread=False)
        with self:
            self._db.execute('pragma foreign_keys = on')
            ver = self._db.execute('pragma user_version').fetchone()[0]
            if ver < 1:
                self._db.execute('create table events '
                        '(added integer not null, '
                        'channel text not null, '
                        'ident text not null)')
                self._db.execute('create unique index events_unique '
                        'on events (channel, ident)')

                self._db.execute('create table channels '
                        '(channel text unique not null, '
                        'seed integer not null, '
                        'groupsize integer not null, '
                        'interval_weeks integer not null)')

                self._db.execute('create table participants '
                        '(channel text not null references channels(channel), '
                        'user text not null)')
                self._db.execute('create unique index participants_unique '
                        'on participants (channel, user)')

                self._db.execute('create table costs '
                        '(user1 text not null, '
                        'user2 text not null, '
                        'cost real not null)')
                self._db.execute('create unique index costs_unique '
                        'on costs (user1, user2)')

                self._db.execute('pragma user_version = 1')

    def __enter__(self):
        '''Start a database transaction.'''
        self._db.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, tb):
        '''Commit or abort a database transaction.'''
        return self._db.__exit__(exc_type, exc_value, tb)

    def add_event(self, channel, ident):
        '''Return False if the event is already present.'''
        try:
            self._db.execute('insert into events (added, channel, ident) '
                    'values (?, ?, ?)', (int(time.time()), channel, ident))
            return True
        except sqlite3.IntegrityError:
            return False

    def prune_events(self, max_age=3600):
        self._db.execute('delete from events where added < ?',
                (int(time.time() - max_age),))

    def get_channels(self):
        # ignore channels without participants
        res = self._db.execute('select distinct channel '
                'from participants').fetchall()
        if res is None:
            return []
        return [r[0] for r in res]

    def _ensure_channel(self, channel):
        '''Create default channel settings if they don't exist.'''
        defaults = ChannelSettings()
        # randomize seed
        self._db.execute('insert or ignore into channels '
                '(channel, seed, groupsize, interval_weeks) '
                'values (?, ?, ?, ?)',
                (channel, random.getrandbits(32), defaults.groupsize,
                defaults.interval_weeks))

    def get_channel_settings(self, channel):
        self._ensure_channel(channel)
        res = self._db.execute('select seed, groupsize, interval_weeks '
                'from channels where channel == ?', (channel,)).fetchone()
        return ChannelSettings(seed=res[0], groupsize=res[1],
                interval_weeks=res[2])

    def set_channel_settings(self, channel, settings):
        self._ensure_channel(channel)
        # don't reset seed
        self._db.execute("update channels set groupsize = ?, "
                "interval_weeks = ? where channel == ?",
                (settings.groupsize, settings.interval_weeks, channel))

    def is_channel_participant(self, channel, user):
        res = self._db.execute('select 1 from participants where '
                'channel == ? and user == ?', (channel, user)).fetchone()
        return res is not None

    def get_channel_participants(self, channel):
        res = self._db.execute('select user from participants where '
                'channel == ?', (channel,)).fetchall()
        return [r[0] for r in res]

    def count_channel_participants(self, channel):
        res = self._db.execute('select count(user) from participants where '
                'channel == ?', (channel,)).fetchone()
        return res[0]

    def add_channel_participant(self, channel, user):
        self._db.execute('insert or ignore into participants (channel, user) '
                'values (?, ?)', (channel, user))

    def remove_channel_participant(self, channel, user):
        self._db.execute('delete from participants where channel == ? and '
                'user == ?', (channel, user))

    def delete_channel(self, channel):
        self._db.execute('delete from participants where channel == ?',
                (channel,))
        self._db.execute('delete from channels where channel == ?',
                (channel,))

    def get_pair_cost(self, user1, user2):
        users = sorted((user1, user2))
        res = self._db.execute('select cost from costs where user1 == ? and '
                'user2 == ?', (users[0], users[1])).fetchone()
        if res is None:
            return 0.0
        return res[0]

    def add_pair_cost(self, user1, user2, cost):
        users = sorted((user1, user2))
        self._db.execute('insert or ignore into costs (user1, user2, cost) '
                'values (?, ?, 0.0)', (users[0], users[1]))
        res = self._db.execute('update costs set cost = cost + ? '
                'where user1 == ? and user2 == ?', (cost, users[0], users[1]))

    def decay_pair_costs(self, factor):
        assert factor != 0
        self._db.execute('update costs set cost = cost * ?', (factor,))


class Channel:
    # Database transactions must be supplied by the caller.

    def __init__(self, config, client, db, id):
        self._config = config
        self._client = client
        self._db = db
        self.id = id
        self.settings = db.get_channel_settings(id)

    @classmethod
    def get_channels(cls, config, client, db):
        '''Get all channels with participants.'''
        return [cls(config, client, db, id) for id in db.get_channels()]

    @staticmethod
    def have_access(client, id):
        '''Return True if we have Slack API access to the specified channel.'''
        try:
            info = DottedDict(client.conversations_info(channel=id)['channel'])
            return ((info.is_channel or info.is_group) and not info.is_archived)
        except SlackApiError as e:
            # might get missing_scope for IM/MPIM conversations
            if e.response['error'] in ('channel_not_found', 'missing_scope'):
                return False
            else:
                raise

    @property
    def interval_str(self):
        '''Return a string describing the interval at which we perform
        groupings.'''
        if self.settings.interval_weeks > 1:
            return f'{self.settings.interval_weeks} weeks'
        else:
            return 'week'

    @property
    def weeks_until_grouping(self):
        '''Return 0 if we're grouping this week, 1 if next week, etc.'''
        day_zero = date(2021, 1, 3)  # arbitrary start point, on a Sunday
        delta = date.today() - day_zero
        current_week = delta.days // 7
        # mix in a per-channel seed so all channels don't reach week 0 at
        # the same time
        v = (current_week + self.settings.seed) % self.settings.interval_weeks
        if v == 0:
            return 0
        return self.settings.interval_weeks - v

    @property
    def next_grouping_str(self):
        '''Return a string describing the next time we're performing a
        grouping.'''
        weeks = self.weeks_until_grouping
        if weeks == 0:
            return 'this week'
        elif weeks == 1:
            return 'next week'
        else:
            return f'in {weeks} weeks'

    def group(self):
        '''Perform a grouping of people in the channel, send messages, and
        record costs.'''
        grouping = Grouping.minimal_cost(self._db, self.settings.groupsize,
                self.participants())
        for uids in grouping.groups:
            message = f"Hi {format_uids(uids)}!  "
            if len(uids) == 1:
                message += f"No one else in <#{self.id}> is currently participating in 1:1s.  :disappointed:  I'll try again next time."
            else:
                message += f"You've been grouped for a 1:1 this week.  Use this group chat to discuss timing and details."
            message += f"\n\nType `/11bot leave` in <#{self.id}> to unsubscribe."
            cid = self._client.conversations_open(users=uids)['channel']['id']
            self._client.chat_postMessage(channel=cid, text=message)
        grouping.save_cost(self._db)

    def _notify_leave(self, user, reason):
        '''Notify user that they've stopped participating for reason.'''
        try:
            self._client.chat_postMessage(channel=user,
                    text=f"You're no longer participating in 1:1s in <#{self.id}> because {reason}.")
        except SlackApiError:
            # best-effort: user might have left the workspace, etc.
            pass

    def sync(self):
        '''Sync database from Slack, notifying users we drop.  Raise KeyError
        if we no longer have access to the channel.'''
        # Drop channel if no access
        if not self.have_access(self._client, self.id):
            self.delete('I lost access to the channel')
            raise KeyError
        # Drop participants if not in channel
        participants = set(self._db.get_channel_participants(self.id))
        for resp in self._client.conversations_members(channel=self.id, limit=200):
            for member in resp['members']:
                participants.discard(member)
        for participant in participants:
            self._db.remove_channel_participant(self.id, participant)
            self._notify_leave(participant, 'you are no longer in the channel')

    def delete(self, reason):
        '''Delete our records of the channel for reason and notify users.'''
        participants = self._db.get_channel_participants(self.id)
        self._db.delete_channel(self.id)
        for participant in participants:
            self._notify_leave(participant, reason)

    def save_settings(self, settings):
        self._db.set_channel_settings(self.id, settings)
        self.settings = settings

    def participants(self):
        return self._db.get_channel_participants(self.id)

    def participant_count(self):
        return self._db.count_channel_participants(self.id)

    def is_participant(self, user):
        return self._db.is_channel_participant(self.id, user)

    def add_participant(self, user):
        self._db.add_channel_participant(self.id, user)

    def remove_participant(self, user):
        self._db.remove_channel_participant(self.id, user)


class Grouping:
    '''One disjoint set of 1:1s for one channel.'''

    def __init__(self, groupsize, uids):
        uids = list(uids)
        random.shuffle(uids)
        self.groups = []
        while uids:
            curuids, uids = uids[:groupsize], uids[groupsize:]
            self.groups.append(curuids)
        # If the last group is more than half the required groupsize, leave
        # it alone.  Otherwise, distribute its members to the other groups.
        if len(self.groups) > 1 and len(self.groups[-1]) <= groupsize // 2:
            for i, uid in enumerate(self.groups.pop()):
                self.groups[i % len(self.groups)].append(uid)

    def get_cost(self, db):
        '''Query the database and return the total cost of this grouping.'''
        cost = 0
        for group in self.groups:
            for (user1, user2) in itertools.combinations(group, 2):
                cost += db.get_pair_cost(user1, user2)
        return cost

    def save_cost(self, db):
        '''Save the additional cost of this grouping back to the database.'''
        for group in self.groups:
            if len(group) == 1:
                # only one participant in the channel
                continue
            # Total cost of 1 for each participant, divided among their
            # pairings with the rest of the group
            group_cost = 1 / (len(group) - 1)
            for (user1, user2) in itertools.combinations(group, 2):
                db.add_pair_cost(user1, user2, group_cost)

    @classmethod
    def minimal_cost(cls, db, groupsize, uids, tries=100):
        groupings = []
        for _ in range(tries):
            grouping = cls(groupsize, uids)
            groupings.append((grouping.get_cost(db), grouping))
        # pick the lowest-cost result
        groupings.sort(key=lambda g: g[0])  # stable sort
        return groupings[0][1]

    @classmethod
    def selftest(cls):
        def t(groupsize, total, lengths):
            grouping = cls(groupsize, list(range(1, total + 1)))
            actual_lengths = [len(g) for g in grouping.groups]
            if lengths != actual_lengths:
                raise Exception(f'Self-test failed: expected lengths {lengths}, found {actual_lengths}')
        t(2, 0, [])
        t(2, 1, [1])
        t(2, 2, [2])
        t(2, 3, [3])
        t(2, 4, [2, 2])
        t(2, 5, [3, 2])
        t(2, 6, [2, 2, 2])
        t(2, 7, [3, 2, 2])
        t(3, 0, [])
        t(3, 1, [1])
        t(3, 2, [2])
        t(3, 3, [3])
        t(3, 4, [4])
        t(3, 5, [3, 2])
        t(3, 6, [3, 3])
        t(3, 7, [4, 3])
        t(3, 8, [3, 3, 2])
        t(4, 0, [])
        t(4, 1, [1])
        t(4, 2, [2])
        t(4, 3, [3])
        t(4, 4, [4])
        t(4, 5, [5])
        t(4, 6, [6])
        t(4, 7, [4, 3])
        t(4, 8, [4, 4])
        t(4, 9, [5, 4])
        t(4, 10, [5, 5])
        t(4, 11, [4, 4, 3])


class HandledError(Exception):
    '''An exception which should just be swallowed.'''
    pass


class Fail(Exception):
    '''An exception with a message that should be displayed to the user.'''
    pass


def report_errors(f):
    '''Decorator that sends exceptions to administrators via Slack DM
    and then swallows them.  The first argument of the function must be
    the config.'''
    import socket, urllib.error
    @wraps(f)
    def wrapper(config, *args, **kwargs):
        def send(message):
            try:
                client = WebClient(token=config.token)
                channel = client.conversations_open(users=config.admins)['channel']['id']
                client.chat_postMessage(channel=channel, text=message)
            except Exception:
                traceback.print_exc()
        try:
            return f(config, *args, **kwargs)
        except Fail as e:
            # Nothing else caught this; just report the error string.
            send(str(e))
        except HandledError:
            pass
        except (requests.ConnectionError, requests.HTTPError, requests.ReadTimeout) as e:
            # Assume transient network problem; don't send message.
            print(e)
        except (socket.timeout, urllib.error.URLError) as e:
            # Exception type leaked from the slack_sdk API.  Assume transient
            # network problem; don't send message.
            print(e)
        except Exception:
            send(f'Caught exception:\n```\n{traceback.format_exc()}```')
    return wrapper


class Registry(type):
    '''Metaclass that creates a dict of functions registered with the
    register decorator.'''

    def __new__(cls, name, bases, attrs):
        cls = super().__new__(cls, name, bases, attrs)
        registry = []
        for f in attrs.values():
            command = getattr(f, 'command', None)
            if command is not None:
                registry.append((command, f))
        registry.sort(key=lambda t: t[1].doc_order)
        cls._registry = OrderedDict(registry)
        return cls


def register(command, args=(), group=None, doc=None, affects_channel=True,
        admin=False):
    '''Decorator that registers the subcommand handled by a function.'''
    def decorator(f):
        f.command = command
        f.args = args
        f.group = group
        f.doc = doc
        f.doc_order = time.time()  # hack alert!
        f.affects_channel = affects_channel
        f.admin = admin
        return f
    return decorator


class CommandHandler(metaclass=Registry):
    '''Wrapper class to handle a single slash command in an OS thread, with
    exception handling.'''

    GROUP_DEFAULT = 'User commands'
    GROUP_CHANNEL = 'Channel-wide settings'
    GROUP_ADMIN = 'Admin commands'

    def __init__(self, config, payload):
        self._config = config
        self._payload = payload
        self._client = WebClient(token=config.token)
        self._db = Database(config)
        self._called = False

    def __call__(self):
        assert not self._called
        self._called = True

        @report_errors
        def wrapper(_config):
            try:
                with self._db:
                    args = self._payload.text.strip().split()
                    try:
                        f = self._registry[args.pop(0)]
                    except (KeyError, IndexError):
                        raise Fail(f"I didn't understand that.  Try `{self._payload.command} help`.")
                    if len(args) != len(f.args):
                        if f.args:
                            argdesc = ' '.join(f'<{a}>' for a in f.args)
                            raise Fail(f'Bad arguments; expect `{argdesc}`.')
                        else:
                            raise Fail('This command takes no arguments.')
                    if f.admin:
                        if self._payload.user_id not in self._config.admins:
                            raise Fail(f'This command is limited to 11bot administrators.')
                    if f.affects_channel:
                        if not Channel.have_access(self._client,
                                self._payload.channel_id):
                            raise Fail(f"I don't have access to this channel.  Try typing `/invite <@{self._config.bot_id}>`.")
                    f(self, *args)
            except Fail as e:
                self._result(str(e))
                # convert to HandledError to indicate that we've displayed this
                # message
                raise HandledError()
            except Exception:
                self._result('Internal error.  Admins have been notified.')
                raise
        # report_errors() requires the config to be the first argument
        threading.Thread(target=wrapper, args=(self._config,)).start()

    def _get_channel(self, sync=False):
        '''Get a Channel for the current payload, optionally syncing it with
        Slack state.'''
        channel = Channel(self._config, self._client, self._db,
                self._payload.channel_id)
        if sync:
            try:
                channel.sync()
            except KeyError:
                raise Fail(f"No access to <#{id}>")
        return channel

    def _result(self, message, in_channel=False):
        '''Send the result message for a command.'''
        if in_channel:
            # many channel members may not have heard of us
            message += f'\n\nFor more information on 11bot, type `{self._payload.command} help`.'
        requests.post(self._payload.response_url, json={
            'response_type': 'in_channel' if in_channel else 'ephemeral',
            'text': message,
        })

    def _report_info(self, channel, user_changed=False, report_channel=True):
        participating = channel.is_participant(self._payload.user_id)
        message = (f"You're "
            f"{'now' if user_changed else 'currently'} "
            f"{'not ' if not participating else ''}"
            f"participating in <#{self._payload.channel_id}> 1:1s.")
        if report_channel:
            message += (f"  {channel.participant_count()} "
                f"{'person is' if channel.participant_count() == 1 else 'people are'} "
                f"participating every {channel.interval_str} "
                f"in groups of {channel.settings.groupsize}.")
            if channel.settings.interval_weeks > 1:
                message += f'  The next 1:1 is {channel.next_grouping_str}.'
        self._result(message)

    @register('info', doc='get info about 1:1s in this channel')
    def _info(self):
        channel = self._get_channel(sync=True)
        self._report_info(channel)

    @register('join', doc='join 1:1s for this channel')
    def _join(self):
        channel = self._get_channel()
        if channel.is_participant(self._payload.user_id):
            self._report_info(channel)
        else:
            channel.add_participant(self._payload.user_id)
            self._report_info(channel, user_changed=True)

    @register('leave', doc='leave 1:1s for this channel')
    def _leave(self):
        channel = self._get_channel()
        if not channel.is_participant(self._payload.user_id):
            self._report_info(channel, report_channel=False)
        else:
            channel.remove_participant(self._payload.user_id)
            self._report_info(channel, user_changed=True, report_channel=False)

    @register('participants',
            doc='list 1:1 participants in the current channel')
    def _participants(self):
        channel = self._get_channel(sync=True)
        participants = [f'<@{user}>' for user in channel.participants()]
        if not participants:
            participants.append('_none_')
        self._result(f'*Current 1:1 participants in <#{channel.id}>:*\n' +
                '\n'.join(participants))

    @register('ping', affects_channel=False,
            doc='check whether the bot is running')
    def _ping(self):
        self._result(':wave:')

    @register('help', affects_channel=False, doc='print this message')
    def _help(self):
        groups = OrderedDict()
        for f in self._registry.values():
            if f.doc is None:
                continue
            if f.admin:
                if self._payload.user_id not in self._config.admins:
                    continue
                group = self.GROUP_ADMIN
            elif f.group is None:
                group = self.GROUP_DEFAULT
            else:
                group = f.group
            groups.setdefault(group, []).append(f)
        lines = []
        for group, funcs in groups.items():
            lines.append(f'*{group}:*')
            for f in funcs:
                lines.append('`{}{}{}` - {}'.format(
                    f.command,
                    ' ' if f.args else '',
                    ' '.join((f'<{a}>' for a in f.args)),
                    f.doc,
                ))
        self._result(HELP
            .replace('%commands%', '\n'.join(lines))
            .replace('%admins%', format_uids(self._config.admins)))

    @register('groupsize', args=('number',), group=GROUP_CHANNEL,
            doc='set number of participants in a 1:1 group')
    def _groupsize(self, count):
        try:
            count = int(count, 10)
            if count < 2:
                raise ValueError
        except ValueError:
            raise Fail('Invalid number of participants.')
        channel = self._get_channel()
        if not channel.is_participant(self._payload.user_id):
            raise Fail('Only participants can change channel settings.')
        channel.save_settings(channel.settings._replace(groupsize=count))
        self._result(f'1:1s in this channel will now include {count} people, '
                f'as requested by <@{self._payload.user_id}>.',
                in_channel=True)

    @register('cadence', args=('weeks',), group=GROUP_CHANNEL,
            doc='set interval between 1:1s, in weeks')
    def _cadence(self, weeks):
        try:
            weeks = int(weeks, 10)
            if weeks < 1:
                raise ValueError
        except ValueError:
            raise Fail('Invalid cadence.')
        channel = self._get_channel()
        if not channel.is_participant(self._payload.user_id):
            raise Fail('Only participants can change channel settings.')
        channel.save_settings(channel.settings._replace(interval_weeks=weeks))
        message = (f'1:1s in this channel will now be held every '
                f'{channel.interval_str}, as requested by '
                f'<@{self._payload.user_id}>.')
        if weeks > 1:
            message += f'  The next 1:1 will be {channel.next_grouping_str}.'
        self._result(message, in_channel=True)

    @register('channels', affects_channel=False, admin=True,
            doc='list channels that have participants')
    def _channels(self):
        channels = [f'<#{channel.id}> - {channel.settings.groupsize}/{channel.participant_count()} every {channel.interval_str}'
                for channel in
                Channel.get_channels(self._config, self._client, self._db)]
        if not channels:
            channels.append('_none_')
        self._result('*Current channels:*\n' + '\n'.join(channels))

    @register('sync', affects_channel=False, admin=True,
            doc='sync channel membership in all channels')
    def _sync(self):
        channels = Channel.get_channels(self._config, self._client, self._db)
        for channel in channels:
            try:
                channel.sync()
            except KeyError:
                pass
        self._result(f'Synced {len(channels)} channels.')

    @register('run', admin=True,
            doc='immediately perform a grouping in this channel')
    def _run(self):
        channel = self._get_channel(sync=True)
        channel.group()
        self._result(f'Paired {channel.participant_count()} participants.')

    @register('abandon', admin=True,
            doc='remove 11bot from this channel and delete settings')
    def _abandon(self):
        self._result(f'Leaving this channel, as requested by <@{self._payload.user_id}>. :wave:',
                in_channel=True)
        try:
            self._client.conversations_leave(channel=self._payload.channel_id)
        except SlackApiError as e:
            if e.response['error'] != 'not_in_channel':
                raise
        channel = self._get_channel()
        channel.delete('I have been removed from the channel')

    @register('throw', affects_channel=False, admin=True)
    def _throw(self):
        raise Exception(f'Throwing exception as requested by <@{self._payload.user_id}>')


@report_errors
def process_event(config, socket_client, req):
    '''Handler for a Slack event.'''
    payload = DottedDict(req.payload)

    if req.type == 'slash_commands':
        # Don't even acknowledge events in forbidden channels, to avoid
        # interfering with separate bot instances in other channels.
        if payload.channel_id in config.get('channels_deny', []):
            return
        if config.get('channels_allow') and payload.channel_id not in config.channels_allow:
            return

        # Acknowledge the event, as required by Slack.
        resp = SocketModeResponse(envelope_id=req.envelope_id)
        socket_client.send_socket_mode_response(resp)

        # Idempotency
        with Database(config) as db:
            if not db.add_event(payload.channel_id, payload.trigger_id):
                # When we ignore some events, Slack can send us duplicate
                # retries.  Detect and ignore those after acknowledging.
                return

        # Process it
        CommandHandler(config, payload)()
    else:
        raise Fail(f'Unexpected event type "{req.type}"')


class Scheduler:
    def __init__(self, config, client, db):
        self._config = config
        self._client = client
        self._db = db
        self._jobs = []
        self._add_timer(self._prune, 'prune_interval', 3600)
        self._add_cron(self._group, 'weekly_grouping_schedule', "0 4 * * 1")

    def _add_cron(self, fn, config_key, default=None):
        schedule = self._config.get(config_key, default)
        if schedule is not None:
            it = croniter(schedule)
            # add the list length as a tiebreaker when sorting, so we don't
            # try to compare two fns
            heappush(self._jobs, (next(it), len(self._jobs), fn, it))

    def _add_timer(self, fn, config_key, default=None):
        interval = self._config.get(config_key, default)
        if interval is not None:
            it = itertools.count(int(time.time()), interval)
            # add the list length as a tiebreaker when sorting, so we don't
            # try to compare two fns
            heappush(self._jobs, (next(it), len(self._jobs), fn, it))

    def run(self):
        while True:
            # get the next job that's due
            nex, idx, fn, it = heappop(self._jobs)
            # wait for the scheduled time, allowing for spurious wakeups
            while True:
                now = time.time()
                if now >= nex:
                    break
                time.sleep(nex - now)
            # run the job, passing the config to make report_errors() happy
            @report_errors
            @wraps(fn)
            def wrapper(_config):
                fn()
            wrapper(self._config)
            # schedule the next run, skipping any times that are already
            # in the past
            now = time.time()
            while True:
                nex = next(it)
                if nex > now:
                    break
            heappush(self._jobs, (nex, idx, fn, it))

    def _group(self):
        '''Perform groupings for the week.'''
        with self._db:
            channels = Channel.get_channels(self._config, self._client,
                    self._db)
            self._db.decay_pair_costs(self._config.get('cost-falloff', 0.96))

        @report_errors
        def handle_one(_config, channel):
            if channel.weeks_until_grouping == 0:
                with self._db:
                    try:
                        channel.sync()
                    except KeyError:
                        return
                    channel.group()
        for channel in channels:
            handle_one(self._config, channel)

    def _prune(self):
        with self._db:
            self._db.prune_events()


def main():
    parser = argparse.ArgumentParser(
            description='Slack bot to send periodic 1:1 invitations.')
    parser.add_argument('-c', '--config', metavar='FILE',
            default='~/.11bot', help='config file')
    parser.add_argument('-d', '--database', metavar='FILE',
            default='~/.11bot-db', help='database file')
    args = parser.parse_args()

    # Self-test
    Grouping.selftest()

    # Read config
    with open(os.path.expanduser(args.config)) as fh:
        config = DottedDict(yaml.safe_load(fh))
        config.database = os.path.expanduser(args.database)
    env_map = (
        ('ELEVENBOT_APP_TOKEN', 'app-token'),
        ('ELEVENBOT_TOKEN', 'token'),
    )
    for env, config_key in env_map:
        v = os.environ.get(env)
        if v:
            setattr(config, config_key, v)

    # Connect to services
    client = WebClient(token=config.token)
    # store our user ID
    config.bot_id = client.auth_test()['user_id']
    db = Database(config)

    # Start socket-mode listener in the background
    socket_client = SocketModeClient(app_token=config.app_token,
            web_client=WebClient(token=config.token))
    socket_client.socket_mode_request_listeners.append(
            lambda socket_client, req: process_event(config, socket_client, req))
    socket_client.connect()

    # Run scheduler
    Scheduler(config, client, db).run()


if __name__ == '__main__':
    main()

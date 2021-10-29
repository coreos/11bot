# 1:1 bot

## Installing

A `setup.cfg` would be nice, but we don't have one right now.

```sh
cd ~
git clone https://github.com/coreos/11bot
cd 11bot
virtualenv env
env/bin/pip install -r requirements.txt
env/bin/python 11bot.py
```

Alternatively, a [container image](https://quay.io/repository/coreos/11bot) is available.

You'll also need to set up a Slack app in your workspace and get an API token for it.

## Slack configuration

- Slash command: `/11bot`
- Scopes:
  - `channels:manage` - leave public channels with `/11bot abandon`
  - `channels:read` - check that public channel participants are members
  - `commands` - add slash command
  - `groups:read` - check that private channel participants are members
  - `groups:write` - leave private channels with `/11bot abandon`
  - `im:write` - send 1:1 notices with only one participant
  - `mpim:write` - send 1:1 notices and admin messages

## Config format

See [config.example](config.example).  Put this in `~/.11bot` by default.

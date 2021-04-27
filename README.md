# 1:1 bot

## Installing

A `setup.cfg` would be nice, but we don't have one right now.

```sh
cd ~
git clone https://github.com/bgilbert/11bot
cd 11bot
virtualenv env
env/bin/pip install -r requirements.txt
crontab -l | cat - crontab.example | crontab -
```

Alternatively, a [container image](https://quay.io/repository/bgilbert/11bot)
is available.

You'll also need to set up a Slack app in your workspace and get an API
token for it.

## Config format

Put this in `~/.11bot` by default.

```yaml
token: <API token>
contact: <uid>
message: <message>
message-extra: <message for 3 participants>
message-lonely: <message if there's only one participant>
participants:
  - uid: <uid>
    cadence: 2  # optional; weeks
```

Message substitution variables:
- `{contact}`: the configured contact UID
- `{uids[0]}`, `{uids[1]}`, `{uids[2]}`: UIDs we're sending to

Use these in messages as e.g. `<@{contact}>`.

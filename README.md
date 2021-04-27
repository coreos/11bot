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

See [config.example](config.example).  Put this in `~/.11bot` by default.

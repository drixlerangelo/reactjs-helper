#!/bin/sh

cd /app
supervisord -c /app/supervisord.conf

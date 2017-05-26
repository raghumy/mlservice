#!/usr/bin/env python3
import connexion
import datetime
import logging

from connexion.resolver import RestyResolver

from connexion import NoContent

logging.basicConfig(level=logging.INFO)
app = connexion.App(__name__)

app.add_api('swagger.yaml', resolver=RestyResolver('api'))

if __name__ == '__main__':
    app.run(port=8080)
